"""
Audio Denoiser - Inference (v2)

This script applies the trained denoiser model to audio files.

Key features:
1. Processes audio in overlapping segments
2. Uses overlap-add for seamless reconstruction
3. Applies complex ratio mask to noisy spectrogram
4. Proper normalization using saved statistics
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    Config, load_audio, save_audio, compute_stft, compute_istft,
    stft_to_features, features_to_stft, normalize_spectrogram,
    denormalize_spectrogram, segment_audio, overlap_add_segments,
    compute_sdr, setup_gpu
)
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np

# Import custom layers from training script for model loading
# These are needed to deserialize the saved model
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class ConvBlock(layers.Layer):
    """Convolutional block with BatchNorm and activation"""

    def __init__(self, filters, kernel_size=3, strides=1, activation=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_activation = activation

        self.conv = layers.Conv2D(
            filters, kernel_size, strides=strides, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )
        self.bn = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(negative_slope=0.2)

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.use_activation:
            x = self.activation(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': self.use_activation
        })
        return config


@tf.keras.utils.register_keras_serializable()
class ResidualBlock(layers.Layer):
    """Residual block for stable deep training"""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = ConvBlock(filters, kernel_size=3)
        self.conv2 = ConvBlock(filters, kernel_size=3, activation=False)
        self.activation = layers.LeakyReLU(negative_slope=0.2)

    def call(self, x, training=None):
        residual = x
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = layers.Add()([x, residual])
        x = self.activation(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


@tf.keras.utils.register_keras_serializable()
class ResizeToMatch(layers.Layer):
    """Resize layer to match target spatial dimensions"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        x, reference = inputs
        target_size = (tf.shape(reference)[1], tf.shape(reference)[2])
        return tf.image.resize(x, target_size)


@tf.keras.utils.register_keras_serializable()
class AttentionGate(layers.Layer):
    """Attention gate for skip connections"""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.W_g = layers.Conv2D(filters, 1, padding='same')
        self.W_x = layers.Conv2D(filters, 1, padding='same')
        self.psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')
        self.bn = layers.BatchNormalization()
        self.resize = ResizeToMatch()

    def call(self, g, x, training=None):
        g_upsampled = self.resize([g, x])
        g1 = self.W_g(g_upsampled)
        x1 = self.W_x(x)
        psi = tf.nn.relu(g1 + x1)
        psi = self.psi(psi)
        psi = tf.cast(psi, x.dtype)
        return x * psi
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


@tf.keras.utils.register_keras_serializable()
class SafeDropout(layers.Layer):
    """Dropout computed in float32, cast back to input dtype."""

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)
        self.dropout = layers.Dropout(self.rate)

    def call(self, x, training=None):
        if self.rate <= 0.0:
            return x
        x32 = tf.cast(x, tf.float32)
        y32 = self.dropout(x32, training=training)
        return tf.cast(y32, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config


class AudioDenoiser:
    """
    Audio denoiser class that handles model loading and inference
    """

    def __init__(self, model_path):
        """
        Initialize denoiser with trained model

        Args:
            model_path: Path to trained .keras model file
        """
        setup_gpu()

        # Load normalization statistics
        if not Config.load_norm_stats():
            print("Warning: Could not load normalization stats, using defaults")

        print(f"Normalization stats: max_mag={Config.GLOBAL_MAX_MAG:.4f}")

        # Load model with custom objects
        print(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'ConvBlock': ConvBlock,
                'ResidualBlock': ResidualBlock,
                'ResizeToMatch': ResizeToMatch,
                'AttentionGate': AttentionGate,
                'SafeDropout': SafeDropout,
            },
            compile=False  # Don't need to compile for inference
        )
        print("Model loaded successfully")

        # Warm up model with dummy input
        dummy_input = np.zeros(
            (1, Config.N_FREQ_BINS, Config.N_TIME_FRAMES, 2), dtype=np.float32)
        _ = self.model.predict(dummy_input, verbose=0)
        print("Model warmed up")

    def process_segment(self, audio_segment):
        """
        Process a single audio segment through the model

        Args:
            audio_segment: Audio array of length SEGMENT_SAMPLES

        Returns:
            Denoised audio segment
        """
        # Compute STFT
        noisy_stft = compute_stft(audio_segment)

        # Store original shape
        orig_shape = noisy_stft.shape

        # Convert to features
        noisy_real, noisy_imag = stft_to_features(noisy_stft)

        # Normalize
        noisy_real_norm, noisy_imag_norm = normalize_spectrogram(
            noisy_real, noisy_imag)

        # Pad if needed to match expected dimensions
        target_time = Config.N_TIME_FRAMES
        current_time = noisy_real_norm.shape[1]

        if current_time < target_time:
            pad_width = target_time - current_time
            noisy_real_norm = np.pad(noisy_real_norm, ((0, 0), (0, pad_width)))
            noisy_imag_norm = np.pad(noisy_imag_norm, ((0, 0), (0, pad_width)))
        elif current_time > target_time:
            noisy_real_norm = noisy_real_norm[:, :target_time]
            noisy_imag_norm = noisy_imag_norm[:, :target_time]

        # Stack into input tensor
        model_input = np.stack([noisy_real_norm, noisy_imag_norm], axis=-1)
        model_input = np.expand_dims(
            model_input, axis=0)  # Add batch dimension
        model_input = model_input.astype(np.float32)

        # Predict mask
        predicted_mask = self.model.predict(model_input, verbose=0)

        # Remove batch dimension
        predicted_mask = predicted_mask[0]

        # Trim back to original time dimension
        predicted_mask = predicted_mask[:, :orig_shape[1], :]

        # Extract mask components
        mask_real = predicted_mask[:, :, 0]
        mask_imag = predicted_mask[:, :, 1]

        # Apply complex mask to noisy spectrogram
        # Clean = Noisy * Mask
        mask_complex = mask_real + 1j * mask_imag
        clean_stft = noisy_stft * mask_complex

        # Inverse STFT
        clean_audio = compute_istft(clean_stft)

        return clean_audio

    def denoise_audio(self, audio, show_progress=True):
        """
        Denoise an audio signal

        Args:
            audio: Input audio array (any length)
            show_progress: Whether to show progress bar

        Returns:
            Denoised audio array
        """
        original_length = len(audio)

        # Segment audio
        segments = segment_audio(
            audio,
            segment_samples=Config.SEGMENT_SAMPLES,
            hop_samples=Config.SEGMENT_HOP
        )

        if not segments:
            # Audio is shorter than segment size, pad it
            padded = np.pad(audio, (0, Config.SEGMENT_SAMPLES - len(audio)))
            clean_audio = self.process_segment(padded)
            return clean_audio[:original_length]

        # Process each segment
        processed_segments = []

        iterator = tqdm(
            segments, desc="Processing") if show_progress else segments

        for start_idx, segment in iterator:
            clean_segment = self.process_segment(segment)
            processed_segments.append((start_idx, clean_segment))

        # Overlap-add reconstruction
        clean_audio = overlap_add_segments(
            processed_segments,
            Config.SEGMENT_SAMPLES,
            Config.SEGMENT_HOP,
            original_length
        )

        return clean_audio

    def denoise_file(self, input_path, output_path, show_progress=True):
        """
        Denoise an audio file and save result

        Args:
            input_path: Path to input audio file
            output_path: Path to save denoised audio
            show_progress: Whether to show progress bar
        """
        # Load audio
        audio, sr = load_audio(input_path)
        print(f"Loaded {input_path}: {len(audio)/sr:.2f}s at {sr}Hz")

        # Denoise
        clean_audio = self.denoise_audio(audio, show_progress=show_progress)

        # Save
        save_audio(clean_audio, output_path)
        print(f"Saved denoised audio to {output_path}")

        return audio, clean_audio


def evaluate_denoiser(denoiser, clean_dir, noisy_dir, num_samples=10):
    """
    Evaluate denoiser on test samples

    Returns average SDR improvement
    """
    from config import get_file_pairs

    pairs = get_file_pairs(clean_dir, noisy_dir)

    if num_samples:
        np.random.shuffle(pairs)
        pairs = pairs[:num_samples]

    sdr_improvements = []

    for clean_path, noisy_path, snr in tqdm(pairs, desc="Evaluating"):
        try:
            # Load audio
            clean_audio, _ = load_audio(clean_path)
            noisy_audio, _ = load_audio(noisy_path)

            # Ensure same length
            min_len = min(len(clean_audio), len(noisy_audio))
            clean_audio = clean_audio[:min_len]
            noisy_audio = noisy_audio[:min_len]

            # Denoise
            denoised_audio = denoiser.denoise_audio(
                noisy_audio, show_progress=False)

            # Ensure same length for SDR calculation
            min_len = min(len(clean_audio), len(denoised_audio))

            # Compute SDR
            sdr_before = compute_sdr(
                clean_audio[:min_len], noisy_audio[:min_len])
            sdr_after = compute_sdr(
                clean_audio[:min_len], denoised_audio[:min_len])
            sdr_improvement = sdr_after - sdr_before

            sdr_improvements.append({
                'file': os.path.basename(noisy_path),
                'snr': snr,
                'sdr_before': sdr_before,
                'sdr_after': sdr_after,
                'improvement': sdr_improvement
            })

        except Exception as e:
            print(f"Error processing {noisy_path}: {e}")
            continue

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for result in sdr_improvements:
        print(f"{result['file']}: SDR {result['sdr_before']:.2f} -> {result['sdr_after']:.2f} dB "
              f"(+{result['improvement']:.2f} dB)")

    avg_improvement = np.mean([r['improvement'] for r in sdr_improvements])
    print("-" * 60)
    print(f"Average SDR improvement: {avg_improvement:.2f} dB")
    print("=" * 60)

    return sdr_improvements, avg_improvement


def process_directory(denoiser, input_dir, output_dir):
    """
    Process all WAV files in a directory
    """
    os.makedirs(output_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    print(f"Found {len(wav_files)} WAV files to process")

    for wav_file in tqdm(wav_files, desc="Processing files"):
        input_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, f"denoised_{wav_file}")

        try:
            denoiser.denoise_file(input_path, output_path, show_progress=False)
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Denoiser Inference')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Single file denoising
    file_parser = subparsers.add_parser('file', help='Denoise a single file')
    file_parser.add_argument('input', help='Input audio file')
    file_parser.add_argument('output', help='Output audio file')
    file_parser.add_argument('--model', default=None,
                             help='Path to model file')

    # Directory processing
    dir_parser = subparsers.add_parser(
        'dir', help='Process a directory of files')
    dir_parser.add_argument('input_dir', help='Input directory')
    dir_parser.add_argument('output_dir', help='Output directory')
    dir_parser.add_argument('--model', default=None, help='Path to model file')

    # Evaluation
    eval_parser = subparsers.add_parser('eval', help='Evaluate denoiser')
    eval_parser.add_argument(
        '--clean-dir', default=Config.CLEAN_DIR, help='Clean audio directory')
    eval_parser.add_argument(
        '--noisy-dir', default=Config.NOISY_DIR, help='Noisy audio directory')
    eval_parser.add_argument('--samples', type=int,
                             default=20, help='Number of samples to evaluate')
    eval_parser.add_argument('--model', default=None,
                             help='Path to model file')

    args = parser.parse_args()

    # Find model if not specified
    if args.command and (args.model is None or not os.path.exists(args.model)):
        # Look for best model in model directory
        model_dir = Config.MODEL_DIR
        possible_models = [
            os.path.join(model_dir, 'denoiser_v2-best.keras'),
            os.path.join(model_dir, 'denoiser_v2-final.keras'),
        ]

        for model_path in possible_models:
            if os.path.exists(model_path):
                args.model = model_path
                break

        if args.model is None:
            print("Error: No model file found. Please specify with --model")
            exit(1)

    # Execute command
    if args.command == 'file':
        denoiser = AudioDenoiser(args.model)
        denoiser.denoise_file(args.input, args.output)

    elif args.command == 'dir':
        denoiser = AudioDenoiser(args.model)
        process_directory(denoiser, args.input_dir, args.output_dir)

    elif args.command == 'eval':
        denoiser = AudioDenoiser(args.model)
        evaluate_denoiser(
            denoiser,
            args.clean_dir,
            args.noisy_dir,
            num_samples=args.samples
        )

    else:
        parser.print_help()
