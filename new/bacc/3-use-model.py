import numpy as np
import os
import librosa
import soundfile as sf
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import argparse

# PARAMETERS
SAMPLE_RATE = 8000  # Target sample rate (resampling from 16kHz)
N_FFT = 256  # FFT size (smaller for 8kHz audio)
HOP_LENGTH = 64  # Hop length
SEGMENT_DURATION = 1.0  # Duration of audio segments in seconds
SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)
MODEL_DIR = "./models"
OUTPUT_DIR = "./enhanced"

class SafeConcatenate(tf.keras.layers.Layer):
    def __init__(self, axis=-1):
        super(SafeConcatenate, self).__init__()
        self.axis = axis
        self.concat = tf.keras.layers.Concatenate(axis=axis)

    def call(self, inputs):
        x1, x2 = inputs
        # Resize x1 to match x2's spatial dimensions
        x1_resized = tf.image.resize(x1, [tf.shape(x2)[1], tf.shape(x2)[2]])
        return self.concat([x1_resized, x2])

    def get_config(self):
        config = super(SafeConcatenate, self).get_config()
        config.update({'axis': self.axis})
        return config


# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Do not allocate all GPU memory at once
if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(
        tf.config.experimental.list_physical_devices('GPU')[0], True)


def prepare_complex_specs(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convert audio to normalized complex spectrograms"""
    # Compute STFT
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Split into real and imaginary components
    real = np.real(spec)
    imag = np.imag(spec)

    # Normalize both components to [-1, 1] range
    real_max = np.max(np.abs(real))
    imag_max = np.max(np.abs(imag))

    real_norm = real / (real_max + 1e-8)
    imag_norm = imag / (imag_max + 1e-8)

    return real_norm, imag_norm, (real_max, imag_max)


def segment_audio(audio, segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_SAMPLES // 4):
    """Segment audio into overlapping chunks with 75% overlap for smoother reconstruction"""
    segments = []
    positions = []

    for i in range(0, len(audio) - segment_samples + 1, hop_samples):
        segment = audio[i:i + segment_samples]
        segments.append(segment)
        positions.append(i)

    # Handle last segment if needed
    if len(audio) > segments[-1].shape[0] + positions[-1]:
        last_segment = audio[-segment_samples:]
        segments.append(last_segment)
        positions.append(len(audio) - segment_samples)

    return segments, positions


def apply_mask_to_spec(noisy_spec, mask):
    """Apply complex mask to noisy spectrogram"""
    # Rebuild complex mask from real and imaginary parts
    complex_mask = mask[..., 0] + 1j * mask[..., 1]

    # Apply mask to noisy spectrogram
    return noisy_spec * complex_mask


def find_latest_model(model_dir):
    """Find the most recent model in the model directory"""
    model_files = glob(os.path.join(model_dir, "*.keras"))
    if not model_files:
        raise FileNotFoundError(f"No .keras model files found in {model_dir}")

    # Sort by modification time, newest first
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]


def enhance_audio(model, audio_path, output_path=None):
    """Enhance noisy audio using the trained model"""
    print(f"Processing {audio_path}...")

    # Load audio
    noisy_audio, orig_sr = librosa.load(audio_path, sr=None)

    # Resample if needed
    if orig_sr != SAMPLE_RATE:
        print(f"Resampling from {orig_sr} to {SAMPLE_RATE} Hz")
        noisy_audio = librosa.resample(
            noisy_audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)

    # Segment audio for processing
    audio_segments, positions = segment_audio(noisy_audio)
    print(f"Processing {len(audio_segments)} segments")

    # Create output buffer with zeros
    enhanced_audio = np.zeros_like(noisy_audio)
    overlap_count = np.zeros_like(noisy_audio)

    # Process each segment
    for i, (segment, pos) in enumerate(tqdm(zip(audio_segments, positions), total=len(audio_segments))):
        # Convert to complex spectrogram
        noisy_spec = librosa.stft(segment, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Prepare normalized inputs for model
        noisy_real, noisy_imag, scaling_factors = prepare_complex_specs(
            segment)

        # Stack real and imaginary parts as channels
        model_input = np.stack([noisy_real, noisy_imag], axis=-1)
        model_input = np.expand_dims(
            model_input, axis=0)  # Add batch dimension

        # Predict mask
        mask = model.predict(model_input, verbose=0)[
            0]  # Remove batch dimension

        # Apply mask to original spectrogram
        enhanced_spec = apply_mask_to_spec(noisy_spec, mask)

        # Convert back to time domain
        segment_enhanced = librosa.istft(
            enhanced_spec, hop_length=HOP_LENGTH, win_length=N_FFT)

        # Add to output buffer with overlap
        segment_len = min(len(segment_enhanced), SEGMENT_SAMPLES)
        segment_end = min(pos + segment_len, len(enhanced_audio))
        enhanced_audio[pos:segment_end] += segment_enhanced[:segment_end - pos]
        overlap_count[pos:segment_end] += 1

    # Average overlapped regions
    overlap_count[overlap_count == 0] = 1  # Avoid division by zero
    enhanced_audio /= overlap_count

    # Resample back to original sample rate if needed
    if orig_sr != SAMPLE_RATE:
        enhanced_audio = librosa.resample(
            enhanced_audio, orig_sr=SAMPLE_RATE, target_sr=orig_sr)

    # Determine output path
    if output_path is None:
        base_name = os.path.basename(audio_path)
        output_path = os.path.join(OUTPUT_DIR, f"enhanced_{base_name}")

    # Save enhanced audio
    sf.write(output_path, enhanced_audio, orig_sr)
    print(f"Enhanced audio saved to {output_path}")

    return enhanced_audio


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Enhance audio using CNN denoiser")
    parser.add_argument("input_file", help="Path to input noisy WAV file")
    parser.add_argument(
        "--output", "-o", help="Path to output enhanced WAV file")
    parser.add_argument("--model", "-m", help="Path to model file")
    args = parser.parse_args()

    # Find model file
    model_path = args.model if args.model else find_latest_model(MODEL_DIR)
    print(f"Using model: {model_path}")

    # Load model with custom objects
    custom_objects = {
        'complex_mse_loss': lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred)),
        'snr_improvement': lambda y_true, y_pred: -10 * tf.math.log(tf.reduce_mean(tf.square(y_true - y_pred)) + 1e-10) / tf.math.log(10.0),
        'SafeConcatenate': SafeConcatenate  # Just pass the class itself
    }

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fall back to loading without compilation if needed
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # Enhance audio
    enhance_audio(model, args.input_file, args.output)


if __name__ == "__main__":
    main()
