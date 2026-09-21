"""
Audio Denoiser - Shared Configuration and Utilities
This module contains all shared constants, configurations, and utility functions
used across dataset creation, training, and inference.
"""

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from typing import Tuple, Optional
import os

# =============================================================================
# CONFIGURATION - All parameters in one place
# =============================================================================


class Config:
    """Configuration class for audio denoiser"""

    # Audio parameters
    SAMPLE_RATE = 16000  # Keep at 16kHz for better speech quality
    N_FFT = 512          # 512-point FFT (32ms window at 16kHz)
    HOP_LENGTH = 128     # 8ms hop (75% overlap)
    WIN_LENGTH = 512     # Same as N_FFT
    WINDOW = 'hann'      # Hann window for better frequency resolution

    # Derived values
    N_FREQ_BINS = N_FFT // 2 + 1  # 257 frequency bins

    # Segment parameters (for TFRecords)
    SEGMENT_DURATION = 2.0  # 2 seconds per segment
    SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)
    SEGMENT_HOP = int(SEGMENT_SAMPLES * 0.5)  # 50% overlap between segments

    # Number of time frames per segment
    N_TIME_FRAMES = SEGMENT_SAMPLES // HOP_LENGTH + 1  # ~251 frames for 2s

    # Training parameters
    BATCH_SIZE = 128     # Good for RTX 4070
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    MIN_LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 1e-4

    # Dataset paths
    CLEAN_DIR = 'dataset/CleanSpeech_training'
    NOISY_DIR = 'dataset/NoisySpeech_training'
    TFRECORD_DIR = 'tfrecords_v2'
    MODEL_DIR = 'models'
    LOG_DIR = 'logs_v2'

    # Normalization (computed from dataset, updated during preprocessing)
    # These are reasonable defaults that will be overwritten
    GLOBAL_MEAN = 0.0
    GLOBAL_STD = 1.0
    GLOBAL_MAX_MAG = 100.0  # Will be computed from data

    @classmethod
    def get_norm_stats_path(cls):
        return os.path.join(cls.TFRECORD_DIR, 'norm_stats.npz')

    @classmethod
    def save_norm_stats(cls, mean, std, max_mag):
        """Save normalization statistics"""
        os.makedirs(cls.TFRECORD_DIR, exist_ok=True)
        np.savez(cls.get_norm_stats_path(),
                 mean=mean, std=std, max_mag=max_mag)
        cls.GLOBAL_MEAN = mean
        cls.GLOBAL_STD = std
        cls.GLOBAL_MAX_MAG = max_mag

    @classmethod
    def load_norm_stats(cls):
        """Load normalization statistics"""
        path = cls.get_norm_stats_path()
        if os.path.exists(path):
            data = np.load(path)
            cls.GLOBAL_MEAN = float(data['mean'])
            cls.GLOBAL_STD = float(data['std'])
            cls.GLOBAL_MAX_MAG = float(data['max_mag'])
            return True
        return False


# =============================================================================
# AUDIO I/O
# =============================================================================

def load_audio(filepath: str, sr: int = Config.SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate"""
    audio, orig_sr = librosa.load(filepath, sr=sr)
    return audio, sr


def save_audio(audio: np.ndarray, filepath: str, sr: int = Config.SAMPLE_RATE):
    """Save audio to file"""
    # Ensure audio is in valid range
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(filepath, audio, sr, subtype='PCM_16')


# =============================================================================
# STFT / ISTFT
# =============================================================================

def compute_stft(audio: np.ndarray) -> np.ndarray:
    """
    Compute STFT of audio signal

    Returns:
        Complex spectrogram of shape (n_freq_bins, n_frames)
    """
    stft = librosa.stft(
        audio,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        win_length=Config.WIN_LENGTH,
        window=Config.WINDOW,
        center=True
    )
    return stft


def compute_istft(stft: np.ndarray) -> np.ndarray:
    """
    Compute inverse STFT to reconstruct audio

    Args:
        stft: Complex spectrogram of shape (n_freq_bins, n_frames)

    Returns:
        Audio signal
    """
    audio = librosa.istft(
        stft,
        hop_length=Config.HOP_LENGTH,
        win_length=Config.WIN_LENGTH,
        window=Config.WINDOW,
        center=True
    )
    return audio


# =============================================================================
# COMPLEX SPECTROGRAM OPERATIONS
# =============================================================================

def stft_to_features(stft: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert complex STFT to real/imag features for neural network

    Args:
        stft: Complex spectrogram (n_freq, n_time)

    Returns:
        real: Real part normalized
        imag: Imaginary part normalized
    """
    real = np.real(stft)
    imag = np.imag(stft)
    return real, imag


def features_to_stft(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """
    Convert real/imag features back to complex STFT

    Args:
        real: Real part
        imag: Imaginary part

    Returns:
        Complex spectrogram
    """
    return real + 1j * imag


def normalize_spectrogram(real: np.ndarray, imag: np.ndarray,
                          max_mag: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize spectrogram using global maximum magnitude

    This preserves phase information and uses consistent scaling
    """
    if max_mag is None:
        max_mag = Config.GLOBAL_MAX_MAG

    real_norm = real / (max_mag + 1e-8)
    imag_norm = imag / (max_mag + 1e-8)

    return real_norm, imag_norm


def denormalize_spectrogram(real: np.ndarray, imag: np.ndarray,
                            max_mag: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denormalize spectrogram back to original scale
    """
    if max_mag is None:
        max_mag = Config.GLOBAL_MAX_MAG

    real_denorm = real * max_mag
    imag_denorm = imag * max_mag

    return real_denorm, imag_denorm


# =============================================================================
# COMPLEX RATIO MASK (cRM)
# =============================================================================

def compute_complex_mask(noisy_stft: np.ndarray, clean_stft: np.ndarray) -> np.ndarray:
    """
    Compute Complex Ratio Mask: M = Clean / Noisy

    This mask, when applied to noisy signal, recovers the clean signal:
    Clean = Noisy * M

    Args:
        noisy_stft: Complex noisy spectrogram
        clean_stft: Complex clean spectrogram

    Returns:
        Complex mask
    """
    # Avoid division by zero
    eps = 1e-8
    noisy_power = np.abs(noisy_stft) ** 2 + eps

    # cRM formula: M = (Clean * conj(Noisy)) / |Noisy|^2
    mask = (clean_stft * np.conj(noisy_stft)) / noisy_power

    return mask


def apply_complex_mask(noisy_stft: np.ndarray, mask_real: np.ndarray,
                       mask_imag: np.ndarray) -> np.ndarray:
    """
    Apply complex mask to noisy spectrogram

    Args:
        noisy_stft: Complex noisy spectrogram
        mask_real: Real part of mask
        mask_imag: Imaginary part of mask

    Returns:
        Estimated clean spectrogram
    """
    mask = mask_real + 1j * mask_imag
    return noisy_stft * mask


# =============================================================================
# SEGMENTATION
# =============================================================================

def segment_audio(audio: np.ndarray,
                  segment_samples: int = Config.SEGMENT_SAMPLES,
                  hop_samples: int = Config.SEGMENT_HOP) -> list:
    """
    Segment audio into overlapping chunks

    Returns list of (start_idx, segment) tuples
    """
    segments = []
    for start in range(0, len(audio) - segment_samples + 1, hop_samples):
        segment = audio[start:start + segment_samples]
        segments.append((start, segment))

    # Handle last segment with padding if needed
    if len(audio) > segment_samples:
        last_start = len(segments[-1][1]) if segments else 0
        remaining = len(audio) - last_start
        if remaining > hop_samples and remaining < segment_samples:
            # Pad the last segment
            last_segment = audio[-segment_samples:]
            if len(last_segment) < segment_samples:
                last_segment = np.pad(
                    last_segment, (0, segment_samples - len(last_segment)))
            segments.append((len(audio) - segment_samples, last_segment))

    return segments


def overlap_add_segments(segments: list, segment_samples: int,
                         hop_samples: int, total_length: int) -> np.ndarray:
    """
    Reconstruct audio from overlapping segments using overlap-add

    Args:
        segments: List of (start_idx, audio_segment) tuples
        segment_samples: Length of each segment
        hop_samples: Hop between segments
        total_length: Target output length

    Returns:
        Reconstructed audio
    """
    output = np.zeros(total_length)
    weights = np.zeros(total_length)

    # Hann window for smooth blending
    window = np.hanning(segment_samples)

    for start_idx, segment in segments:
        end_idx = min(start_idx + segment_samples, total_length)
        seg_len = end_idx - start_idx

        output[start_idx:end_idx] += segment[:seg_len] * window[:seg_len]
        weights[start_idx:end_idx] += window[:seg_len]

    # Normalize by weights (avoid division by zero)
    weights = np.maximum(weights, 1e-8)
    output /= weights

    return output


# =============================================================================
# METRICS
# =============================================================================

def compute_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """
    Compute Signal-to-Distortion Ratio in dB

    SDR = 10 * log10(||reference||^2 / ||reference - estimated||^2)
    """
    reference = reference.flatten()
    estimated = estimated.flatten()

    # Ensure same length
    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]

    # Compute SDR
    noise = reference - estimated
    ref_power = np.sum(reference ** 2) + 1e-8
    noise_power = np.sum(noise ** 2) + 1e-8

    sdr = 10 * np.log10(ref_power / noise_power)
    return sdr


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio in dB
    """
    signal_power = np.mean(signal ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# =============================================================================
# TENSORFLOW UTILITIES
# =============================================================================

def setup_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Enable mixed precision for tensor cores
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

            print(f"GPU(s) configured: {[gpu.name for gpu in gpus]}")
            print("Mixed precision enabled (FP16)")

        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")


def get_strategy():
    """Get distribution strategy for training"""
    # Single GPU strategy
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        return tf.distribute.OneDeviceStrategy("/gpu:0")
    return tf.distribute.get_strategy()  # Default strategy


# =============================================================================
# DATASET UTILITIES
# =============================================================================

def get_file_pairs(clean_dir: str, noisy_dir: str) -> list:
    """
    Get matching pairs of clean and noisy audio files

    Returns:
        List of (clean_path, noisy_path, snr) tuples
    """
    import re

    pairs = []
    clean_files = sorted(
        [f for f in os.listdir(clean_dir) if f.endswith('.wav')])
    noisy_files = sorted(
        [f for f in os.listdir(noisy_dir) if f.endswith('.wav')])

    # Create lookup for noisy files
    noisy_lookup = {}
    for nf in noisy_files:
        # Parse: noisy1_SNRdb_0.0_clnsp1.wav
        match = re.match(r'noisy\d+_SNRdb_([\d.]+)_(.+)\.wav', nf)
        if match:
            snr = float(match.group(1))
            base_name = match.group(2)  # e.g., "clnsp1"
            if base_name not in noisy_lookup:
                noisy_lookup[base_name] = []
            noisy_lookup[base_name].append((nf, snr))

    # Match clean files with noisy files
    for cf in clean_files:
        base_name = os.path.splitext(cf)[0]  # e.g., "clnsp1"
        if base_name in noisy_lookup:
            for nf, snr in noisy_lookup[base_name]:
                pairs.append((
                    os.path.join(clean_dir, cf),
                    os.path.join(noisy_dir, nf),
                    snr
                ))

    return pairs


if __name__ == "__main__":
    # Quick test
    print("Config test:")
    print(f"  Sample rate: {Config.SAMPLE_RATE}")
    print(f"  FFT size: {Config.N_FFT}")
    print(f"  Freq bins: {Config.N_FREQ_BINS}")
    print(f"  Time frames per segment: {Config.N_TIME_FRAMES}")
    print(f"  Segment duration: {Config.SEGMENT_DURATION}s")

    # Test STFT
    test_audio = np.random.randn(Config.SEGMENT_SAMPLES)
    stft = compute_stft(test_audio)
    print(f"  STFT shape: {stft.shape}")

    reconstructed = compute_istft(stft)
    print(
        f"  Reconstruction error: {np.mean(np.abs(test_audio - reconstructed[:len(test_audio)])):.6f}")
