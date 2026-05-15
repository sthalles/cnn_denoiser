# import tensorflow as tf
import numpy as np
# import os
import librosa
import soundfile as sf


def create_complex_mask(noisy_spec, clean_spec):
    """Create complex ratio mask"""
    mask_real = (noisy_spec.real * clean_spec.real +
                 noisy_spec.imag * clean_spec.imag) / (np.abs(noisy_spec)**2 + 1e-8)
    mask_imag = (noisy_spec.real * clean_spec.imag -
                 noisy_spec.imag * clean_spec.real) / (np.abs(noisy_spec)**2 + 1e-8)
    return mask_real + 1j * mask_imag

def apply_complex_mask(noisy_spec, mask):
    """Apply complex mask to noisy spectrogram"""
    return noisy_spec * mask

def reconstruct_with_mask(noisy_audio, predicted_mask, n_fft=512, hop_length=128):
    """Complete pipeline using mask-based enhancement"""
    # Get noisy spectrogram
    noisy_spec = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)

    # Apply mask
    enhanced_spec = apply_complex_mask(noisy_spec, predicted_mask)

    # Reconstruct
    return librosa.istft(enhanced_spec, hop_length=hop_length, win_length=n_fft)

def prepare_complex_specs(audio, n_fft=512, hop_length=128):
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

def prepare_mask_targets(noisy_spec, clean_spec):
    """Create normalized mask targets for training"""
    # Create complex mask
    mask = create_complex_mask(noisy_spec, clean_spec)

    # Split mask into real and imaginary components
    mask_real = np.real(mask)
    mask_imag = np.imag(mask)

    # Clip mask values to prevent extreme values
    mask_real = np.clip(mask_real, -1, 1)
    mask_imag = np.clip(mask_imag, -1, 1)

    return mask_real, mask_imag


def main():
    # Example usage
    clean_file = 'clnsp1.wav'
    noisy_file = 'noisy.wav'
    sr = 16000
    n_fft = 512
    hop_length = 128

    # Load audio files
    noisy_audio, _ = librosa.load(noisy_file, sr=sr)
    clean_audio, _ = librosa.load(clean_file, sr=sr)

    # Ensure same length
    min_len = min(len(noisy_audio), len(clean_audio))
    noisy_audio = noisy_audio[:min_len]
    clean_audio = clean_audio[:min_len]

    # Convert to frequency domain (STFT)
    noisy_spec = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)
    clean_spec = librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length)

    # Generate ideal mask (this would be predicted by a CNN in practice)
    mask = create_complex_mask(noisy_spec, clean_spec)

    # Prepare normalized spectrograms (as would be done for CNN training)
    noisy_real, noisy_imag, noisy_scaling = prepare_complex_specs(noisy_audio, n_fft, hop_length)

    # Prepare normalized mask targets (as expected from CNN output)
    mask_real, mask_imag = prepare_mask_targets(noisy_spec, clean_spec)

    # SIMULATION: Convert normalized mask back to complex form (as if from CNN)
    # In a real CNN application, mask_real and mask_imag would be the network outputs
    predicted_mask = mask_real + 1j * mask_imag

    # Apply mask to denoise in spectral domain
    enhanced_spec = apply_complex_mask(noisy_spec, predicted_mask)

    # Reconstruct time-domain audio
    enhanced_audio = librosa.istft(enhanced_spec, hop_length=hop_length, win_length=n_fft)

    # Ensure all audio arrays are the same length for comparison
    min_len = min(len(noisy_audio), len(clean_audio), len(enhanced_audio))
    noisy_audio = noisy_audio[:min_len]
    clean_audio = clean_audio[:min_len]
    enhanced_audio = enhanced_audio[:min_len]

    # Save files
    sf.write('original_noisy.wav', noisy_audio, sr)
    sf.write('enhanced_ideal.wav', enhanced_audio, sr)
    sf.write('original_clean.wav', clean_audio, sr)

    # Calculate SNR improvement for evaluation
    noise_before = noisy_audio - clean_audio
    noise_after = enhanced_audio - clean_audio

    snr_before = 10 * np.log10(np.sum(clean_audio**2) / (np.sum(noise_before**2) + 1e-10))
    snr_after = 10 * np.log10(np.sum(clean_audio**2) / (np.sum(noise_after**2) + 1e-10))

    print(f"SNR before enhancement: {snr_before:.2f} dB")
    print(f"SNR after enhancement: {snr_after:.2f} dB")
    print(f"SNR improvement: {snr_after - snr_before:.2f} dB")

    print(f"Audio files saved: original_noisy.wav, enhanced_ideal.wav, original_clean.wav")


main()
