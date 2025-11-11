import super_utils as sutils
import librosa
import numpy as np


def phase_aware_scaling(clean_spectral_magnitude, clean_phase, noise_phase):
    assert clean_phase.shape == noise_phase.shape, "Shapes must match."
    return clean_spectral_magnitude * np.cos(clean_phase - noise_phase)

def denormalize_and_reconstruct(predicted_magnitude, noisy_phase, original_min, original_max, reference_audio=None):
    
    # Undo min-max normalization
    predicted_magnitude = predicted_magnitude * (original_max - original_min) + original_min

    # Undo log scaling
    # eps = 1e-10
    predicted_magnitude = np.exp(predicted_magnitude)

    # Scale back to a reasonable range based on input characteristics
    scale_factor = np.exp(original_max)  # This represents our expected maximum
    predicted_magnitude = predicted_magnitude / predicted_magnitude.max() * scale_factor

    # Combine magnitude with phase
    complex_spec = predicted_magnitude * np.exp(1j * noisy_phase)

    # Inverse STFT (assuming your sutils has this function)
    reconstructed_audio = sutils.istft_only(complex_spec, noisy_phase)

    # Match RMS of reference audio if provided
    if reference_audio is not None:
        ref_rms = np.sqrt(np.mean(reference_audio**2))
        current_rms = np.sqrt(np.mean(reconstructed_audio**2))
        reconstructed_audio = reconstructed_audio * (ref_rms / current_rms)

    return reconstructed_audio

clean_file = 'clnsp1.wav'
noisy_file = 'noisy.wav'

clean_audio, sr = librosa.load(clean_file, sr=16000)
clean_magnitude, clean_phase = sutils.preprocess(clean_audio, sr)


noisy_audio, sr = librosa.load(noisy_file, sr=16000)
noisy_magnitude, noisy_phase = sutils.preprocess(noisy_audio, sr)

# phase aware scaling
clean_magnitude = phase_aware_scaling(clean_magnitude, clean_phase, noisy_phase)


# Log-magnitude scaling (adding small constant to avoid log(0))
eps = 1e-10
noisy_magnitude = np.maximum(noisy_magnitude, eps)
clean_magnitude = np.maximum(clean_magnitude, eps)

noisy_magnitude = np.log(noisy_magnitude + eps)
clean_magnitude = np.log(clean_magnitude + eps)

# ##### get min and max for denormalization

noisy_magnitude_min = noisy_magnitude.min()
noisy_magnitude_max = noisy_magnitude.max()

clean_magnitude_min = clean_magnitude.min()
clean_magnitude_max = clean_magnitude.max()

# ########

# Optional: normalize to [0,1] if your CNN expects this range
noisy_magnitude = (noisy_magnitude - noisy_magnitude.min()) / (noisy_magnitude.max() - noisy_magnitude.min())
clean_magnitude = (clean_magnitude - clean_magnitude.min()) / (clean_magnitude.max() - clean_magnitude.min())

# feito o brique

predicted_magnitude = clean_magnitude  # a saida, num mundo perfeito

reconstructed_audio = denormalize_and_reconstruct(
    predicted_magnitude,
    noisy_phase,
    noisy_magnitude_min,
    noisy_magnitude_max,
    reference_audio=noisy_audio
)

# reconstructed_audio = denormalize_and_reconstruct(
#     predicted_magnitude,
#     clean_phase,
#     clean_magnitude_min,
#     clean_magnitude_max
# )

# Save the reconstructed audio
sutils.write_audio(reconstructed_audio, 'reconstructed_v2.wav')

# Add this before writing the audio file
print("Original clean magnitude range:", np.abs(clean_audio).min(), np.abs(clean_audio).max())
print("Noisy magnitude range:", np.abs(noisy_audio).min(), np.abs(noisy_audio).max())
print("Reconstructed magnitude range:", np.abs(reconstructed_audio).min(), np.abs(reconstructed_audio).max())