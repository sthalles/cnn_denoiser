import numpy as np
import librosa
import soundfile as sf

num_features = 129
window_length = 256
hop_length = 64
sample_rate = 16000

def phase_aware_scaling(clean_spectral_magnitude, clean_phase, noise_phase):
    assert clean_phase.shape == noise_phase.shape, "Shapes must match."
    return clean_spectral_magnitude * np.cos(clean_phase - noise_phase)


# separar stft em segmentos sobrepostos
def prepare_input_features(stft_features, numSegments, numFeatures):
    noisySTFT = np.concatenate([stft_features[:, 0:numSegments - 1], stft_features], axis=1)
    stftSegments = np.zeros((numFeatures, numSegments, noisySTFT.shape[1] - numSegments + 1))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:, :, index] = noisySTFT[:, index:index + numSegments]
    return stftSegments

def preprocess(audio, sr, n_fft=window_length, hop_length=hop_length, window='hamming', write_name=''):

    # Resample from 16k to 8k
    if sr == 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=8000)
        sr = 8000

    if write_name != '':
        sf.write(write_name, audio, samplerate=sr, format='WAV', subtype='PCM_16')

    # Compute the STFT with windowing
    # win_length = 256 e n_fft = 256
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window, center=True)
    magnitude, phase = librosa.magphase(stft)  # magnitude já vem o valor absoluto
    phase_angle = np.angle(phase)

    return magnitude, phase_angle

def revert_features_to_audio(features, phase, cleanMean=None, cleanStd=None, transform=True):

    # magnitude clean: (2501, 129, 1)
    if transform:
        features = np.squeeze(features)
        features = np.transpose(features, (1, 0))

    # undo the scaling
    if cleanMean is not None and cleanStd is not None:
        if cleanMean.any() and cleanStd.any():
            features = cleanStd * features + cleanMean

    # phase: (2501, 129)
    # phase = np.transpose(phase, (1, 0))

    # features: (129, 2501)
    # phase: (129, 2501)
    features = features * np.exp(1j * phase)

    audio = librosa.istft(features, n_fft=window_length, win_length=window_length, hop_length=hop_length, window='hamming', center=True)

    return audio

def istft_only(features, phase):
    return librosa.istft(features, n_fft=window_length, win_length=window_length, hop_length=hop_length, window='hamming', center=True)

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
    reconstructed_audio = istft_only(complex_spec, noisy_phase)

    # Match RMS of reference audio if provided
    if reference_audio is not None:
        ref_rms = np.sqrt(np.mean(reference_audio**2))
        current_rms = np.sqrt(np.mean(reconstructed_audio**2))
        reconstructed_audio = reconstructed_audio * (ref_rms / current_rms)

    return reconstructed_audio

def write_audio(clean_audio, output_file='clean.wav'):

    # for noisy_mag_features, clean_magnitude in dataset.take(5):
    #     print('clean_magnitude:', clean_magnitude.shape)

    # if clean_audio is None:
    #     clean_audio = revert_features_to_audio(clean_magnitude.numpy(), noisy_phase)
    # else:
    #     clean_audio = np.append(clean_audio, revert_features_to_audio(clean_magnitude.numpy(), noisy_phase))

    # clean_audio = clean_audio.flatten()
    # clean_audio = clean_audio / np.max(np.abs(clean_audio))
    # clean_audio = np.clip(clean_audio, -1.0, 1.0)
    # clean_audio = clean_audio.astype(np.float32)
    # print('clean_audio max min:', clean_audio.max(), clean_audio.min())
    sf.write(output_file, clean_audio, samplerate=8000, format='WAV', subtype='PCM_16')
