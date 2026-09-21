import numpy as np
import librosa
import super_utils as sutils
from sklearn.metrics import f1_score, precision_score, recall_score # noqa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf # noqa

num_features = 129
window_length = 256
hop_length = 64
sample_rate = 16000

# não usar toda memoria da GPU
if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

def sdr_metric(y_true, y_pred):
    # Flatten the tensors
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Compute the SDR
    noise = y_true - y_pred
    signal_power = tf.reduce_sum(tf.square(y_true))
    noise_power = tf.reduce_sum(tf.square(noise))
    sdr = 10 * tf.math.log(signal_power / noise_power) / tf.math.log(10.0)

    return sdr

def combined_loss(y_true, y_pred):
    # Time domain (MSE) component
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

    # Frequency domain component
    # Compare magnitudes of frequency components
    freq_true = tf.abs(y_true)
    freq_pred = tf.abs(y_pred)
    freq_loss = tf.reduce_mean(tf.abs(freq_true - freq_pred))

    # Combine losses with weighting
    alpha = 0.8  # Weight for MSE
    beta = 0.2   # Weight for frequency loss

    return alpha * mse + beta * freq_loss

tf.keras.utils.get_custom_objects().update({"sdr_metric": sdr_metric})
tf.keras.utils.get_custom_objects().update({"combined_loss": combined_loss})

model = tf.keras.models.load_model('8k-v8-checkpoint.keras', custom_objects={"sdr_metric": sdr_metric, "combined_loss": combined_loss})

def denoise(input_file='noisy.wav', output_file='denoised.wav'):

    # Load the noisy audio
    noisy_audio, sr = librosa.load(input_file, sr=16000)

    noisy_magnitude, noisy_phase = sutils.preprocess(noisy_audio, sr)

    # log-magnitude scaling
    eps = 1e-10
    noisy_magnitude = np.maximum(noisy_magnitude, eps)
    noisy_magnitude = np.log(noisy_magnitude + eps)

    # ##### get min and max for denormalization
    noisy_magnitude_min = noisy_magnitude.min()
    noisy_magnitude_max = noisy_magnitude.max()

    # Normalize the features
    noisy_magnitude = (noisy_magnitude - noisy_magnitude.min()) / (noisy_magnitude.max() - noisy_magnitude.min())

    # reshape
    noisy_mag_features = sutils.prepare_input_features(noisy_magnitude, 8, 129)

    # Transpose the dimensions
    noisy_mag_features = np.reshape(noisy_mag_features, (noisy_mag_features.shape[0], noisy_mag_features.shape[1], 1, noisy_mag_features.shape[2]))
    noisy_mag_features = np.transpose(noisy_mag_features, (3, 0, 1, 2)).astype(np.float32)
    # print('noisy mag features shape:', noisy_mag_features.shape)

    # Predict the clean features
    clean_features = model.predict(noisy_mag_features)

    # Remove singleton dimensions
    clean_features = np.squeeze(clean_features)  # Remove singleton dimensions

    clean_features = np.transpose(clean_features, (1, 0))

    denoised_audio = sutils.denormalize_and_reconstruct(
        clean_features,
        noisy_phase,
        noisy_magnitude_min,
        noisy_magnitude_max,
        reference_audio=noisy_audio
    )

    print('original file range:', np.abs(noisy_audio).min(), np.abs(noisy_audio).max())
    print('denoised file range:', np.abs(denoised_audio).min(), np.abs(denoised_audio).max())

    # Write the denoised audio to a file
    sutils.write_audio(denoised_audio, output_file)


def process_all_files(input_dir, output_base_dir):
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Get all wav files
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    for wav_file in wav_files:
        # Extract SNR value from filename (format: "noisy1_SNRdb_0.0_clnsp1.wav")
        parts = wav_file.split('_')
        try:
            # Find the SNR value between "SNRdb" and the next underscore
            snr_idx = parts.index('SNRdb')
            snr_value = parts[snr_idx + 1]
            output_dir = os.path.join(output_base_dir, f'SNR_{snr_value}db')
        except (ValueError, IndexError):
            output_dir = os.path.join(output_base_dir, 'unknown_snr')

        # Create SNR-specific output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create output filename by replacing 'noisy' with 'denoised'
        output_filename = 'denoised_' + '_'.join(parts[1:])

        # Process the file
        input_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, output_filename)
        print(f'Processing: {wav_file} -> {output_path}')
        denoise(input_path, output_path)

# Example usage
input_directory = 'dataset/NoisySpeech_training'
output_directory = 'dataset/DenoisedSpeech_training'
# Process all files in the input directory
process_all_files(input_directory, output_directory)

# denoise('test_saga/noisy.wav', 'denoised_rede.wav')
