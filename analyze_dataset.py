import numpy as np
import librosa
import os
from scipy import stats

def analyze_dataset(data_dir):
    durations = []
    snr_values = []
    rms_values = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.wav'):
            # Load audio
            audio, sr = librosa.load(os.path.join(data_dir, file))
            
            # Duration
            durations.append(len(audio) / sr)
            
            # RMS energy
            rms = np.sqrt(np.mean(audio**2))
            rms_values.append(rms)
            
            # # If you have clean/noisy pairs, calculate SNR
            # if '_noisy' in file:
            #     clean_file = file.replace('_noisy', '_clean')
            #     clean_audio, _ = librosa.load(os.path.join(data_dir, clean_file))
            #     noise = audio - clean_audio
            #     snr = 10 * np.log10(np.sum(clean_audio**2) / np.sum(noise**2))
            #     snr_values.append(snr)
    
    stats_dict = {
        "total_files": len(durations),
        "total_duration_hours": sum(durations) / 3600,
        "mean_duration": np.mean(durations),
        "std_duration": np.std(durations),
        "mean_snr": np.mean(snr_values),
        # "snr_range": (min(snr_values), max(snr_values)),
        "rms_stats": stats.describe(rms_values)
    }
    
    return stats_dict

def analyze_spectral_content(audio_path):
    audio, sr = librosa.load(audio_path)
    
    # Compute spectral features
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    return {
        "spectral_centroid_mean": np.mean(spec_cent),
        "spectral_bandwidth_mean": np.mean(spec_bw),
        "spectral_rolloff_mean": np.mean(spec_rolloff)
    }

print(analyze_dataset('dataset/CleanSpeech_training'))
