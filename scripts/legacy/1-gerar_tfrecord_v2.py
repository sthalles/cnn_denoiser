"""
Audio Denoiser - Dataset Creation (v2)

This script creates TFRecord dataset from clean/noisy audio pairs.
Key improvements:
1. Uses Complex Ratio Mask (cRM) as target
2. Global normalization statistics
3. Proper segmentation with overlap
4. Stores both noisy spectrogram and target mask
"""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from config import (
    Config, load_audio, compute_stft, stft_to_features,
    compute_complex_mask, segment_audio, get_file_pairs, normalize_spectrogram
)


# =============================================================================
# TFRecord Serialization
# =============================================================================

def _float_feature(value):
    """Returns a float_list from a numpy array"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def _int64_feature(value):
    """Returns an int64_list from an int"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string"""
    if isinstance(value, str):
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(noisy_real, noisy_imag, mask_real, mask_imag, snr):
    """
    Serialize a training example to TFRecord format

    Args:
        noisy_real: Real part of normalized noisy spectrogram (n_freq, n_time)
        noisy_imag: Imaginary part of normalized noisy spectrogram
        mask_real: Real part of complex ratio mask
        mask_imag: Imaginary part of complex ratio mask
        snr: SNR level in dB
    """
    n_freq, n_time = noisy_real.shape

    feature = {
        'noisy_real': _float_feature(noisy_real.astype(np.float32)),
        'noisy_imag': _float_feature(noisy_imag.astype(np.float32)),
        'mask_real': _float_feature(mask_real.astype(np.float32)),
        'mask_imag': _float_feature(mask_imag.astype(np.float32)),
        'n_freq': _int64_feature(n_freq),
        'n_time': _int64_feature(n_time),
        'snr': _float_feature(np.array([snr], dtype=np.float32))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# =============================================================================
# Dataset Processing
# =============================================================================

def process_audio_pair(clean_path, noisy_path, snr, max_mag):
    """
    Process a single clean/noisy audio pair

    Returns list of (noisy_real, noisy_imag, mask_real, mask_imag, snr) tuples
    """
    try:
        # Load audio
        clean_audio, _ = load_audio(clean_path)
        noisy_audio, _ = load_audio(noisy_path)

        # Ensure same length
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        # Segment audio
        clean_segments = segment_audio(clean_audio)
        noisy_segments = segment_audio(noisy_audio)

        examples = []

        for (_, clean_seg), (_, noisy_seg) in zip(clean_segments, noisy_segments):
            # Compute STFT
            clean_stft = compute_stft(clean_seg)
            noisy_stft = compute_stft(noisy_seg)

            # Get features from noisy (input)
            noisy_real, noisy_imag = stft_to_features(noisy_stft)

            # Normalize noisy spectrogram
            noisy_real_norm, noisy_imag_norm = normalize_spectrogram(
                noisy_real, noisy_imag, max_mag
            )

            # Compute complex ratio mask (target)
            mask = compute_complex_mask(noisy_stft, clean_stft)
            mask_real = np.real(mask)
            mask_imag = np.imag(mask)

            # Clip mask to reasonable range (helps training stability)
            # Typical masks are in [-2, 2] range for reasonable SNRs
            mask_real = np.clip(mask_real, -5.0, 5.0)
            mask_imag = np.clip(mask_imag, -5.0, 5.0)

            examples.append((
                noisy_real_norm.astype(np.float32),
                noisy_imag_norm.astype(np.float32),
                mask_real.astype(np.float32),
                mask_imag.astype(np.float32),
                snr
            ))

        return examples

    except Exception as e:
        print(f"Error processing {noisy_path}: {e}")
        return []


def compute_global_stats(pairs, num_samples=1000):
    """
    Compute global normalization statistics from a sample of the dataset
    """
    print("Computing global normalization statistics...")

    # Sample pairs
    np.random.shuffle(pairs)
    sample_pairs = pairs[:min(num_samples, len(pairs))]

    all_mags = []

    for clean_path, noisy_path, _ in tqdm(sample_pairs, desc="Computing stats"):
        try:
            noisy_audio, _ = load_audio(noisy_path)
            stft = compute_stft(noisy_audio)
            mag = np.abs(stft)
            all_mags.append(mag.flatten())
        except Exception as e:
            continue

    all_mags = np.concatenate(all_mags)

    # Compute statistics
    mean = float(np.mean(all_mags))
    std = float(np.std(all_mags))
    # Use 99.9th percentile to avoid outliers
    max_mag = float(np.percentile(all_mags, 99.9))

    print(f"  Mean magnitude: {mean:.4f}")
    print(f"  Std magnitude: {std:.4f}")
    print(f"  Max magnitude (99.9%): {max_mag:.4f}")

    return mean, std, max_mag


def create_tfrecord_dataset(
    clean_dir=Config.CLEAN_DIR,
    noisy_dir=Config.NOISY_DIR,
    output_dir=Config.TFRECORD_DIR,
    samples_per_shard=5000,
    num_workers=None
):
    """
    Create TFRecord dataset from audio files
    
    Writes shards incrementally to avoid memory issues.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get file pairs
    print("Finding audio file pairs...")
    pairs = get_file_pairs(clean_dir, noisy_dir)
    print(f"Found {len(pairs)} audio pairs")

    if len(pairs) == 0:
        raise ValueError("No audio pairs found!")

    # Compute global normalization statistics
    mean, std, max_mag = compute_global_stats(pairs)
    Config.save_norm_stats(mean, std, max_mag)
    print(f"Normalization stats saved to {Config.get_norm_stats_path()}")

    # Process pairs and write TFRecords serially (memory efficient)
    print("Processing audio pairs and writing TFRecords...")

    # Initialize shard tracking with open writer
    shard_idx = 0
    current_shard_count = 0
    total_examples = 0
    
    # Open first writer
    shard_path = os.path.join(output_dir, f'denoiser-{shard_idx:05d}.tfrecord')
    writer = tf.io.TFRecordWriter(shard_path)
    
    # Process pairs one at a time (memory efficient)
    for clean_path, noisy_path, snr in tqdm(pairs, desc="Processing & Writing"):
        examples = process_audio_pair(clean_path, noisy_path, snr, max_mag)
        
        # Write each example immediately
        for noisy_real, noisy_imag, mask_real, mask_imag, snr_val in examples:
            # Check if we need a new shard
            if current_shard_count >= samples_per_shard:
                writer.close()
                shard_idx += 1
                current_shard_count = 0
                shard_path = os.path.join(output_dir, f'denoiser-{shard_idx:05d}.tfrecord')
                writer = tf.io.TFRecordWriter(shard_path)
            
            # Write example
            serialized = serialize_example(
                noisy_real, noisy_imag, mask_real, mask_imag, snr_val
            )
            writer.write(serialized)
            current_shard_count += 1
            total_examples += 1
    
    # Close final writer
    if writer is not None:
        writer.close()
    
    num_shards = shard_idx + 1
    
    num_shards = shard_idx
    print(f"\nCreated {num_shards} TFRecord shards with {total_examples} total examples")

    # Save dataset info
    info = {
        'num_examples': total_examples,
        'num_shards': num_shards,
        'n_freq': Config.N_FREQ_BINS,
        'n_time': Config.N_TIME_FRAMES,
        'sample_rate': Config.SAMPLE_RATE,
        'n_fft': Config.N_FFT,
        'hop_length': Config.HOP_LENGTH,
        'segment_duration': Config.SEGMENT_DURATION,
        'global_mean': mean,
        'global_std': std,
        'global_max_mag': max_mag
    }

    info_path = os.path.join(output_dir, 'dataset_info.npz')
    np.savez(info_path, **info)
    print(f"Dataset info saved to {info_path}")

    return total_examples, num_shards


# =============================================================================
# TFRecord Parsing for Training
# =============================================================================

def parse_tfrecord(serialized_example):
    """
    Parse a single TFRecord example

    Returns:
        inputs: (n_freq, n_time, 2) tensor [noisy_real, noisy_imag]
        targets: (n_freq, n_time, 2) tensor [mask_real, mask_imag]
    """
    feature_description = {
        'noisy_real': tf.io.VarLenFeature(tf.float32),
        'noisy_imag': tf.io.VarLenFeature(tf.float32),
        'mask_real': tf.io.VarLenFeature(tf.float32),
        'mask_imag': tf.io.VarLenFeature(tf.float32),
        'n_freq': tf.io.FixedLenFeature([], tf.int64),
        'n_time': tf.io.FixedLenFeature([], tf.int64),
        'snr': tf.io.VarLenFeature(tf.float32)
    }

    example = tf.io.parse_single_example(
        serialized_example, feature_description)

    n_freq = tf.cast(example['n_freq'], tf.int32)
    n_time = tf.cast(example['n_time'], tf.int32)

    # Convert sparse to dense and reshape
    noisy_real = tf.reshape(tf.sparse.to_dense(
        example['noisy_real']), [n_freq, n_time])
    noisy_imag = tf.reshape(tf.sparse.to_dense(
        example['noisy_imag']), [n_freq, n_time])
    mask_real = tf.reshape(tf.sparse.to_dense(
        example['mask_real']), [n_freq, n_time])
    mask_imag = tf.reshape(tf.sparse.to_dense(
        example['mask_imag']), [n_freq, n_time])

    # Stack into channels
    inputs = tf.stack([noisy_real, noisy_imag], axis=-1)  # (n_freq, n_time, 2)
    targets = tf.stack([mask_real, mask_imag], axis=-1)   # (n_freq, n_time, 2)

    return inputs, targets


def load_tfrecord_dataset(
    data_dir=Config.TFRECORD_DIR,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    validation_split=0.1
):
    """
    Load TFRecord dataset for training

    Returns:
        train_dataset, val_dataset, train_steps, val_steps
    """
    # Load normalization stats
    Config.load_norm_stats()

    # Find all TFRecord files
    tfrecord_pattern = os.path.join(data_dir, '*.tfrecord')
    filenames = sorted(tf.io.gfile.glob(tfrecord_pattern))

    if not filenames:
        raise ValueError(f"No TFRecord files found in {data_dir}")

    print(f"Found {len(filenames)} TFRecord files")

    # Load dataset info
    info_path = os.path.join(data_dir, 'dataset_info.npz')
    if os.path.exists(info_path):
        info = np.load(info_path)
        num_examples = int(info['num_examples'])
    else:
        # Estimate from file count
        num_examples = len(filenames) * 5000

    # Split files for train/val
    n_val_files = max(1, int(len(filenames) * validation_split))
    n_train_files = len(filenames) - n_val_files

    if shuffle:
        np.random.shuffle(filenames)

    train_files = filenames[:n_train_files]
    val_files = filenames[n_train_files:]

    # Create datasets
    train_dataset = tf.data.TFRecordDataset(
        train_files, num_parallel_reads=tf.data.AUTOTUNE)
    val_dataset = tf.data.TFRecordDataset(
        val_files, num_parallel_reads=tf.data.AUTOTUNE)

    # Parse and prepare
    train_dataset = train_dataset.map(
        parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(
        parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle, batch, and prefetch for training
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=10000)

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Estimate steps
    train_examples = int(num_examples * (1 - validation_split))
    val_examples = int(num_examples * validation_split)
    train_steps = train_examples // batch_size
    val_steps = max(1, val_examples // batch_size)

    print(f"Training examples: ~{train_examples}, steps/epoch: {train_steps}")
    print(f"Validation examples: ~{val_examples}, steps/epoch: {val_steps}")

    return train_dataset, val_dataset, train_steps, val_steps


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Create TFRecord dataset for denoiser')
    parser.add_argument('--clean-dir', default=Config.CLEAN_DIR,
                        help='Clean audio directory')
    parser.add_argument('--noisy-dir', default=Config.NOISY_DIR,
                        help='Noisy audio directory')
    parser.add_argument(
        '--output-dir', default=Config.TFRECORD_DIR, help='Output TFRecord directory')
    parser.add_argument('--samples-per-shard', type=int,
                        default=5000, help='Examples per shard')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker threads')

    args = parser.parse_args()

    print("=" * 60)
    print("Creating TFRecord Dataset")
    print("=" * 60)
    print(f"Clean directory: {args.clean_dir}")
    print(f"Noisy directory: {args.noisy_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"STFT params: n_fft={Config.N_FFT}, hop={Config.HOP_LENGTH}")
    print(f"Segment duration: {Config.SEGMENT_DURATION}s")
    print("=" * 60)

    num_examples, num_shards = create_tfrecord_dataset(
        clean_dir=args.clean_dir,
        noisy_dir=args.noisy_dir,
        output_dir=args.output_dir,
        samples_per_shard=args.samples_per_shard,
        num_workers=args.workers
    )

    print("=" * 60)
    print(f"Dataset created successfully!")
    print(f"  Total examples: {num_examples}")
    print(f"  Number of shards: {num_shards}")
    print("=" * 60)
