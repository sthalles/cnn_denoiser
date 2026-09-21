"""
Audio Denoiser - Dataset Module (v2)

This module provides TFRecord parsing and dataset loading functions
for use by the training script.
"""

import os
import sys
import re

# Suppress warnings before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class StderrFilter:
    """Filter to suppress noisy CUDA/XLA warnings from stderr"""
    
    FILTER_PATTERNS = [
        r'gpu_timer',
        r'ptx85',
        r'NUMA',
        r'numa_node', 
        r'delay kernel',
        r'recognized feature',
        r'Skipping the delay',
        r'ignoring feature',
        r'cuFFT factory',
        r'cuDNN factory',
        r'cuBLAS factory',
        r'absl::InitializeLog',
    ]
    
    def __init__(self, stream):
        self.stream = stream
        self.pattern = re.compile('|'.join(self.FILTER_PATTERNS), re.IGNORECASE)
        
    def write(self, msg):
        if msg and not self.pattern.search(msg):
            self.stream.write(msg)
            
    def flush(self):
        self.stream.flush()
        
    def close(self):
        if hasattr(self.stream, 'close'):
            self.stream.close()


# Install stderr filter before any TF imports
if not isinstance(sys.stderr, StderrFilter):
    sys.stderr = StderrFilter(sys.__stderr__)

import warnings
warnings.filterwarnings('ignore')

# Re-export functions from the dataset creation script
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
from typing import Tuple
from config import Config
from importlib import import_module
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration

# Import the dataset functions


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
    data_dir=None,
    batch_size=None,
    shuffle=True,
    validation_split=0.1
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Load TFRecord dataset for training

    Args:
        data_dir: Directory containing TFRecord files
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        validation_split: Fraction of data for validation

    Returns:
        train_dataset, val_dataset, train_steps, val_steps
    """
    if data_dir is None:
        data_dir = Config.TFRECORD_DIR
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

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
        print(f"Dataset info: {num_examples} examples")
    else:
        # Estimate from file count
        num_examples = len(filenames) * 5000
        print(f"Estimated examples: {num_examples}")

    # Split files for train/val
    n_val_files = max(1, int(len(filenames) * validation_split))
    n_train_files = len(filenames) - n_val_files

    if shuffle:
        np.random.shuffle(filenames)

    train_files = filenames[:n_train_files]
    val_files = filenames[n_train_files:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Create datasets with interleaving for better I/O throughput
    train_dataset = tf.data.TFRecordDataset(
        train_files,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=8 * 1024 * 1024  # 8MB read buffer
    )
    val_dataset = tf.data.TFRecordDataset(
        val_files,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=8 * 1024 * 1024
    )

    # Parse records with parallel calls
    train_dataset = train_dataset.map(
        parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False  # Allow out-of-order for speed
    )
    val_dataset = val_dataset.map(
        parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    # Cache datasets to memory to avoid re-reading from disk
    # This eliminates the periodic GPU stalls from disk I/O
    print("Caching datasets to memory (this may take a moment)...")
    train_dataset = train_dataset.cache()
    val_dataset = val_dataset.cache()

    # Shuffle, batch, and prefetch for training
    if shuffle:
        # Smaller shuffle buffer to reduce memory and startup time
        train_dataset = train_dataset.shuffle(buffer_size=2000)

    # IMPORTANT: Add repeat() to prevent "ran out of data" errors
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    # Validation also needs repeat() to prevent exhaustion
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Estimate steps
    train_examples = int(num_examples * (1 - validation_split))
    val_examples = int(num_examples * validation_split)
    train_steps = train_examples // batch_size
    val_steps = max(1, val_examples // batch_size)

    print(f"Training: ~{train_examples} examples, {train_steps} steps/epoch")
    print(f"Validation: ~{val_examples} examples, {val_steps} steps/epoch")

    return train_dataset, val_dataset, train_steps, val_steps


def augment_data(inputs, targets):
    """
    Data augmentation for training

    Args:
        inputs: Input spectrogram (n_freq, n_time, 2)
        targets: Target mask (n_freq, n_time, 2)

    Returns:
        Augmented inputs and targets
    """
    # Random gain (simulate different volume levels)
    gain = tf.random.uniform([], 0.5, 2.0)
    inputs = inputs * gain
    # Mask remains the same (it's relative)

    # Random time shift (small shifts)
    shift = tf.random.uniform([], -5, 5, dtype=tf.int32)
    inputs = tf.roll(inputs, shift, axis=1)
    targets = tf.roll(targets, shift, axis=1)

    return inputs, targets


def load_tfrecord_dataset_with_augmentation(
    data_dir=None,
    batch_size=None,
    validation_split=0.1
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Load TFRecord dataset with data augmentation for training
    """
    train_dataset, val_dataset, train_steps, val_steps = load_tfrecord_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split
    )

    # Apply augmentation only to training data (before batching)
    # Note: We need to unbatch, augment, then rebatch
    # This is less efficient, so we'll apply augmentation as a map operation
    # on the batched dataset instead

    def batch_augment(inputs, targets):
        # Random gain per sample in batch
        gain = tf.random.uniform([tf.shape(inputs)[0], 1, 1, 1], 0.7, 1.3)
        inputs = inputs * gain
        return inputs, targets

    train_dataset = train_dataset.map(
        batch_augment,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return train_dataset, val_dataset, train_steps, val_steps


if __name__ == "__main__":
    # Test loading
    print("Testing dataset loading...")

    try:
        train_ds, val_ds, train_steps, val_steps = load_tfrecord_dataset(
            batch_size=32
        )

        print("\nSample batch:")
        for inputs, targets in train_ds.take(1):
            print(f"  Inputs shape: {inputs.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(
                f"  Inputs range: [{tf.reduce_min(inputs):.4f}, {tf.reduce_max(inputs):.4f}]")
            print(
                f"  Targets range: [{tf.reduce_min(targets):.4f}, {tf.reduce_max(targets):.4f}]")

        print("\nDataset loading successful!")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you've run the dataset creation script first.")
