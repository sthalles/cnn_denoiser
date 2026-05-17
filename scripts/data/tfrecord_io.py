"""
Parse TFRecord shards produced by build_tfrecords.py into tf.data pipelines.

Pipeline design for tensor-core-fed training:
  - Reads shards in parallel via tf.data.Dataset.interleave.
  - Uses FixedLenFeature (static shape) so each example is parsed via the fast
    dense path -- no SparseTensor → DenseTensor conversion.
  - Heavy prefetch and a large interleave cycle keep the GPU busy.

Two factory functions:
  - make_crm_dataset    : (input=noisy_real_imag, target=complex_ratio_mask)
                          for cRM models (U-Net and Conformer-CRN).
  - make_mag_dataset    : (input=log1p(|noisy|), target=log1p(|clean|))
                          for the Park-style magnitude-domain FCN.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import tensorflow as tf

# Each TFRecord example holds 4 spectrograms of shape (n_freq, n_time) = (257, 251).
_N_FREQ = 257
_N_TIME = 251
_FLAT_LEN = _N_FREQ * _N_TIME  # 64_507

FEATURE_SPEC = {
    "noisy_real": tf.io.FixedLenFeature([_FLAT_LEN], tf.float32),
    "noisy_imag": tf.io.FixedLenFeature([_FLAT_LEN], tf.float32),
    "clean_real": tf.io.FixedLenFeature([_FLAT_LEN], tf.float32),
    "clean_imag": tf.io.FixedLenFeature([_FLAT_LEN], tf.float32),
    "n_freq": tf.io.FixedLenFeature([], tf.int64),
    "n_time": tf.io.FixedLenFeature([], tf.int64),
    "snr": tf.io.FixedLenFeature([1], tf.float32),
}


def load_dataset_info(tfrecord_root: Path) -> dict:
    return json.loads((tfrecord_root / "dataset_info.json").read_text())


def _parse_raw(example):
    feats = tf.io.parse_single_example(example, FEATURE_SPEC)
    nr = tf.reshape(feats["noisy_real"], [_N_FREQ, _N_TIME])
    ni = tf.reshape(feats["noisy_imag"], [_N_FREQ, _N_TIME])
    cr = tf.reshape(feats["clean_real"], [_N_FREQ, _N_TIME])
    ci = tf.reshape(feats["clean_imag"], [_N_FREQ, _N_TIME])
    snr = feats["snr"][0]
    return nr, ni, cr, ci, snr


def _to_crm(nr, ni, cr, ci, snr):
    """input: stacked noisy (real,imag); target: complex ratio mask (real,imag)."""
    eps = 1e-6
    denom = nr * nr + ni * ni + eps
    mr = (cr * nr + ci * ni) / denom
    mi = (ci * nr - cr * ni) / denom
    mr = tf.clip_by_value(mr, -5.0, 5.0)
    mi = tf.clip_by_value(mi, -5.0, 5.0)
    inputs = tf.stack([nr, ni], axis=-1)
    targets = tf.stack([mr, mi], axis=-1)
    return inputs, targets


def _to_mag(nr, ni, cr, ci, snr):
    """input: log1p(|noisy|); target: log1p(|clean|)."""
    eps = 1e-8
    noisy_mag = tf.sqrt(nr * nr + ni * ni + eps)
    clean_mag = tf.sqrt(cr * cr + ci * ci + eps)
    inputs = tf.math.log1p(noisy_mag)[..., tf.newaxis]
    targets = tf.math.log1p(clean_mag)[..., tf.newaxis]
    return inputs, targets


def _build_pipeline(files: list[str],
                    transform: Callable,
                    batch_size: int,
                    shuffle: bool,
                    shuffle_buffer: int,
                    interleave_cycle: int = 16) -> tf.data.Dataset:
    file_ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        file_ds = file_ds.shuffle(len(files), reshuffle_each_iteration=True)
    # Read shards in parallel; sloppy ordering keeps the pipeline saturated.
    ds = file_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, buffer_size=8 * 1024 * 1024),
        cycle_length=min(interleave_cycle, len(files)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(_parse_raw, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda *xs: transform(*xs), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_crm_dataset(tfrecord_root: Path,
                     batch_size: int = 64,
                     shuffle_buffer: int = 2048,
                     repeat: bool = True
                     ) -> tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    info = load_dataset_info(tfrecord_root)
    train_files = sorted(str(p) for p in (tfrecord_root / "train").glob("*.tfrecord"))
    val_files = sorted(str(p) for p in (tfrecord_root / "val").glob("*.tfrecord"))
    train_ds = _build_pipeline(train_files, _to_crm, batch_size, True, shuffle_buffer)
    val_ds = _build_pipeline(val_files, _to_crm, batch_size, False, shuffle_buffer)
    if repeat:
        train_ds = train_ds.repeat()
    return train_ds, val_ds, info


def make_mag_dataset(tfrecord_root: Path,
                     batch_size: int = 64,
                     shuffle_buffer: int = 2048,
                     repeat: bool = True
                     ) -> tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    info = load_dataset_info(tfrecord_root)
    train_files = sorted(str(p) for p in (tfrecord_root / "train").glob("*.tfrecord"))
    val_files = sorted(str(p) for p in (tfrecord_root / "val").glob("*.tfrecord"))
    train_ds = _build_pipeline(train_files, _to_mag, batch_size, True, shuffle_buffer)
    val_ds = _build_pipeline(val_files, _to_mag, batch_size, False, shuffle_buffer)
    if repeat:
        train_ds = train_ds.repeat()
    return train_ds, val_ds, info
