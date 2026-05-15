"""
Parse TFRecord shards produced by build_tfrecords.py into tf.data pipelines.

Two factory functions:
  - make_crm_dataset    : (input=noisy_real_imag, target=complex_ratio_mask)
                          for the cRM U-Net.
  - make_mag_dataset    : (input=log1p(|noisy|), target=log1p(|clean|))
                          for the Park-style magnitude-domain FCN.

Both factories accept a glob of TFRecord files; train/val are kept in separate
subfolders by build_tfrecords.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import tensorflow as tf

FEATURE_SPEC = {
    "noisy_real": tf.io.VarLenFeature(tf.float32),
    "noisy_imag": tf.io.VarLenFeature(tf.float32),
    "clean_real": tf.io.VarLenFeature(tf.float32),
    "clean_imag": tf.io.VarLenFeature(tf.float32),
    "n_freq": tf.io.FixedLenFeature([], tf.int64),
    "n_time": tf.io.FixedLenFeature([], tf.int64),
    "snr": tf.io.VarLenFeature(tf.float32),
}


def load_dataset_info(tfrecord_root: Path) -> dict:
    return json.loads((tfrecord_root / "dataset_info.json").read_text())


def _parse_raw(example):
    feats = tf.io.parse_single_example(example, FEATURE_SPEC)
    n_freq = tf.cast(feats["n_freq"], tf.int32)
    n_time = tf.cast(feats["n_time"], tf.int32)
    nr = tf.reshape(tf.sparse.to_dense(feats["noisy_real"]), [n_freq, n_time])
    ni = tf.reshape(tf.sparse.to_dense(feats["noisy_imag"]), [n_freq, n_time])
    cr = tf.reshape(tf.sparse.to_dense(feats["clean_real"]), [n_freq, n_time])
    ci = tf.reshape(tf.sparse.to_dense(feats["clean_imag"]), [n_freq, n_time])
    snr = tf.sparse.to_dense(feats["snr"])[0]
    return nr, ni, cr, ci, snr


def _to_crm(nr, ni, cr, ci, snr):
    """input: stacked noisy (real,imag); target: complex ratio mask (real,imag).
    cRM = (clean * conj(noisy)) / (|noisy|^2 + eps); clipped to a stable range."""
    eps = 1e-6
    denom = nr * nr + ni * ni + eps
    mr = (cr * nr + ci * ni) / denom
    mi = (ci * nr - cr * ni) / denom
    mr = tf.clip_by_value(mr, -5.0, 5.0)
    mi = tf.clip_by_value(mi, -5.0, 5.0)
    inputs = tf.stack([nr, ni], axis=-1)         # (F, T, 2)
    targets = tf.stack([mr, mi], axis=-1)        # (F, T, 2)
    return inputs, targets


def _to_mag(nr, ni, cr, ci, snr):
    """input: log1p(|noisy|); target: log1p(|clean|). 1 channel each."""
    eps = 1e-8
    noisy_mag = tf.sqrt(nr * nr + ni * ni + eps)
    clean_mag = tf.sqrt(cr * cr + ci * ci + eps)
    inputs = tf.math.log1p(noisy_mag)[..., tf.newaxis]   # (F, T, 1)
    targets = tf.math.log1p(clean_mag)[..., tf.newaxis]  # (F, T, 1)
    return inputs, targets


def _build_pipeline(files: list[str],
                    transform: Callable,
                    batch_size: int,
                    shuffle: bool,
                    shuffle_buffer: int) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_raw, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda *xs: transform(*xs), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_crm_dataset(tfrecord_root: Path,
                     batch_size: int = 64,
                     shuffle_buffer: int = 4096,
                     repeat: bool = True
                     ) -> tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    """Return (train_ds, val_ds, info) for the cRM U-Net."""
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
                     shuffle_buffer: int = 4096,
                     repeat: bool = True
                     ) -> tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    """Return (train_ds, val_ds, info) for the magnitude-domain Park model."""
    info = load_dataset_info(tfrecord_root)
    train_files = sorted(str(p) for p in (tfrecord_root / "train").glob("*.tfrecord"))
    val_files = sorted(str(p) for p in (tfrecord_root / "val").glob("*.tfrecord"))
    train_ds = _build_pipeline(train_files, _to_mag, batch_size, True, shuffle_buffer)
    val_ds = _build_pipeline(val_files, _to_mag, batch_size, False, shuffle_buffer)
    if repeat:
        train_ds = train_ds.repeat()
    return train_ds, val_ds, info
