"""
Audio Denoiser - Automated Hyperparameter Search

Systematically trains and evaluates different model configurations to find the
best performing architecture. Runs unsupervised and can be stopped/resumed.

Usage:
    # Start or resume search:
    python 5-hyperparameter_search.py

    # Dry run (preview configurations without training):
    python 5-hyperparameter_search.py --dry-run

    # Show results so far:
    python 5-hyperparameter_search.py --results

    # Limit epochs per trial:
    python 5-hyperparameter_search.py --epochs 30

    # Stop gracefully: create file  search_results/STOP
    #   or press Ctrl+C (saves current state)

System specs targeted: RTX 4070 12GB VRAM, 24-thread CPU, ~20GB RAM
"""

import os
import sys
import re
import json
import time
import copy
import hashlib
import subprocess
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
# Keep XLA disabled for search stability; mixed precision + dropout + XLA has
# shown graph precision mismatches on this workload.


class StderrFilter:
    FILTER_PATTERNS = [
        r'gpu_timer', r'ptx85', r'NUMA', r'numa_node', r'delay kernel',
        r'recognized feature', r'Skipping the delay', r'ignoring feature',
        r'cuFFT factory', r'cuDNN factory', r'cuBLAS factory',
        r'absl::InitializeLog', r'StreamExecutor', r'TF-TRT', r'TensorRT',
        r'successful NUMA', r'XLA service',
    ]

    def __init__(self, stream):
        self.stream = stream
        self.pattern = re.compile('|'.join(self.FILTER_PATTERNS), re.IGNORECASE)

    def write(self, msg):
        if msg and not self.pattern.search(msg):
            self.stream.write(msg)

    def flush(self):
        self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        pass

sys.stderr = StderrFilter(sys.__stderr__)

import warnings
warnings.filterwarnings('ignore')
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, BackupAndRestore
)
from config import (
    Config, load_audio, save_audio, compute_sdr, setup_gpu
)

# Signal handlers for graceful shutdown
import signal

_state_for_signal_handler = None

def signal_handler(signum, frame):
    """Save state on signals (SIGINT, SIGTERM, etc)."""
    global _state_for_signal_handler
    if _state_for_signal_handler is not None:
        print(f"\n  [Signal {signum}] Saving state before exit...")
        try:
            save_state(_state_for_signal_handler)
        except Exception:
            pass
    sys.exit(130 if signum == signal.SIGINT else 1)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
try:
    signal.signal(signal.SIGABRT, signal_handler)
except Exception:
    pass

from dataset_v2 import load_tfrecord_dataset

# =============================================================================
# PATHS
# =============================================================================

SEARCH_DIR = 'search_results'
STATE_FILE = os.path.join(SEARCH_DIR, 'search_state.json')
STOP_FILE = os.path.join(SEARCH_DIR, 'STOP')


# =============================================================================
# CUSTOM LAYERS (same as training script, needed for model building)
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, activation=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_activation = activation
        self.conv = layers.Conv2D(
            filters, kernel_size, strides=strides, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU(negative_slope=0.2)

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.use_activation:
            x = self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters, 'kernel_size': self.kernel_size,
            'strides': self.strides, 'activation': self.use_activation
        })
        return config


@tf.keras.utils.register_keras_serializable()
class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = ConvBlock(filters, kernel_size=3)
        self.conv2 = ConvBlock(filters, kernel_size=3, activation=False)
        self.act = layers.LeakyReLU(negative_slope=0.2)

    def call(self, x, training=None):
        residual = x
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = layers.Add()([x, residual])
        x = self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


@tf.keras.utils.register_keras_serializable()
class ResizeToMatch(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, reference = inputs
        target_size = (tf.shape(reference)[1], tf.shape(reference)[2])
        # tf.image.resize may promote dtype under mixed precision; cast back.
        resized = tf.image.resize(x, target_size)
        return tf.cast(resized, x.dtype)


@tf.keras.utils.register_keras_serializable()
class AttentionGate(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.W_g = layers.Conv2D(filters, 1, padding='same')
        self.W_x = layers.Conv2D(filters, 1, padding='same')
        self.psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')
        self.bn = layers.BatchNormalization()
        self.resize = ResizeToMatch()

    def call(self, g, x, training=None):
        g_upsampled = self.resize([g, x])
        g1 = self.W_g(g_upsampled)
        x1 = self.W_x(x)
        psi = tf.nn.relu(g1 + x1)
        psi = self.psi(psi)
        psi = tf.cast(psi, x.dtype)
        return x * psi

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def spectral_convergence_loss(y_true, y_pred):
    return tf.norm(y_true - y_pred, ord='fro', axis=[-2, -1]) / (
        tf.norm(y_true, ord='fro', axis=[-2, -1]) + 1e-8)


def make_combined_loss(mse_w=0.5, mag_w=0.3, sc_w=0.2):
    """Create a combined loss with configurable weights"""

    @tf.keras.utils.register_keras_serializable()
    def combined_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        true_real, true_imag = y_true[..., 0], y_true[..., 1]
        pred_real, pred_imag = y_pred[..., 0], y_pred[..., 1]

        true_mag = tf.sqrt(tf.square(true_real) + tf.square(true_imag) + 1e-8)
        pred_mag = tf.sqrt(tf.square(pred_real) + tf.square(pred_imag) + 1e-8)
        mag_loss = tf.reduce_mean(tf.abs(true_mag - pred_mag))

        sc_loss = tf.reduce_mean(spectral_convergence_loss(y_true, y_pred))

        return mse_w * mse + mag_w * mag_loss + sc_w * sc_loss

    return combined_loss


@tf.keras.utils.register_keras_serializable()
def mask_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.keras.utils.register_keras_serializable()
def mask_mae(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


# =============================================================================
# PARAMETERIZED MODEL BUILDER
# =============================================================================

def build_model(hparams: dict, input_shape: tuple) -> Model:
    """
    Build a U-Net model with arbitrary hyperparameters.

    hparams keys:
        encoder_filters:  list[int]  e.g. [64, 128, 256, 512]
        res_blocks:       int        residual blocks per encoder/decoder level
        use_attention:    bool       attention gates on skip connections
        bottleneck:       str        'dilated' | 'standard' | 'none'
        dropout:          float      base dropout rate (scaled by depth)
        first_kernel:     int        first conv kernel size
        mask_scale:       float      output mask range [-scale, scale]
        mask_activation:  str        'tanh' | 'linear'
    """
    enc_filters = hparams['encoder_filters']
    n_levels = len(enc_filters)
    res_blocks = hparams.get('res_blocks', 2)
    use_attention = hparams.get('use_attention', True)
    bottleneck_type = hparams.get('bottleneck', 'dilated')
    base_dropout = hparams.get('dropout', 0.1)
    first_kernel = hparams.get('first_kernel', 7)
    mask_scale = hparams.get('mask_scale', 5.0)
    mask_act = hparams.get('mask_activation', 'tanh')

    inputs = Input(shape=input_shape, name='input')

    # ── ENCODER ──
    skips = []
    x = inputs

    for i, filters in enumerate(enc_filters):
        if i == 0:
            x = ConvBlock(filters, kernel_size=first_kernel)(x)
            x = ConvBlock(filters)(x)
        else:
            x = ConvBlock(filters, strides=2)(x)
            for _ in range(res_blocks):
                x = ResidualBlock(filters)(x)

        drop_rate = base_dropout * (1 + i) / n_levels
        if drop_rate > 0.0:
            x = SafeDropout(drop_rate)(x)
        skips.append(x)

    # ── BOTTLENECK ──
    bn_filters = enc_filters[-1]

    if bottleneck_type == 'dilated':
        x = ConvBlock(bn_filters)(x)
        for rate in [2, 4, 8]:
            x = layers.Conv2D(bn_filters, 3, padding='same',
                              dilation_rate=rate, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        x = ResidualBlock(bn_filters)(x)
    elif bottleneck_type == 'standard':
        x = ConvBlock(bn_filters)(x)
        for _ in range(res_blocks):
            x = ResidualBlock(bn_filters)(x)
    # 'none' -> skip bottleneck entirely

    bottleneck_drop = base_dropout * 1.5
    if bottleneck_drop > 0.0:
        x = SafeDropout(bottleneck_drop)(x)

    # ── DECODER ──
    for i in range(n_levels - 1, 0, -1):
        dec_filters = enc_filters[i - 1]
        x = layers.Conv2DTranspose(dec_filters, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = ResizeToMatch()([x, skips[i - 1]])

        skip = skips[i - 1]
        if use_attention:
            skip = AttentionGate(dec_filters)(x, skip)

        x = layers.Concatenate()([x, skip])
        x = ConvBlock(dec_filters)(x)
        for _ in range(min(res_blocks, 2)):
            x = ResidualBlock(dec_filters)(x)

        drop_rate = base_dropout * i / n_levels
        if drop_rate > 0.0:
            x = SafeDropout(drop_rate)(x)

    # ── Final upscale to match input ──
    x = ResizeToMatch()([x, inputs])

    # ── OUTPUT ──
    if mask_act == 'tanh':
        out = layers.Conv2D(
            2, 3, padding='same', activation='tanh',
            dtype='float32', name='output'
        )(x)
        out = out * mask_scale
    else:
        out = layers.Conv2D(
            2, 3, padding='same', activation='linear',
            dtype='float32', name='output'
        )(x)

    model = Model(inputs, out, name='UNet_search')
    return model


class SafeDropout(layers.Layer):
    """Dropout computed in float32, cast back to input dtype for stability."""

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)
        self.dropout = layers.Dropout(self.rate)

    def call(self, x, training=None):
        if self.rate <= 0.0:
            return x
        x32 = tf.cast(x, tf.float32)
        y32 = self.dropout(x32, training=training)
        return tf.cast(y32, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config


# =============================================================================
# SEARCH SPACE - carefully curated configurations
# =============================================================================

def generate_search_space() -> List[Dict[str, Any]]:
    """
    Generate the list of model configurations to try.

    Returns a list of dicts, each specifying a unique configuration.
    Configurations are ordered roughly from fastest to slowest.
    """
    configs = []

    def add(name, **kw):
        cfg = {
            'name': name,
            'encoder_filters': kw.get('encoder_filters', [64, 128, 256, 512]),
            'res_blocks': kw.get('res_blocks', 2),
            'use_attention': kw.get('use_attention', True),
            'bottleneck': kw.get('bottleneck', 'dilated'),
            'dropout': kw.get('dropout', 0.1),
            'first_kernel': kw.get('first_kernel', 7),
            'mask_scale': kw.get('mask_scale', 5.0),
            'mask_activation': kw.get('mask_activation', 'tanh'),
            'learning_rate': kw.get('learning_rate', 1e-3),
            'loss_weights': kw.get('loss_weights', [0.5, 0.3, 0.2]),
            'batch_size': kw.get('batch_size', None),  # None = auto
        }
        configs.append(cfg)

    # ── 1. Baselines with different filter widths ──
    add('baseline_current',
        encoder_filters=[64, 128, 256, 512])

    add('narrow_32',
        encoder_filters=[32, 64, 128, 256])

    add('mid_48',
        encoder_filters=[48, 96, 192, 384])

    add('wide_96',
        encoder_filters=[96, 192, 384, 512])

    # ── 2. Depth variations ──
    add('shallow_3lvl',
        encoder_filters=[64, 128, 256])

    add('deep_5lvl',
        encoder_filters=[32, 64, 128, 256, 512])

    # ── 3. Residual block count ──
    add('res1',
        res_blocks=1)

    add('res3',
        res_blocks=3)

    # ── 4. Attention vs no attention ──
    add('no_attention',
        use_attention=False)

    # ── 5. Bottleneck variations ──
    add('bottleneck_standard',
        bottleneck='standard')

    add('bottleneck_none',
        bottleneck='none')

    # ── 6. Dropout variations ──
    add('dropout_0',
        dropout=0.0)

    add('dropout_high',
        dropout=0.25)

    # ── 7. First kernel size ──
    add('firstk3',
        first_kernel=3)

    add('firstk5',
        first_kernel=5)

    # ── 8. Mask output ──
    add('mask_linear',
        mask_activation='linear', mask_scale=1.0)

    add('mask_scale3',
        mask_scale=3.0)

    add('mask_scale10',
        mask_scale=10.0)

    # ── 9. Loss weight variations ──
    add('loss_mse_heavy',
        loss_weights=[0.7, 0.2, 0.1])

    add('loss_mag_heavy',
        loss_weights=[0.3, 0.5, 0.2])

    add('loss_sc_heavy',
        loss_weights=[0.3, 0.2, 0.5])

    # ── 10. Learning rate ──
    add('lr_low',
        learning_rate=5e-4)

    add('lr_high',
        learning_rate=2e-3)

    # ── 11. Promising combos ──
    add('combo_narrow_deep',
        encoder_filters=[32, 64, 128, 256, 512],
        res_blocks=1, dropout=0.15)

    add('combo_wide_shallow',
        encoder_filters=[96, 192, 384],
        res_blocks=2, use_attention=True, dropout=0.1)

    add('combo_balanced',
        encoder_filters=[48, 96, 192, 384],
        res_blocks=2, bottleneck='dilated',
        first_kernel=5, mask_scale=5.0,
        learning_rate=5e-4, loss_weights=[0.4, 0.4, 0.2])

    add('combo_aggressive',
        encoder_filters=[64, 128, 256, 512],
        res_blocks=3, dropout=0.2,
        learning_rate=5e-4, loss_weights=[0.3, 0.4, 0.3],
        mask_scale=7.0)

    add('combo_lightweight',
        encoder_filters=[32, 64, 128],
        res_blocks=1, bottleneck='standard',
        first_kernel=5, dropout=0.05,
        learning_rate=1e-3)

    # ── Final winner (best from full search + refinement) ──
    # ref_combo_aggressive__plus_mask_linear: +3.35 dB SDR avg
    # combo_aggressive config with mask_linear traits (linear mask, scale=1.0)
    add('final_winner',
        encoder_filters=[64, 128, 256, 512],
        res_blocks=3,
        use_attention=True,
        bottleneck='dilated',
        dropout=0.2,
        first_kernel=7,
        mask_scale=1.0,
        mask_activation='linear',
        learning_rate=5e-4,
        loss_weights=[0.3, 0.4, 0.3])

    return configs


REFINEMENT_KEYS = [
    'encoder_filters', 'res_blocks', 'use_attention', 'bottleneck',
    'dropout', 'first_kernel', 'mask_scale', 'mask_activation',
    'learning_rate', 'loss_weights', 'batch_size',
]


def _generate_static_refinement_space() -> List[Dict[str, Any]]:
    """Fallback refinement space used when no completed source state exists."""
    configs = []

    def add(name, **kw):
        cfg = {
            'name': name,
            'encoder_filters': kw.get('encoder_filters', [64, 128, 256, 512]),
            'res_blocks': kw.get('res_blocks', 2),
            'use_attention': kw.get('use_attention', True),
            'bottleneck': kw.get('bottleneck', 'dilated'),
            'dropout': kw.get('dropout', 0.1),
            'first_kernel': kw.get('first_kernel', 7),
            'mask_scale': kw.get('mask_scale', 5.0),
            'mask_activation': kw.get('mask_activation', 'tanh'),
            'learning_rate': kw.get('learning_rate', 1e-3),
            'loss_weights': kw.get('loss_weights', [0.5, 0.3, 0.2]),
            'batch_size': kw.get('batch_size', None),
        }
        configs.append(cfg)

    add('ref_mask_linear__self',
        mask_activation='linear', mask_scale=1.0)
    add('ref_mask_linear__plus_res3',
        mask_activation='linear', mask_scale=1.0,
        res_blocks=3)
    add('ref_mask_linear__plus_deep_5lvl',
        mask_activation='linear', mask_scale=1.0,
        encoder_filters=[32, 64, 128, 256, 512])

    add('ref_res3__self',
        res_blocks=3)
    add('ref_res3__plus_mask_linear',
        res_blocks=3,
        mask_activation='linear', mask_scale=1.0)
    add('ref_res3__plus_deep_5lvl',
        res_blocks=3,
        encoder_filters=[32, 64, 128, 256, 512])

    add('ref_deep_5lvl__self',
        encoder_filters=[32, 64, 128, 256, 512])
    add('ref_deep_5lvl__plus_mask_linear',
        encoder_filters=[32, 64, 128, 256, 512],
        mask_activation='linear', mask_scale=1.0)
    add('ref_deep_5lvl__plus_res3',
        encoder_filters=[32, 64, 128, 256, 512],
        res_blocks=3)

    return configs


def generate_refinement_space(source_state_path: str = '', top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Build a refinement space from the actual top-N completed configs.

    For each winner family, include:
      - the original winning config (`__self`)
      - one hybrid per other top config by applying that donor's delta from
        `baseline_current` onto the family config

    This yields one best candidate family per top config while still probing a
    couple of local permutations around each winner.
    """
    if not source_state_path or not os.path.exists(source_state_path):
        return _generate_static_refinement_space()

    base_space = generate_search_space()
    by_name = {cfg['name']: copy.deepcopy(cfg) for cfg in base_space}
    baseline = copy.deepcopy(by_name.get('baseline_current', base_space[0]))

    top_names = [name for name in top_config_names_from_state(source_state_path, top_n)
                 if name in by_name][:max(1, top_n)]
    if not top_names:
        return _generate_static_refinement_space()

    configs = []
    seen_signatures = set()

    def register(cfg: Dict[str, Any]) -> None:
        signature = json.dumps({k: cfg[k] for k in REFINEMENT_KEYS}, sort_keys=True)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        configs.append(cfg)

    for family_name in top_names:
        family_cfg = copy.deepcopy(by_name[family_name])
        family_cfg['name'] = f"ref_{family_name}__self"
        register(family_cfg)

        for donor_name in top_names:
            if donor_name == family_name:
                continue

            donor_cfg = by_name[donor_name]
            donor_delta = {
                key: copy.deepcopy(donor_cfg[key])
                for key in REFINEMENT_KEYS
                if donor_cfg.get(key) != baseline.get(key)
            }
            if not donor_delta:
                continue

            hybrid_cfg = copy.deepcopy(by_name[family_name])
            hybrid_cfg.update(donor_delta)
            hybrid_cfg['name'] = f"ref_{family_name}__plus_{donor_name}"
            register(hybrid_cfg)

    return configs or _generate_static_refinement_space()


# =============================================================================
# VRAM-AWARE BATCH SIZE ESTIMATION
# =============================================================================

def estimate_batch_size(model: Model, target_vram_gb: float = 9.0) -> int:
    """
    Estimate safe batch size for a model given VRAM budget.

    Uses param count as a rough proxy. Calibrated against the known
    baseline: 35.2M params → batch 16 fits 12 GB (~9GB working).
    
    NOTE: Conservative max of 32 to avoid GPU event manager thrashing.
    Smaller models don't benefit from proportionally larger batches and
    create excessive GPU events that exhaust kernel FD limits.
    """
    n_params = model.count_params()

    # Reference: 35.2M params, batch 16, ~9GB working memory
    ref_params = 35_200_000
    ref_batch = 16
    ref_vram = 9.0

    # Memory scales roughly linearly with params and batch size
    # vram ≈ (n_params / ref_params) * (batch / ref_batch) * ref_vram
    # => batch = target_vram * ref_batch * ref_params / (n_params * ref_vram)

    estimated = target_vram_gb * ref_batch * ref_params / (n_params * ref_vram)
    # Clamp to [4, 32] to be conservative on GPU event creation
    # (prevents exhausting kernel FD limits even with smaller models)
    batch = int(estimated)
    batch = max(4, min(32, batch))

    # Round down to nearest multiple of 4
    batch = (batch // 4) * 4
    batch = max(4, batch)

    return batch


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model_path: str, n_samples: int = 15,
                   snr_levels: list = None) -> Dict[str, float]:
    """
    Evaluate a trained model across multiple SNR levels.

    Returns dict with per-SNR and overall metrics.
    """
    if snr_levels is None:
        snr_levels = [0, 5, 10, 15, 20]

    # Import the AudioDenoiser from inference script
    from importlib import import_module
    usar = import_module('3-usar_modelo_v2')
    denoiser = usar.AudioDenoiser(model_path)

    # Load test files
    clean_dir = 'dataset/clean_test'
    noise_dir = 'dataset/noise_test'

    clean_files = sorted([
        os.path.join(clean_dir, f)
        for f in os.listdir(clean_dir) if f.endswith('.wav')
    ])
    noise_files = sorted([
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir) if f.endswith('.wav')
    ])

    np.random.seed(42)  # Reproducible evaluation
    results_per_snr = {}

    for snr_db in snr_levels:
        improvements = []

        for i in range(n_samples):
            try:
                clean_audio, _ = load_audio(
                    clean_files[np.random.randint(len(clean_files))])
                noise_audio, _ = load_audio(
                    noise_files[np.random.randint(len(noise_files))])

                if len(clean_audio) < Config.SAMPLE_RATE:
                    continue

                # Mix at target SNR
                if len(noise_audio) < len(clean_audio):
                    noise_audio = np.tile(
                        noise_audio, (len(clean_audio) // len(noise_audio)) + 1)
                noise_audio = noise_audio[:len(clean_audio)]

                clean_power = np.mean(clean_audio ** 2) + 1e-8
                noise_power = np.mean(noise_audio ** 2) + 1e-8
                scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
                noisy_audio = clean_audio + noise_audio * scale

                peak = np.max(np.abs(noisy_audio))
                if peak > 0.99:
                    noisy_audio *= 0.99 / peak
                    clean_audio *= 0.99 / peak

                denoised = denoiser.denoise_audio(noisy_audio, show_progress=False)
                min_len = min(len(clean_audio), len(denoised))

                if min_len <= 0:
                    continue

                # Guard against invalid model outputs polluting ranking.
                if not np.isfinite(denoised[:min_len]).all():
                    continue

                sdr_before = compute_sdr(clean_audio[:min_len], noisy_audio[:min_len])
                sdr_after = compute_sdr(clean_audio[:min_len], denoised[:min_len])
                delta = sdr_after - sdr_before
                if np.isfinite(delta):
                    improvements.append(float(delta))

            except Exception:
                continue

        if improvements:
            results_per_snr[str(snr_db)] = {
                'mean_improvement': float(np.mean(improvements)),
                'median_improvement': float(np.median(improvements)),
                'std': float(np.std(improvements)),
                'min': float(np.min(improvements)),
                'max': float(np.max(improvements)),
                'n_samples': len(improvements),
            }

    # Aggregate
    all_means = [v['mean_improvement'] for v in results_per_snr.values()]
    overall = {
        'overall_mean_improvement': float(np.mean(all_means)) if all_means else float('nan'),
        'per_snr': results_per_snr,
    }
    return overall


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def config_id(cfg: dict) -> str:
    """Generate a stable short ID for a configuration"""
    return cfg['name']


def parse_only_configs(only_configs: str, all_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter configs by comma-separated names while preserving search-space order."""
    requested = [x.strip() for x in (only_configs or '').split(',') if x.strip()]
    if not requested:
        return all_configs

    requested_set = set(requested)
    known = {c['name'] for c in all_configs}
    unknown = sorted(requested_set - known)
    if unknown:
        raise ValueError(f"Unknown config name(s): {', '.join(unknown)}")

    return [c for c in all_configs if c['name'] in requested_set]


def top_config_names_from_state(state_path: str, top_n: int) -> List[str]:
    """Return top-N config names ranked by evaluation score from a state file."""
    if not os.path.exists(state_path):
        return []

    with open(state_path, 'r') as f:
        state = json.load(f)

    trials = state.get('trials', {})
    ranked = []
    for name, trial in trials.items():
        ev = trial.get('eval_results')
        if not ev:
            continue
        score = ev.get('overall_mean_improvement')
        if score is None:
            continue
        score = float(score)
        if not np.isfinite(score):
            continue
        ranked.append((name, score))

    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:max(1, top_n)]]


def run_stage_with_restarts(cmd: List[str], stage_name: str, max_restarts: int = 20) -> int:
    """Run a stage command and transparently retry on requested restart (exit code 2)."""
    attempt = 0
    while attempt < max_restarts:
        attempt += 1
        rc = subprocess.call(cmd)
        if rc == 2:
            print(f"\n{stage_name}: restart requested ({attempt}/{max_restarts}). Retrying stage...")
            continue
        return rc
    print(f"\n{stage_name}: exceeded restart limit ({max_restarts}).")
    return 2


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'trials': {}, 'best_trial': None, 'best_score': -999.0}


def save_state(state: dict):
    os.makedirs(SEARCH_DIR, exist_ok=True)
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def should_stop() -> bool:
    return os.path.exists(STOP_FILE)


def is_retriable_error(error_msg: str) -> bool:
    """Errors that should be retried on next pass instead of being skipped."""
    text = (error_msg or '').lower()
    retriable_patterns = [
        'oom',
        'resourceexhausted',
        'failed to allocate memory',
        'out of memory',
        'no algorithm worked',
        'mixed precision is disallowed',
        'different precisions',
        'internalerror',
        'graph execution error',
        'cudnn',
        'cuda',
        'file descriptor',
        'unable to get',
        'event status',
        'a kerastensor cannot be used as input to a tensorflow function',
    ]
    return any(p in text for p in retriable_patterns)


def is_memory_pressure_error(error_msg: str) -> bool:
    """Detect OOM-like errors that commonly leave CUDA memory fragmented."""
    text = (error_msg or '').lower()
    patterns = [
        'resourceexhausted',
        'out of memory',
        'failed to allocate memory',
        'no algorithm worked',
        'tf-allocator-allocation-error',
        'cuda_error_out_of_memory',
    ]
    return any(p in text for p in patterns)


# =============================================================================
# SINGLE TRIAL
# =============================================================================

def run_trial(cfg: dict, state: dict, max_epochs: int,
              train_dataset, val_dataset, train_steps, val_steps,
              input_shape: tuple,
              use_mixed_precision: bool = True,
              eval_samples: int = 15,
              dataset_fraction: float = 1.0) -> dict:
    """
    Train and evaluate a single configuration.

    Returns updated trial record.
    """
    trial_name = config_id(cfg)
    trial_dir = os.path.join(SEARCH_DIR, trial_name)
    os.makedirs(trial_dir, exist_ok=True)

    trial = state['trials'].get(trial_name, {
        'config': cfg,
        'status': 'pending',
        'epochs_completed': 0,
        'best_val_loss': None,
        'eval_results': None,
        'model_path': None,
        'params': None,
        'train_time_s': 0,
        'error': None,
        'needs_restart': False,
    })

    if trial['status'] == 'completed':
        return trial

    print(f"\n{'='*70}")
    print(f"  TRIAL: {trial_name}")
    print(f"{'='*70}")
    for k, v in cfg.items():
        if k != 'name':
            print(f"  {k}: {v}")

    best_model_path = os.path.join(trial_dir, 'best.keras')
    csv_log_path = os.path.join(trial_dir, 'training.csv')
    backup_dir = os.path.join(trial_dir, 'backup')

    # Precision mode is fixed for the run; no per-trial precision fallback.
    if not use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('float32')
        print("  Precision mode: float32")
    else:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("  Precision mode: mixed_float16")

    # Determine initial epoch for resume
    initial_epoch = trial.get('epochs_completed', 0)
    if initial_epoch >= max_epochs:
        trial['status'] = 'evaluating'
    else:
        trial['status'] = 'training'
        save_state(state)

    # ── BUILD / LOAD MODEL ──
    try:
        runtime_cfg = copy.deepcopy(cfg)
        if use_mixed_precision and runtime_cfg.get('dropout', 0.0) > 0.0:
            # FP16 path on this stack is unstable with dropout SelectV2 kernels.
            # Keep tensor-core acceleration and disable dropout for search trials.
            runtime_cfg['dropout'] = 0.0
            print("  Runtime adjustment: dropout -> 0.0 for FP16 stability")

        lr = runtime_cfg.get('learning_rate', 1e-3)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=1e-4,
            beta_1=0.9, beta_2=0.999, epsilon=1e-7,
        )
        loss_fn = make_combined_loss(*runtime_cfg['loss_weights'])

        if initial_epoch > 0 and os.path.exists(best_model_path):
            print(f"  Resuming from epoch {initial_epoch}")
            model = tf.keras.models.load_model(
                best_model_path,
                compile=False,
                custom_objects={
                    'combined_loss': make_combined_loss(*cfg['loss_weights']),
                    'mask_mse': mask_mse, 'mask_mae': mask_mae,
                    'ConvBlock': ConvBlock, 'ResidualBlock': ResidualBlock,
                    'ResizeToMatch': ResizeToMatch, 'AttentionGate': AttentionGate,
                    'SafeDropout': SafeDropout,
                }
            )
        else:
            model = build_model(runtime_cfg, input_shape)

            batch_size = cfg.get('batch_size') or estimate_batch_size(model)
            trial['batch_size_used'] = batch_size
            trial['params'] = int(model.count_params())

            print(f"  Parameters: {model.count_params():,}")
            print(f"  Batch size: {batch_size}")

        # Always compile explicitly to enforce non-XLA training path.
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[mask_mse, mask_mae],
            jit_compile=False,
        )

    except Exception as e:
        err = str(e)
        if is_memory_pressure_error(err):
            trial['status'] = 'pending'
            trial['error'] = f'memory-pressure-build: {err}'
            trial['needs_restart'] = True
            print("  BUILD ERROR (memory pressure). Marking pending and requesting restart.")
            tf.keras.backend.clear_session()
            return trial

        trial['status'] = 'error'
        trial['error'] = err
        print(f"  BUILD ERROR: {e}")
        return trial

    if trial.get('params') is None:
        trial['params'] = int(model.count_params())

    batch_size = trial.get('batch_size_used') or cfg.get('batch_size')
    if batch_size is None:
        batch_size = estimate_batch_size(model)
        trial['batch_size_used'] = batch_size

    # Adjust steps for this batch size
    effective_train_steps = max(1, (train_steps * 16) // batch_size)
    effective_val_steps = max(1, (val_steps * 16) // batch_size)

    if dataset_fraction < 1.0:
        effective_train_steps = max(1, int(effective_train_steps * dataset_fraction))
        effective_val_steps = max(1, int(effective_val_steps * dataset_fraction))
        print(f"  Fast mode: dataset_fraction={dataset_fraction:.6f} "
              f"=> train_steps={effective_train_steps}, val_steps={effective_val_steps}")

    # Re-batch datasets for this trial's batch size
    # We unbatch then re-batch from the provided datasets
    trial_train = train_dataset.unbatch().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    trial_val = val_dataset.unbatch().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # ── TRAIN ──
    if trial['status'] == 'training':
        fast_screen_mode = dataset_fraction < 1.0
        monitor_metric = 'loss' if fast_screen_mode else 'val_loss'

        callbacks = [
            BackupAndRestore(backup_dir=backup_dir),
            ModelCheckpoint(
                filepath=best_model_path,
                monitor=monitor_metric, save_best_only=True,
                mode='min', verbose=0,
            ),
            EarlyStopping(
                monitor=monitor_metric, patience=10,
                restore_best_weights=True, mode='min', verbose=1,
            ),
            ReduceLROnPlateau(
                monitor=monitor_metric, factor=0.5, patience=4,
                min_lr=1e-6, mode='min', verbose=1,
            ),
            CSVLogger(csv_log_path, append=(initial_epoch > 0)),
        ]

        print(f"  Monitor metric: {monitor_metric}")

        print(f"  Training epochs {initial_epoch+1} → {max_epochs}  "
              f"(steps/epoch: {effective_train_steps}, val: {effective_val_steps})")

        t0 = time.time()
        try:
            history = model.fit(
                trial_train,
                epochs=max_epochs,
                initial_epoch=initial_epoch,
                steps_per_epoch=effective_train_steps,
                validation_data=trial_val,
                validation_steps=effective_val_steps,
                callbacks=callbacks,
                verbose=2,  # one line per epoch
            )
            elapsed = time.time() - t0
            trial['train_time_s'] = trial.get('train_time_s', 0) + elapsed

            # Record best tracked metric, ignoring NaN values.
            hist_values = history.history.get(monitor_metric, [])
            finite_values = [float(v) for v in hist_values if np.isfinite(v)]
            if finite_values:
                trial['best_val_loss'] = float(min(finite_values))
            else:
                trial['best_val_loss'] = None
            trial['epochs_completed'] = max_epochs

            # Ensure best model was saved
            if not os.path.exists(best_model_path):
                model.save(best_model_path)

        except tf.errors.ResourceExhaustedError:
            # OOM - try halving batch size
            print(f"  OOM! Halving batch size from {batch_size}")
            trial['batch_size_used'] = max(4, batch_size // 2)
            trial['status'] = 'pending'  # retry next run
            trial['error'] = 'OOM - batch size halved'
            tf.keras.backend.clear_session()
            return trial

        except tf.errors.InternalError as e:
            err = str(e)
            if ('mixed precision is disallowed' in err.lower()
                    or 'different precisions' in err.lower()):
                err_lines = [ln.strip() for ln in err.splitlines() if ln.strip()]
                detail = ''
                for ln in err_lines:
                    if 'op_name=' in ln or 'Seen floating point types' in ln:
                        detail = ln
                        break
                if not detail and err_lines:
                    detail = err_lines[-1]
                print("  Mixed-precision graph error in this trial.")
                if detail:
                    print(f"  Detail: {detail}")
                trial['status'] = 'error'
                trial['error'] = f'mixed-precision-error: {err}'
            else:
                trial['status'] = 'error'
                trial['error'] = err
                traceback.print_exc()
            tf.keras.backend.clear_session()
            return trial

        except Exception as e:
            err = str(e)
            if is_memory_pressure_error(err):
                print(f"  Memory pressure during training. Halving batch size from {batch_size}")
                trial['batch_size_used'] = max(4, batch_size // 2)
                trial['status'] = 'pending'
                trial['error'] = f'memory-pressure-train: {err}'
                trial['needs_restart'] = True
                tf.keras.backend.clear_session()
                return trial

            trial['status'] = 'error'
            trial['error'] = err
            print(f"  TRAINING ERROR: {e}")
            traceback.print_exc()
            tf.keras.backend.clear_session()
            return trial

    # ── EVALUATE ──
    if os.path.exists(best_model_path):
        print(f"  Evaluating model...")
        trial['status'] = 'evaluating'
        save_state(state)

        try:
            tf.keras.backend.clear_session()
            setup_gpu()
            eval_results = evaluate_model(best_model_path, n_samples=eval_samples)
            trial['eval_results'] = eval_results
            trial['model_path'] = best_model_path

            score = eval_results['overall_mean_improvement']
            print(f"  Overall SDR improvement: {score:+.2f} dB")

            for snr_k, snr_v in eval_results['per_snr'].items():
                print(f"    SNR {snr_k:>3s} dB: {snr_v['mean_improvement']:+.2f} dB  "
                      f"(median {snr_v['median_improvement']:+.2f})")

            trial['status'] = 'completed'

        except Exception as e:
            trial['status'] = 'error'
            trial['error'] = f'Eval error: {e}'
            print(f"  EVAL ERROR: {e}")
    else:
        trial['status'] = 'error'
        trial['error'] = 'No saved model found after training'

    tf.keras.backend.clear_session()
    return trial


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_results(state: dict):
    """Print a summary table of all completed trials."""
    trials = state['trials']
    completed = {k: v for k, v in trials.items() if v.get('eval_results')}

    if not completed:
        print("No completed trials yet.")
        return

    # Sort by overall score descending
    ranked = sorted(
        completed.items(),
        key=lambda kv: kv[1]['eval_results']['overall_mean_improvement'],
        reverse=True,
    )

    snr_keys = ['0', '5', '10', '15', '20']

    print(f"\n{'='*120}")
    print(f"  HYPERPARAMETER SEARCH RESULTS  ({len(ranked)} completed trials)")
    print(f"{'='*120}")

    header = (f"{'Rank':>4} {'Name':<25} {'Params':>10} {'Batch':>5} "
              f"{'ValLoss':>8} {'Overall':>8}")
    for s in snr_keys:
        header += f" {'SNR'+s:>7}"
    header += f" {'Time':>8}"

    print(header)
    print("-" * len(header))

    for rank, (name, trial) in enumerate(ranked, 1):
        ev = trial['eval_results']
        params = trial.get('params', 0)
        batch = trial.get('batch_size_used', '?')
        val_loss = trial.get('best_val_loss', 0)
        overall = ev['overall_mean_improvement']
        t = trial.get('train_time_s', 0)
        time_str = f"{t/60:.0f}m" if t < 3600 else f"{t/3600:.1f}h"

        row = (f"{rank:4d} {name:<25} {params:>10,} {batch:>5} "
               f"{val_loss:>8.4f} {overall:>+8.2f}")

        for s in snr_keys:
            if s in ev['per_snr']:
                row += f" {ev['per_snr'][s]['mean_improvement']:>+7.2f}"
            else:
                row += f" {'N/A':>7}"

        row += f" {time_str:>8}"
        print(row)

    print("-" * len(header))

    # Print best
    best_name, best_trial = ranked[0]
    print(f"\n  BEST: {best_name}")
    print(f"  Score: {best_trial['eval_results']['overall_mean_improvement']:+.2f} dB avg SDR improvement")
    print(f"  Model: {best_trial.get('model_path', 'N/A')}")

    # Also show worst SNR performance of best model
    per_snr = best_trial['eval_results']['per_snr']
    worst_snr = min(per_snr.items(), key=lambda kv: kv[1]['mean_improvement'])
    best_snr = max(per_snr.items(), key=lambda kv: kv[1]['mean_improvement'])
    print(f"  Best  SNR level: {best_snr[0]} dB ({best_snr[1]['mean_improvement']:+.2f} dB)")
    print(f"  Worst SNR level: {worst_snr[0]} dB ({worst_snr[1]['mean_improvement']:+.2f} dB)")

    print(f"{'='*120}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    global SEARCH_DIR, STATE_FILE, STOP_FILE
    parser = argparse.ArgumentParser(
        description='Automated hyperparameter search for audio denoiser')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs per trial (default: 50)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview configurations without training')
    parser.add_argument('--results', action='store_true',
                        help='Show results and exit')
    parser.add_argument('--eval-samples', type=int, default=15,
                        help='Eval samples per SNR level (default: 15)')
    parser.add_argument('--dataset-fraction', type=float, default=1.0,
                        help=('Fraction of dataset per epoch for fast screening '
                              '(0 < f <= 1, e.g. 0.002 for ~1/500)'))
    parser.add_argument('--quick-screen', action='store_true',
                        help='Shortcut for a fast viability pass (dataset_fraction=0.002, eval_samples=5)')
    parser.add_argument('--only-configs', type=str, default='',
                        help='Comma-separated config names to run (default: all)')
    parser.add_argument('--two-stage', action='store_true',
                        help='Run quick-screen for all configs, then full train top-N configs')
    parser.add_argument('--screen-epochs', type=int, default=6,
                        help='Epochs for stage-1 quick-screen (used with --two-stage)')
    parser.add_argument('--screen-fraction', type=float, default=0.02,
                        help='Dataset fraction for stage-1 quick-screen (used with --two-stage)')
    parser.add_argument('--screen-eval-samples', type=int, default=5,
                        help='Eval samples/SNR for stage-1 quick-screen (used with --two-stage)')
    parser.add_argument('--promote-top', type=int, default=5,
                        help='Top-N configs promoted to full stage (used with --two-stage)')
    parser.add_argument('--screen-dir', type=str, default='',
                        help='Directory for stage-1 quick-screen state (default: <search-dir>_quick)')
    parser.add_argument('--force-fp32', action='store_true',
                        help='Force float32 for all trials (fallback mode)')
    parser.add_argument('--allow-cpu', action='store_true',
                        help='Allow CPU-only execution when GPU is unavailable')
    parser.add_argument('--refinement', action='store_true',
                        help='Use refinement search space (permutations of top-3 traits) instead of main space')
    parser.add_argument('--refine-from-state', type=str, default='',
                        help='State file used to derive refinement families from the actual top completed configs')
    parser.add_argument('--search-dir', type=str, default='search_results',
                        help='Directory to store trial outputs and state')
    args = parser.parse_args()

    SEARCH_DIR = args.search_dir
    STATE_FILE = os.path.join(SEARCH_DIR, 'search_state.json')
    STOP_FILE = os.path.join(SEARCH_DIR, 'STOP')

    if args.quick_screen:
        if args.dataset_fraction == 1.0:
            args.dataset_fraction = 0.002
        if args.eval_samples == 15:
            args.eval_samples = 5

    if not (0 < args.dataset_fraction <= 1.0):
        parser.error('--dataset-fraction must be in (0, 1].')
    if not (0 < args.screen_fraction <= 1.0):
        parser.error('--screen-fraction must be in (0, 1].')
    if args.promote_top < 1:
        parser.error('--promote-top must be >= 1.')

    # Two-stage orchestrator: run this script twice via subprocess.
    if args.two_stage:
        script_path = os.path.abspath(__file__)
        screen_dir = args.screen_dir or f"{args.search_dir}_quick"

        stage1_cmd = [
            sys.executable, script_path,
            '--epochs', str(args.screen_epochs),
            '--dataset-fraction', str(args.screen_fraction),
            '--eval-samples', str(args.screen_eval_samples),
            '--search-dir', screen_dir,
        ]
        if args.only_configs:
            stage1_cmd += ['--only-configs', args.only_configs]
        if args.force_fp32:
            stage1_cmd += ['--force-fp32']
        if args.refinement:
            stage1_cmd += ['--refinement']
        if args.refine_from_state:
            stage1_cmd += ['--refine-from-state', args.refine_from_state]

        print("\n" + "=" * 70)
        print("  TWO-STAGE SEARCH: STAGE 1 (QUICK SCREEN)")
        print("=" * 70)
        print("  " + " ".join(stage1_cmd))
        rc1 = run_stage_with_restarts(stage1_cmd, 'Stage 1')
        if rc1 != 0:
            print(f"\nStage 1 exited with code {rc1}. Aborting stage 2.")
            sys.exit(rc1)

        top_names = top_config_names_from_state(
            os.path.join(screen_dir, 'search_state.json'),
            args.promote_top,
        )
        if not top_names:
            print("\nNo completed quick-screen trials found. Nothing to promote.")
            sys.exit(1)

        promoted_csv = ','.join(top_names)
        print(f"\nPromoting top {len(top_names)} config(s): {promoted_csv}")

        stage2_cmd = [
            sys.executable, script_path,
            '--epochs', str(args.epochs),
            '--dataset-fraction', str(args.dataset_fraction),
            '--eval-samples', str(args.eval_samples),
            '--search-dir', args.search_dir,
            '--only-configs', promoted_csv,
        ]
        if args.force_fp32:
            stage2_cmd += ['--force-fp32']
        if args.refinement:
            stage2_cmd += ['--refinement']
        if args.refine_from_state:
            stage2_cmd += ['--refine-from-state', args.refine_from_state]

        print("\n" + "=" * 70)
        print("  TWO-STAGE SEARCH: STAGE 2 (FULL TRAIN TOP CONFIGS)")
        print("=" * 70)
        print("  " + " ".join(stage2_cmd))
        rc2 = run_stage_with_restarts(stage2_cmd, 'Stage 2')
        sys.exit(rc2)

    os.makedirs(SEARCH_DIR, exist_ok=True)
    state = load_state()
    configs = (generate_refinement_space(args.refine_from_state)
               if args.refinement else generate_search_space())
    if args.only_configs:
        try:
            configs = parse_only_configs(args.only_configs, configs)
        except ValueError as e:
            parser.error(str(e))

    # Register state for signal handler to save on exit
    global _state_for_signal_handler
    _state_for_signal_handler = state

    # ── Just show results ──
    if args.results:
        print_results(state)
        return

    # ── Dry run ──
    if args.dry_run:
        print(f"\n{'='*70}")
        print(f"  SEARCH SPACE: {len(configs)} configurations")
        print(f"{'='*70}\n")

        setup_gpu()

        for i, cfg in enumerate(configs, 1):
            tid = config_id(cfg)
            status = state['trials'].get(tid, {}).get('status', 'pending')

            model = build_model(cfg, (Config.N_FREQ_BINS, Config.N_TIME_FRAMES, 2))
            n_params = model.count_params()
            batch = cfg.get('batch_size') or estimate_batch_size(model)

            print(f"  [{i:2d}/{len(configs)}] {tid:<25} "
                  f"params={n_params:>10,}  batch={batch:>3}  "
                  f"status={status}")

            tf.keras.backend.clear_session()

        pending = sum(1 for c in configs
                      if state['trials'].get(config_id(c), {}).get('status', 'pending')
                      in ('pending', 'training'))
        done = sum(1 for c in configs
                   if state['trials'].get(config_id(c), {}).get('status') == 'completed')
        print(f"\n  Pending: {pending}  Completed: {done}  Total: {len(configs)}")
        return

    # ── RUN SEARCH ──
    # Remove stop file if present
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    setup_gpu()
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus and not args.allow_cpu:
        print("ERROR: No GPU detected. Refusing CPU fallback for search.")
        print("Hint: rerun with --allow-cpu only if this is intentional.")
        sys.exit(3)

    # Disable XLA JIT for stability in long-running unattended search.
    tf.config.optimizer.set_jit(False)
    if args.force_fp32:
        print("Mixed precision: DISABLED (using float32 for stability)")
    else:
        print("Mixed precision: ENABLED (Tensor Cores active)")

    # Load dataset once (shared across all trials, cached in RAM)
    print("Loading dataset into memory (shared across all trials)...")
    train_dataset, val_dataset, train_steps, val_steps = load_tfrecord_dataset(
        batch_size=16  # base batch size; trials will re-batch
    )

    # Force dataset caching now by iterating a single element
    for _ in train_dataset.take(1):
        pass
    for _ in val_dataset.take(1):
        pass

    input_shape = (Config.N_FREQ_BINS, Config.N_TIME_FRAMES, 2)

    total = len(configs)
    completed_count = sum(
        1 for c in configs
        if state['trials'].get(config_id(c), {}).get('status') == 'completed'
    )

    print(f"\n{'='*70}")
    print(f"  HYPERPARAMETER SEARCH")
    print(f"  {total} configurations, {completed_count} already completed")
    print(f"  {total - completed_count} remaining")
    print(f"  Max epochs per trial: {args.epochs}")
    print(f"  Dataset fraction: {args.dataset_fraction:.6f}")
    print(f"  Eval samples/SNR: {args.eval_samples}")
    print(f"  Stop file: {os.path.abspath(STOP_FILE)}")
    print(f"{'='*70}\n")

    search_start = time.time()

    for idx, cfg in enumerate(configs, 1):
        tid = config_id(cfg)

        # Check stop conditions
        if should_stop():
            print(f"\n  STOP file detected. Saving state and exiting.")
            break

        existing = state['trials'].get(tid, {})
        if existing.get('status') == 'completed':
            continue
        if (existing.get('status') == 'error'
                and not is_retriable_error(str(existing.get('error', '')))):
            print(f"  Skipping {tid} (previous error: {existing.get('error', '?')})")
            continue

        # Progress header
        done_so_far = sum(
            1 for c in configs
            if state['trials'].get(config_id(c), {}).get('status') == 'completed'
        )
        elapsed = time.time() - search_start
        print(f"\n  [{done_so_far + 1}/{total}] Starting trial: {tid}")
        if elapsed > 60:
            print(f"  Search elapsed: {elapsed/3600:.1f}h")

        try:
            trial = run_trial(
                cfg, state, args.epochs,
                train_dataset, val_dataset, train_steps, val_steps,
                input_shape,
                use_mixed_precision=not args.force_fp32,
                eval_samples=args.eval_samples,
                dataset_fraction=args.dataset_fraction,
            )
            state['trials'][tid] = trial

            if trial.get('needs_restart'):
                save_state(state)
                print("  Restart requested to recover GPU memory state.")
                sys.exit(2)

            # Track global best
            if (trial.get('eval_results')
                    and trial['eval_results']['overall_mean_improvement']
                    > state.get('best_score', -999)):
                state['best_trial'] = tid
                state['best_score'] = trial['eval_results']['overall_mean_improvement']
                print(f"  ★ NEW BEST: {tid} ({state['best_score']:+.2f} dB)")

        except KeyboardInterrupt:
            print(f"\n  Interrupted during trial {tid}. Saving state...")
            save_state(state)
            print_results(state)
            return

        except Exception as e:
            print(f"  Unexpected error in trial {tid}: {e}")
            traceback.print_exc()
            state['trials'][tid] = {
                'config': cfg, 'status': 'error',
                'error': str(e), 'eval_results': None,
            }

        save_state(state)

        # Cleanup GPU memory between trials
        tf.keras.backend.clear_session()

    # ── FINAL RESULTS ──
    save_state(state)
    print_results(state)

    total_time = time.time() - search_start
    print(f"Total search time: {total_time/3600:.1f} hours")


if __name__ == '__main__':
    main()
