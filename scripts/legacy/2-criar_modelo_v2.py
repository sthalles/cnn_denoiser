"""
Audio Denoiser - Model Training (v2)

This script trains a U-Net based CNN for speech denoising in the frequency domain.

Key improvements:
1. U-Net architecture with skip connections
2. Predicts Complex Ratio Mask (real + imag)
3. Multi-component loss (MSE + SDR + spectral)
4. Mixed precision training for RTX 4070 tensor cores
5. Proper learning rate scheduling
"""

import os
import sys
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Suppress ALL TensorFlow, CUDA, and XLA warnings BEFORE importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logging (0=all, 3=none)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Better async memory allocation
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Consistent GPU ordering
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'  # Suppress XLA warnings
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # Quiet XLA


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


# Install stderr filter
sys.stderr = StderrFilter(sys.__stderr__)

import warnings
warnings.filterwarnings('ignore')

# Suppress absl logging (source of many warnings)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from dataset_v2 import load_tfrecord_dataset
from config import Config, setup_gpu
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard,
    ReduceLROnPlateau, LearningRateScheduler
)
from tensorflow.keras import layers, Model, Input
import tensorflow as tf
import numpy as np

# Additional TF warning suppression
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# =============================================================================
# CUSTOM LAYERS
# =============================================================================

class ConvBlock(layers.Layer):
    """Convolutional block with BatchNorm and activation"""

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
        self.activation = layers.LeakyReLU(alpha=0.2)

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.use_activation:
            x = self.activation(x)
        return x


class ResidualBlock(layers.Layer):
    """Residual block for stable deep training"""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvBlock(filters, kernel_size=3)
        self.conv2 = ConvBlock(filters, kernel_size=3, activation=False)
        self.activation = layers.LeakyReLU(alpha=0.2)

    def call(self, x, training=None):
        residual = x
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = layers.Add()([x, residual])
        x = self.activation(x)
        return x


class ResizeToMatch(layers.Layer):
    """Resize layer to match target spatial dimensions (Keras 3.x compatible)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        """inputs = [tensor_to_resize, reference_tensor]"""
        x, reference = inputs
        # Get spatial dims from reference (height, width)
        target_size = (tf.shape(reference)[1], tf.shape(reference)[2])
        return tf.image.resize(x, target_size)
        

class AttentionGate(layers.Layer):
    """Attention gate for skip connections"""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.W_g = layers.Conv2D(filters, 1, padding='same')
        self.W_x = layers.Conv2D(filters, 1, padding='same')
        self.psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')
        self.bn = layers.BatchNormalization()
        self.resize = ResizeToMatch()

    def call(self, g, x, training=None):
        """
        g: gating signal from decoder (lower resolution)
        x: skip connection from encoder (higher resolution)
        """
        # Upsample g to match x's spatial dims
        g_upsampled = self.resize([g, x])

        g1 = self.W_g(g_upsampled)
        x1 = self.W_x(x)

        psi = tf.nn.relu(g1 + x1)
        psi = self.psi(psi)
        psi = tf.cast(psi, x.dtype)

        return x * psi


# =============================================================================
# U-NET MODEL
# =============================================================================

def build_unet_denoiser(input_shape):
    """
    Build U-Net model for audio denoising

    Input: (n_freq, n_time, 2) - [noisy_real, noisy_imag]
    Output: (n_freq, n_time, 2) - [mask_real, mask_imag]
    """
    inputs = Input(shape=input_shape, name='input')

    # =========================================================================
    # ENCODER
    # =========================================================================

    # Level 1: (257, 251, 2) -> (257, 251, 64)
    e1 = ConvBlock(64, kernel_size=7)(inputs)
    e1 = ConvBlock(64)(e1)
    e1 = layers.Dropout(0.1)(e1)

    # Level 2: (257, 251, 64) -> (129, 126, 128)
    e2 = ConvBlock(128, strides=2)(e1)
    e2 = ResidualBlock(128)(e2)
    e2 = ResidualBlock(128)(e2)
    e2 = layers.Dropout(0.1)(e2)

    # Level 3: (129, 126, 128) -> (65, 63, 256)
    e3 = ConvBlock(256, strides=2)(e2)
    e3 = ResidualBlock(256)(e3)
    e3 = ResidualBlock(256)(e3)
    e3 = layers.Dropout(0.2)(e3)

    # Level 4: (65, 63, 256) -> (33, 32, 512)
    e4 = ConvBlock(512, strides=2)(e3)
    e4 = ResidualBlock(512)(e4)
    e4 = ResidualBlock(512)(e4)
    e4 = layers.Dropout(0.2)(e4)

    # =========================================================================
    # BOTTLENECK with dilated convolutions for larger receptive field
    # =========================================================================

    b = ConvBlock(512)(e4)

    # Dilated convolutions
    b = layers.Conv2D(512, 3, padding='same',
                      dilation_rate=2, activation='relu')(b)
    b = layers.BatchNormalization()(b)
    b = layers.Conv2D(512, 3, padding='same',
                      dilation_rate=4, activation='relu')(b)
    b = layers.BatchNormalization()(b)
    b = layers.Conv2D(512, 3, padding='same',
                      dilation_rate=8, activation='relu')(b)
    b = layers.BatchNormalization()(b)

    b = ResidualBlock(512)(b)
    b = layers.Dropout(0.3)(b)

    # =========================================================================
    # DECODER with skip connections
    # =========================================================================

    # Level 4: (33, 32, 512) -> (65, 63, 256)
    d4 = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(b)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.LeakyReLU(alpha=0.2)(d4)

    # Resize to match e4 dimensions
    d4 = ResizeToMatch()([d4, e4])

    # Attention gate
    att4 = AttentionGate(256)(d4, e4)
    d4 = layers.Concatenate()([d4, att4])
    d4 = ConvBlock(256)(d4)
    d4 = ResidualBlock(256)(d4)
    d4 = layers.Dropout(0.2)(d4)

    # Level 3: (65, 63, 256) -> (129, 126, 128)
    d3 = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(d4)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.LeakyReLU(alpha=0.2)(d3)

    d3 = ResizeToMatch()([d3, e3])

    att3 = AttentionGate(128)(d3, e3)
    d3 = layers.Concatenate()([d3, att3])
    d3 = ConvBlock(128)(d3)
    d3 = ResidualBlock(128)(d3)
    d3 = layers.Dropout(0.1)(d3)

    # Level 2: (129, 126, 128) -> (257, 251, 64)
    d2 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(d3)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.LeakyReLU(alpha=0.2)(d2)

    d2 = ResizeToMatch()([d2, e2])

    att2 = AttentionGate(64)(d2, e2)
    d2 = layers.Concatenate()([d2, att2])
    d2 = ConvBlock(64)(d2)
    d2 = ResidualBlock(64)(d2)
    d2 = layers.Dropout(0.1)(d2)

    # Level 1: (257, 251, 64) -> (257, 251, 64)
    d1 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(d2)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.LeakyReLU(alpha=0.2)(d1)

    d1 = ResizeToMatch()([d1, e1])

    att1 = AttentionGate(64)(d1, e1)
    d1 = layers.Concatenate()([d1, att1])
    d1 = ConvBlock(64)(d1)
    d1 = ConvBlock(64)(d1)

    # =========================================================================
    # OUTPUT LAYER
    # =========================================================================

    # Match input dimensions exactly
    d1 = ResizeToMatch()([d1, inputs])

    # Output: complex mask (2 channels)
    # Using tanh to bound output, then scale
    # Masks typically in range [-2, 2] for reasonable SNRs
    outputs = layers.Conv2D(
        2, 3, padding='same', activation='tanh',
        dtype='float32', name='output'
    )(d1)

    # Scale to [-5, 5] range (matches mask clipping in dataset)
    outputs = outputs * 5.0

    model = Model(inputs, outputs, name='UNetDenoiser')
    return model


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

@tf.keras.utils.register_keras_serializable()
def mse_loss(y_true, y_pred):
    """Mean Squared Error on mask components"""
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.keras.utils.register_keras_serializable()
def spectral_convergence_loss(y_true, y_pred):
    """Spectral convergence loss"""
    return tf.norm(y_true - y_pred, ord='fro', axis=[-2, -1]) / (tf.norm(y_true, ord='fro', axis=[-2, -1]) + 1e-8)


@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    """
    Combined loss for mask prediction

    Components:
    1. MSE on real and imaginary parts
    2. Magnitude difference loss
    3. Spectral convergence loss
    """
    # Cast to float32 for stability
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # MSE loss on components
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Extract real and imaginary parts
    true_real = y_true[..., 0]
    true_imag = y_true[..., 1]
    pred_real = y_pred[..., 0]
    pred_imag = y_pred[..., 1]

    # Magnitude loss
    true_mag = tf.sqrt(tf.square(true_real) + tf.square(true_imag) + 1e-8)
    pred_mag = tf.sqrt(tf.square(pred_real) + tf.square(pred_imag) + 1e-8)
    mag_loss = tf.reduce_mean(tf.abs(true_mag - pred_mag))

    # Spectral convergence
    sc_loss = tf.reduce_mean(spectral_convergence_loss(y_true, y_pred))

    # Combine with weights
    total_loss = 0.5 * mse + 0.3 * mag_loss + 0.2 * sc_loss

    return total_loss


# =============================================================================
# METRICS
# =============================================================================

@tf.keras.utils.register_keras_serializable()
def mask_mse(y_true, y_pred):
    """MSE metric for monitoring"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.keras.utils.register_keras_serializable()
def mask_mae(y_true, y_pred):
    """MAE metric for monitoring"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


# =============================================================================
# TRAINING
# =============================================================================

def create_callbacks(model_name, log_dir):
    """Create training callbacks"""

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(
                Config.MODEL_DIR, f'{model_name}-best.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),

        # Save periodic checkpoints (every 10 epochs)
        ModelCheckpoint(
            filepath=os.path.join(
                Config.MODEL_DIR, f'{model_name}-{{epoch:03d}}.keras'),
            save_freq=10 * 886,  # 10 epochs * steps_per_epoch (will be adjusted at runtime)
            save_best_only=False,
            verbose=0
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),

        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=Config.MIN_LEARNING_RATE,
            mode='min',
            verbose=1
        ),

        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(log_dir, timestamp),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),

        # CSV logging
        tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, f'{model_name}_training.csv'),
            append=True
        )
    ]

    return callbacks


def find_latest_checkpoint(model_name, model_dir=None):
    """Find the latest checkpoint for resuming training"""
    if model_dir is None:
        model_dir = Config.MODEL_DIR
    
    import glob
    import re
    
    # Look for epoch checkpoints: denoiser_v2-001.keras, denoiser_v2-010.keras, etc.
    pattern = os.path.join(model_dir, f'{model_name}-[0-9][0-9][0-9].keras')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        # Try the best model
        best_path = os.path.join(model_dir, f'{model_name}-best.keras')
        if os.path.exists(best_path):
            return best_path, 0  # Unknown epoch, start from 0
        return None, 0
    
    # Parse epoch numbers and find the latest
    latest_epoch = 0
    latest_checkpoint = None
    
    for ckpt in checkpoints:
        match = re.search(r'-(\d{3})\.keras$', ckpt)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = ckpt
    
    return latest_checkpoint, latest_epoch


def train_model(
    model_name='denoiser_v2',
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    resume_from=None,
    auto_resume=True
):
    """
    Train the denoiser model
    
    Args:
        model_name: Name for saving model files
        epochs: Total epochs to train
        batch_size: Batch size
        learning_rate: Initial learning rate
        resume_from: Explicit path to resume from
        auto_resume: If True, automatically find and resume from latest checkpoint
    """
    # Setup
    setup_gpu()
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Determine starting epoch and checkpoint
    initial_epoch = 0
    checkpoint_path = resume_from
    
    if auto_resume and not resume_from:
        checkpoint_path, initial_epoch = find_latest_checkpoint(model_name)
        if checkpoint_path:
            print(f"\n{'='*60}")
            print(f"AUTO-RESUME: Found checkpoint at epoch {initial_epoch}")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"{'='*60}\n")

    # Load dataset
    print("Loading dataset...")
    train_dataset, val_dataset, train_steps, val_steps = load_tfrecord_dataset(
        batch_size=batch_size
    )

    # Get input shape from dataset
    for inputs, targets in train_dataset.take(1):
        input_shape = inputs.shape[1:]  # (n_freq, n_time, 2)
        print(f"Input shape: {input_shape}")
        print(f"Target shape: {targets.shape[1:]}")

    # Build or load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={
                'combined_loss': combined_loss,
                'mask_mse': mask_mse,
                'mask_mae': mask_mae,
                'ConvBlock': ConvBlock,
                'ResidualBlock': ResidualBlock,
                'ResizeToMatch': ResizeToMatch,
                'AttentionGate': AttentionGate
            }
        )
        print(f"Resuming from epoch {initial_epoch}")
    else:
        print("Building new model...")
        model = build_unet_denoiser(input_shape)

        # Compile
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=Config.WEIGHT_DECAY,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[mask_mse, mask_mae]
        )

    # Model summary
    model.summary()

    # Callbacks
    callbacks = create_callbacks(model_name, Config.LOG_DIR)

    # Train
    remaining_epochs = epochs - initial_epoch
    if remaining_epochs <= 0:
        print(f"Already completed {initial_epoch} epochs (target: {epochs})")
        return model, None
        
    print(f"\n{'='*60}")
    print(f"TRAINING SESSION")
    print(f"{'='*60}")
    print(f"Target epochs:    {epochs}")
    print(f"Starting from:    {initial_epoch}")
    print(f"Remaining:        {remaining_epochs}")
    print(f"Batch size:       {batch_size}")
    print(f"Steps per epoch:  {train_steps}")
    print(f"Validation steps: {val_steps}")
    print(f"{'='*60}\n")

    history = model.fit(
        train_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_path = os.path.join(Config.MODEL_DIR, f'{model_name}-final.keras')
    model.save(final_path)
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Final model saved to {final_path}")
    print(f"{'='*60}")

    return model, history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description='Train audio denoiser model')
    parser.add_argument('--name', default='denoiser_v2', help='Model name')
    parser.add_argument('--epochs', type=int,
                        default=Config.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int,
                        default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float,
                        default=Config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--resume', default=None,
                        help='Path to model to resume from (auto-detects if not specified)')
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='Disable automatic checkpoint detection')

    args = parser.parse_args()
    
    start_time = datetime.now()

    print("\n" + "=" * 60)
    print("AUDIO DENOISER - TRAINING SESSION")
    print("=" * 60)
    print(f"Started at:     {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model name:     {args.name}")
    print(f"Target epochs:  {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"Auto-resume:    {not args.no_auto_resume}")
    print("=" * 60 + "\n")

    try:
        model, history = train_model(
            model_name=args.name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume_from=args.resume,
            auto_resume=not args.no_auto_resume
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nTotal training time: {duration}")
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED BY USER")
        print("Model checkpoints saved - can resume with --auto-resume")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TRAINING ERROR: {e}")
        print("Model checkpoints saved - can resume with --auto-resume")
        print("=" * 60)
        raise

    print("\nTraining complete!")
