import os
import tensorflow as tf
import math
# import numpy as np
from datetime import datetime
from tensorflow.keras.layers import ( # noqa
    Input, Conv2D, BatchNormalization, LeakyReLU, Concatenate, Add,
    Conv2DTranspose, Dropout, LayerNormalization # noqa
)
from tensorflow.keras.models import Model # noqa
from tensorflow.keras.optimizers import Adam # noqa
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping # noqa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors, not warnings

# Enable mixed precision for tensor cores
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Do not allocate all GPU memory at once
if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


# Configuration
BATCH_SIZE = 32  # Increased batch size
EPOCHS = 50
DATASET_DIR = './dataset'
MODEL_DIR = './models'
LOG_DIR = './logs'

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# Parse TFRecord function
def parse_tfrecord_fn(example):
    feature_description = {
        'noisy_real': tf.io.VarLenFeature(tf.float32),
        'noisy_imag': tf.io.VarLenFeature(tf.float32),
        'mask_real': tf.io.VarLenFeature(tf.float32),
        'mask_imag': tf.io.VarLenFeature(tf.float32),
        'clean_file': tf.io.FixedLenFeature([], tf.string),
        'noisy_file': tf.io.FixedLenFeature([], tf.string),
        'snr': tf.io.FixedLenFeature([], tf.float32),
        'segment_idx': tf.io.FixedLenFeature([], tf.int64)
    }

    example = tf.io.parse_single_example(example, feature_description)

    # Reshape dense tensors
    noisy_real = tf.reshape(tf.sparse.to_dense(
        example['noisy_real']), (129, 126))
    noisy_imag = tf.reshape(tf.sparse.to_dense(
        example['noisy_imag']), (129, 126))
    mask_real = tf.reshape(tf.sparse.to_dense(
        example['mask_real']), (129, 126))
    mask_imag = tf.reshape(tf.sparse.to_dense(
        example['mask_imag']), (129, 126))

    # Stack real and imaginary parts as channels
    inputs = tf.stack([noisy_real, noisy_imag], axis=-1)
    targets = tf.stack([mask_real, mask_imag], axis=-1)

    return inputs, targets


# Create dataset
def create_dataset(data_dir, batch_size, validation_split=0.1):
    filenames = tf.data.Dataset.list_files(
        os.path.join(data_dir, "*.tfrecord"))

    # Shuffle and split into training and validation
    n_files = len(list(filenames))
    n_val = int(n_files * validation_split)
    n_train = n_files - n_val

    # Shuffle the filenames
    filenames = filenames.shuffle(buffer_size=n_files)

    # Split into train and validation
    train_filenames = filenames.take(n_train)
    val_filenames = filenames.skip(n_train)

    # Count examples before batching and repeating
    raw_train_dataset = train_filenames.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    train_count = raw_train_dataset.reduce(0, lambda x, _: x + 1).numpy()

    # Same for validation
    raw_val_dataset = val_filenames.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    val_count = raw_val_dataset.reduce(0, lambda x, _: x + 1).numpy()

    # Create final datasets with batching and repeating
    train_dataset = raw_train_dataset.shuffle(buffer_size=1000).batch(
        batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).repeat()

    val_dataset = raw_val_dataset.batch(batch_size).prefetch(
        buffer_size=tf.data.AUTOTUNE).repeat()

    return train_dataset, val_dataset, train_count, val_count


# Create context-aware dataset
def create_context_aware_dataset(data_dir, batch_size, context_frames=3, validation_split=0.1):
    filenames = tf.data.Dataset.list_files(
        os.path.join(data_dir, "*.tfrecord"))

    # Shuffle and split into training and validation
    n_files = len(list(filenames))
    n_val = int(n_files * validation_split)
    n_train = n_files - n_val

    # Shuffle the filenames
    filenames = filenames.shuffle(buffer_size=n_files)

    # Split into train and validation
    train_filenames = filenames.take(n_train)
    val_filenames = filenames.skip(n_train)

    # Parse TFRecords
    raw_train_dataset = train_filenames.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    raw_val_dataset = val_filenames.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    def add_context(inputs, targets):
        """Simplified context window creation without loops"""
        # Using Python indexing directly for simplicity
        output_size = tf.shape(inputs)[1] - 2 * context_frames

        # Pre-allocate an array for all windows
        all_windows = []

        # Create windows without loops
        for i in range(output_size):
            # Extract window for each position
            window = inputs[:, i:i + 2 * context_frames + 1, :]
            all_windows.append(window)

        # Stack windows outside the loop
        stacked_windows = tf.stack(all_windows)

        # Get center targets
        center_targets = targets[:, context_frames:tf.shape(inputs)[1] - context_frames, :]

        # Create dataset directly
        return tf.data.Dataset.from_tensor_slices((stacked_windows, center_targets))

    def apply_augmentation(inputs, targets):
        # Random gain adjustment (volume variation)
        gain = tf.random.uniform([], 0.8, 1.2)
        inputs = inputs * gain

        # Keep targets unchanged - the mask should be scale-invariant
        return inputs, targets

    # Apply context to training and validation datasets
    train_dataset = raw_train_dataset.flat_map(add_context)
    val_dataset = raw_val_dataset.flat_map(add_context)

    # Count examples before batching and repeating
    train_count = raw_train_dataset.reduce(0, lambda x, _: x + 1).numpy()
    val_count = raw_val_dataset.reduce(0, lambda x, _: x + 1).numpy()

    train_dataset = train_dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    # Create final datasets with batching and repeating
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(
        batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).repeat()

    val_dataset = val_dataset.batch(batch_size).prefetch(
        buffer_size=tf.data.AUTOTUNE).repeat()

    return train_dataset, val_dataset, train_count, val_count


# Custom layer for safe concatenation
class SafeConcatenate(tf.keras.layers.Layer):
    def __init__(self, axis=-1):
        super(SafeConcatenate, self).__init__()
        self.axis = axis
        self.concat = Concatenate(axis=axis)

    def call(self, inputs):
        x1, x2 = inputs
        # Always resize x1 to match x2's spatial dimensions
        x1_resized = tf.image.resize(x1, [tf.shape(x2)[1], tf.shape(x2)[2]])
        return self.concat([x1_resized, x2])


# Build improved U-Net model with residual connections and phase attention
def build_improved_temporal_unet(input_shape, context_frames=3):
    # Calculate input shape with context
    temporal_input_shape = (2 * context_frames + 1,) + input_shape

    # Create input layer for temporal data
    inputs = Input(shape=temporal_input_shape)

    # Process temporal information with ConvLSTM
    x = tf.keras.layers.ConvLSTM2D(64, kernel_size=3, padding='same',
                                   return_sequences=False)(inputs)

    def conv_block(inputs, filters, kernel_size=3, strides=1, use_bn=True, activation=True):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)  # Add L2 regularization
        if use_bn:
            x = BatchNormalization()(x)
        if activation:
            x = LeakyReLU(alpha=0.2)(x)
        return x

    # Residual block
    def residual_block(inputs, filters):
        x = conv_block(inputs, filters)
        x = conv_block(x, filters, activation=False)
        x = Add()([x, inputs])
        x = LeakyReLU(alpha=0.2)(x)
        return x

    # Phase-sensitive attention module
    def phase_attention(x):
        # Extract phase information from complex input
        real, imag = tf.split(x, 2, axis=-1)
        phase = tf.math.atan2(imag, real)

        # Get the number of channels from the input tensor
        input_channels = x.shape[-1]

        # Create attention weights based on phase coherence with MATCHING channels
        attn = Conv2D(input_channels, 1, activation='sigmoid')(phase)
        return x * attn

    # Encoder path - INCREASE CAPACITY
    e1 = conv_block(x, 64, kernel_size=7, use_bn=False)  # Using ConvLSTM output
    e1 = Dropout(0.1)(e1)

    e2 = conv_block(e1, 128, strides=2)  # Increase from 64
    e2 = residual_block(e2, 128)
    e2 = residual_block(e2, 128)  # Add extra residual block
    e2 = Dropout(0.1)(e2)

    e3 = conv_block(e2, 256, strides=2)  # Increase from 128
    e3 = residual_block(e3, 256)
    e3 = residual_block(e3, 256)
    e3 = residual_block(e3, 256)  # Add extra residual block
    e3 = Dropout(0.2)(e3)

    # Bottleneck with phase attention
    b = conv_block(e3, 512, strides=2)
    b = phase_attention(b)  # Add phase-aware attention
    b = residual_block(b, 512)
    b = residual_block(b, 512)
    b = residual_block(b, 512)

    # Decoder path - add channel adjustment after concat
    d3 = Conv2DTranspose(256, 4, strides=2, padding='same')(b)
    d3 = SafeConcatenate()([d3, e3])
    d3 = Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(d3)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(alpha=0.2)(d3)
    d3 = residual_block(d3, 256)
    d3 = residual_block(d3, 256)
    d3 = Dropout(0.2)(d3)

    d2 = Conv2DTranspose(128, 4, strides=2, padding='same')(d3)
    d2 = SafeConcatenate()([d2, e2])
    d2 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(d2)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(alpha=0.2)(d2)
    d2 = residual_block(d2, 128)
    d2 = Dropout(0.1)(d2)

    d1 = Conv2DTranspose(64, 4, strides=2, padding='same')(d2)
    d1 = SafeConcatenate()([d1, e1])
    d1 = Conv2D(64, 1, padding='same', kernel_initializer='he_normal')(d1)
    d1 = BatchNormalization()(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    d1 = conv_block(d1, 64)
    d1 = Dropout(0.1)(d1)

    # Output - complex mask (2 channels: real and imaginary)
    outputs = Conv2D(2, 3, padding='same', activation='tanh')(d1)

    model = Model(inputs, outputs)
    return model

# Build improved U-Net model with residual connections and phase attention
def build_improved_unet_model(input_shape):
    # Helper function for conv blocks
    def conv_block(inputs, filters, kernel_size=3, strides=1, use_bn=True, activation=True):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal')(inputs)
        if use_bn:
            x = BatchNormalization()(x)
        if activation:
            x = LeakyReLU(alpha=0.2)(x)
        return x

    # Residual block
    def residual_block(inputs, filters):
        x = conv_block(inputs, filters)
        x = conv_block(x, filters, activation=False)
        x = Add()([x, inputs])
        x = LeakyReLU(alpha=0.2)(x)
        return x

    # Phase-sensitive attention module
    def phase_attention(x):
        # Extract phase information from complex input
        real, imag = tf.split(x, 2, axis=-1)
        phase = tf.math.atan2(imag, real)

        # Get the number of channels from the input tensor
        input_channels = x.shape[-1]

        # Create attention weights based on phase coherence with MATCHING channels
        attn = Conv2D(input_channels, 1, activation='sigmoid')(phase)
        return x * attn

    # Input
    inputs = Input(shape=input_shape)

    # Encoder path - INCREASE CAPACITY
    e1 = conv_block(inputs, 64, kernel_size=7, use_bn=False)  # Increase from 32
    e1 = Dropout(0.1)(e1)

    e2 = conv_block(e1, 128, strides=2)  # Increase from 64
    e2 = residual_block(e2, 128)
    e2 = residual_block(e2, 128)  # Add extra residual block
    e2 = Dropout(0.1)(e2)

    e3 = conv_block(e2, 256, strides=2)  # Increase from 128
    e3 = residual_block(e3, 256)
    e3 = residual_block(e3, 256)
    e3 = residual_block(e3, 256)  # Add extra residual block
    e3 = Dropout(0.2)(e3)

    # Bottleneck with phase attention
    b = conv_block(e3, 512, strides=2)
    b = phase_attention(b)  # Add phase-aware attention
    b = residual_block(b, 512)
    b = residual_block(b, 512)
    b = residual_block(b, 512)

    # Decoder path - add channel adjustment after concat
    d3 = Conv2DTranspose(256, 4, strides=2, padding='same')(b)
    d3 = SafeConcatenate()([d3, e3])
    d3 = Conv2D(256, 1, padding='same', kernel_initializer='he_normal')(d3)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(alpha=0.2)(d3)
    d3 = residual_block(d3, 256)
    d3 = residual_block(d3, 256)
    d3 = Dropout(0.2)(d3)

    d2 = Conv2DTranspose(128, 4, strides=2, padding='same')(d3)
    d2 = SafeConcatenate()([d2, e2])
    d2 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(d2)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(alpha=0.2)(d2)
    d2 = residual_block(d2, 128)
    d2 = Dropout(0.1)(d2)

    d1 = Conv2DTranspose(64, 4, strides=2, padding='same')(d2)
    d1 = SafeConcatenate()([d1, e1])
    d1 = Conv2D(64, 1, padding='same', kernel_initializer='he_normal')(d1)
    d1 = BatchNormalization()(d1)
    d1 = LeakyReLU(alpha=0.2)(d1)
    d1 = conv_block(d1, 64)
    d1 = Dropout(0.1)(d1)

    # Output - complex mask (2 channels: real and imaginary)
    outputs = Conv2D(2, 3, padding='same', activation='tanh')(d1)

    model = Model(inputs, outputs)
    return model

# Loss function - improved complex-aware loss
def improved_complex_loss(y_true, y_pred):
    # Original MSE components - cast to float32 for stability
    real_loss = tf.cast(tf.reduce_mean(tf.square(y_true[..., 0] - y_pred[..., 0])), tf.float32)
    imag_loss = tf.cast(tf.reduce_mean(tf.square(y_true[..., 1] - y_pred[..., 1])), tf.float32)

    # Add magnitude consistency loss (perceptually important)
    y_true_mag = tf.sqrt(tf.square(y_true[..., 0]) + tf.square(y_true[..., 1]) + 1e-6)
    y_pred_mag = tf.sqrt(tf.square(y_pred[..., 0]) + tf.square(y_pred[..., 1]) + 1e-6)

    # Cast magnitudes to float32 for more stable loss calculation
    y_true_mag = tf.cast(y_true_mag, tf.float32)
    y_pred_mag = tf.cast(y_pred_mag, tf.float32)

    # Calculate magnitude loss (already in float32)
    magnitude_loss = tf.reduce_mean(tf.square(y_true_mag - y_pred_mag))

    # Higher weight on speech frequencies (1000-4000 Hz)
    speech_mask = tf.ones_like(y_true_mag)
    speech_mask = speech_mask * tf.constant([1.0 if 32 <= i < 128 else 0.5 for i in range(129)],
                                            dtype=tf.float32)[:, tf.newaxis]

    weighted_mag_loss = tf.reduce_mean(tf.square(y_true_mag - y_pred_mag) * speech_mask)

    # Combined loss - now using both magnitude losses (all in float32)
    return 0.4 * real_loss + 0.4 * imag_loss + 0.1 * magnitude_loss + 0.1 * weighted_mag_loss

# Create mel-scale frequency weights for perceptual emphasis
def create_mel_weights(n_freq_bins, emphasis_factor=2.0):
    """
    Creates weights based on mel scale for emphasizing perceptually important frequencies.

    Args:
        n_freq_bins: Number of frequency bins (129 for your STFT)
        emphasis_factor: How much to emphasize mid-range frequencies

    Returns:
        A tensor of shape [n_freq_bins, 1] with mel-based weights
    """
    # Approximate frequency values for each bin (assuming standard STFT with 44.1kHz sampling)
    # For 129 bins, this roughly corresponds to 0-22050 Hz
    freq_hz = tf.linspace(0., 22050., n_freq_bins)

    # Convert to mel scale (emphasized mid-range frequencies)
    # Formula: m = 2595 * log10(1 + f/700)
    mel = 2595.0 * tf.math.log(1.0 + freq_hz / 700.0) / tf.math.log(10.0)

    # Normalize to 0-1 range
    mel_normalized = mel / tf.reduce_max(mel)

    # Create emphasis curve (emphasize speech frequencies ~300-3000 Hz)
    # This creates a curve that peaks in the speech range
    weights = tf.pow(4.0 * mel_normalized * (1.0 - mel_normalized), emphasis_factor)

    # Add small constant to avoid zero weights
    weights = weights + 0.1

    # Normalize again to keep reasonable scale
    weights = weights / tf.reduce_mean(weights)

    # Reshape for broadcasting
    return tf.cast(tf.reshape(weights, [n_freq_bins, 1]), tf.float32)

def enhanced_perceptual_loss(y_true, y_pred):
    # Existing complex loss components
    complex_loss = improved_complex_loss(y_true, y_pred)

    # Add perceptual weighting based on loudness perception
    mel_weights = create_mel_weights(129)  # Create weights for your frequency bins

    # Apply mel weighting to magnitude differences (this is perceptually important)
    y_true_mag = tf.sqrt(tf.square(y_true[..., 0]) + tf.square(y_true[..., 1]) + 1e-6)
    y_pred_mag = tf.sqrt(tf.square(y_pred[..., 0]) + tf.square(y_pred[..., 1]) + 1e-6)
    
    # Cast to float32 explicitly before multiplication
    y_true_mag = tf.cast(y_true_mag, tf.float32)
    y_pred_mag = tf.cast(y_pred_mag, tf.float32)
    
    # Now both tensors are float32
    weighted_loss = tf.reduce_mean(mel_weights * tf.square(y_true_mag - y_pred_mag))

    return complex_loss + 0.2 * weighted_loss

# Custom metric: SNR improvement
def snr_improvement(y_true, y_pred):
    # This is a placeholder - actual SNR improvement would need original signal
    # For training feedback purpose, use MSE as a proxy
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return -10 * tf.math.log(mse) / tf.math.log(10.0)  # Convert to dB scale

# Add this before the train() function
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)

    def __call__(self, step):
        # Convert to float
        step = tf.cast(step, tf.float32)

        # Linear warmup
        if step < self.warmup_steps:
            return self.initial_learning_rate * (step / self.warmup_steps)

        # Cosine decay
        cosine_decay = tf.cos(
            (step - self.warmup_steps) /
            (self.total_steps - self.warmup_steps) *
            (math.pi / 2)
        )

        return self.initial_learning_rate * cosine_decay

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }

# Main training function
def train():
    # Use your existing regular dataset function instead of context-aware
    train_dataset, val_dataset, train_count, val_count = create_dataset(
        DATASET_DIR, BATCH_SIZE)

    # Build model with standard input (no temporal context in data)
    # The model will process one spectrogram frame at a time
    input_shape = (129, 126, 2)  # Height, Width, Channels (real + imag)
    model = build_improved_unet_model(input_shape)  # Use non-temporal model

    # Compile model
    initial_learning_rate = 0.001
    warmup_steps = 1000
    total_steps = 15000

    # Create learning rate schedule
    lr_schedule = WarmupCosineDecay(initial_learning_rate, warmup_steps, total_steps)

    # Use it with the optimizer
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=enhanced_perceptual_loss,
        metrics=[snr_improvement]
    )

    # Model summary
    model.summary()

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(MODEL_DIR, f"model-{timestamp}" + "-epoch{epoch:03d}.keras")

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1,
            save_freq='epoch'
        ),
        TensorBoard(
            log_dir=os.path.join(LOG_DIR, timestamp),
            histogram_freq=1,
            update_freq='epoch'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            verbose=1,
            restore_best_weights=True
        )
    ]

    # Calculate steps per epoch
    steps_per_epoch = train_count // BATCH_SIZE
    validation_steps = val_count // BATCH_SIZE

    print(f"Training on {train_count} examples ({steps_per_epoch} steps per epoch)")
    print(f"Validating on {val_count} examples ({validation_steps} steps per epoch)")

    # Train model
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # Save final model
    model.save(os.path.join(MODEL_DIR, f"final_model_{timestamp}.keras"))
    print(f"Model saved to {MODEL_DIR}/final_model_{timestamp}.keras")

    return history, model


if __name__ == "__main__":
    print("Starting CNN audio denoiser training...")
    history, model = train()
    print("Training complete!")
