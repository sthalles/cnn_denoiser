# %%
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf # noqa
from tensorflow.keras import layers, models, Input # noqa
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # noqa

num_features = 129
window_length = 256
hop_length = 64
num_segments = 8
sr = 16000

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# não usar toda memoria da GPU
if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# %%
def tf_record_parser(record):
    keys_to_features = {
        'noisy_mag_features': tf.io.FixedLenFeature([num_features * 8 * 1], tf.float32),
        'clean_magnitude': tf.io.FixedLenFeature([num_features * 1 * 1], tf.float32)
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    noisy_mag_features = features['noisy_mag_features']
    clean_magnitude = features['clean_magnitude']

    # reshape input and annotation images
    noisy_mag_features = tf.reshape(noisy_mag_features, (num_features, num_segments, 1), name="noisy_mag_features")
    clean_magnitude = tf.reshape(clean_magnitude, (num_features, 1, 1), name="clean_magnitude")

    return noisy_mag_features, clean_magnitude

# def sdr_metric(y_true, y_pred):
#     # Flatten the tensors
#     y_true = K.flatten(y_true)
#     y_pred = K.flatten(y_pred)

#     # Compute the SDR
#     noise = y_true - y_pred
#     signal_power = K.sum(K.square(y_true))
#     noise_power = K.sum(K.square(noise))
#     sdr = 10 * K.log(signal_power / noise_power) / K.log(10.0)

#     return sdr

@tf.keras.utils.register_keras_serializable()
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

@tf.keras.utils.register_keras_serializable()
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

def goat_model_v3(input_shape, l2_strength):
    inputs = Input(shape=input_shape)
    x = inputs

    # Encoder

    x = tf.keras.layers.ZeroPadding2D(((4, 4), (0, 0)))(x)

    x = layers.Conv2D(filters=16, kernel_size=[9, 8], strides=[1, 1], padding='valid', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    skip0 = layers.Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', use_bias=False,
                          kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(skip0)
    x = skip0

    x = layers.Conv2D(filters=64, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    skip1 = layers.Conv2D(filters=128, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                          kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(skip1)
    x = skip1

    x = layers.Conv2D(filters=256, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    skip2 = layers.Conv2D(filters=384, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                          kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(skip2)
    x = skip2

    x = layers.Conv2D(filters=512, kernel_size=(7, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=640, kernel_size=(11, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    # ############### METADE ################

    x = layers.Conv2D(filters=512, kernel_size=(7, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=384, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = x + skip2

    x = layers.Conv2D(filters=256, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=128, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = x + skip1

    x = layers.Conv2D(filters=64, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = x + skip0

    x = layers.Conv2D(filters=16, kernel_size=(11, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # fim

    x = layers.Conv2D(filters=1, kernel_size=[129, 1], strides=[1, 1], padding='same', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=x)

    return model

def goat_model_v4(input_shape, l2_strength):
    inputs = Input(shape=input_shape)
    x = inputs

    # Encoder

    x = tf.keras.layers.ZeroPadding2D(((4, 4), (0, 0)))(x)

    x = layers.Conv2D(filters=32, kernel_size=[9, 8], strides=[1, 1], padding='valid', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    skip0 = layers.Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', use_bias=False,
                          kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(skip0)
    x = skip0

    x = layers.Conv2D(filters=128, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    skip1 = layers.Conv2D(filters=256, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                          kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(skip1)
    x = skip1

    x = layers.Conv2D(filters=512, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=768, kernel_size=(7, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    # ############### METADE ################

    x = layers.Conv2D(filters=512, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=256, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = x + skip1

    x = layers.Conv2D(filters=128, kernel_size=(5, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = x + skip0

    x = layers.Conv2D(filters=32, kernel_size=(11, 1), strides=(1, 1), padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # fim

    x = layers.Conv2D(filters=1, kernel_size=[129, 1], strides=[1, 1], padding='same', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=x)

    return model

def goat_model_v5(input_shape, l2_strength):
    inputs = Input(shape=input_shape)
    x = inputs

    # Initial feature extraction
    x = tf.keras.layers.ZeroPadding2D(((4, 4), (0, 0)))(x)
    x = layers.Conv2D(filters=128, kernel_size=[9, 8], strides=[1, 1], padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Encoder path with dense blocks
    skip_connections = []
    filter_sizes = [256, 512, 768, 1024]

    for filters in filter_sizes:
        # Dense block
        db = x
        for _ in range(4):  # 4 layers per dense block
            y = layers.Conv2D(filters // 4, (1, 1), padding='same')(db)
            y = layers.BatchNormalization()(y)
            y = layers.LeakyReLU(alpha=0.2)(y)

            y = layers.Conv2D(filters, (5, 1), padding='same')(y)
            y = layers.BatchNormalization()(y)
            y = layers.LeakyReLU(alpha=0.2)(y)

            db = layers.Concatenate()([db, y])

        x = layers.Conv2D(filters, (1, 1), padding='same')(db)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        skip_connections.append(x)

    # Bridge with dilated convolutions
    dilation_rates = [1, 2, 4, 8, 16]
    bridge = x
    for rate in dilation_rates:
        y = layers.Conv2D(1024, (3, 1), padding='same', dilation_rate=rate)(bridge)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(alpha=0.2)(y)
        bridge = layers.Add()([bridge, y])

    x = bridge

    # Decoder path with attention
    for filters, skip in zip(reversed(filter_sizes), reversed(skip_connections)):

        # Project x to match skip connection dimensions
        x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Now x and skip have same dimensions
        # Attention mechanism
        attention = layers.Multiply()([x, skip])
        x = layers.Concatenate()([x, attention])

        x = layers.Conv2D(filters, (5, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        # Residual connection
        x = layers.Add()([x, skip])

    # Final refinement
    x = layers.Conv2D(256, (5, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, (3, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Output layer
    x = layers.Conv2D(1, kernel_size=[129, 1], padding='same', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model

# %%
# Load the dataset
raw_dataset = tf.data.TFRecordDataset(['tfrecords/denoising8k.tfrecord'])

# Parse the dataset
full_dataset = raw_dataset.map(tf_record_parser)

# full_dataset = full_dataset.take(100000)  # Limit to 1 million samples for testing

# After loading and parsing the dataset
full_dataset = full_dataset.cache()

# samples = 1801216

# Create validation split using sharding
NUM_SHARDS = 5  # This will give us an 80/20 split
# Create validation split using sharding
val_dataset = full_dataset.shard(NUM_SHARDS, index=0)  # Take first shard (20%)
train_dataset = None
for i in range(1, NUM_SHARDS):
    shard = full_dataset.shard(NUM_SHARDS, index=i)
    if train_dataset is None:
        train_dataset = shard
    else:
        train_dataset = train_dataset.concatenate(shard)

# Prepare the datasets for training
batch_size = 64
train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Define the input shape for the model
input_shape = (num_features, num_segments, 1)

# Build the model
model = goat_model_v3(input_shape, l2_strength=0.0001)

# formula l2:
# loss = original_loss + l2_strength * sum(weight^2)

# Display the model summary
model.summary()

# Train the model
# steps_per_epoch = train_size // batch_size
epochs = 10000

# reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='loss',
#     factor=0.5,
#     patience=5,
#     min_lr=0.00001
# )

model_file_name = '8k-v7'
checkpoint_file_name = model_file_name + '-checkpoint.keras'

# lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
#     initial_learning_rate=3e-4,
#     first_decay_steps=1000,
#     t_mul=2.0,
#     m_mul=0.9
# )

# Use a more aggressive learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=3e-4,  # Back to original safer value
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=0.9
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    mode='min'
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_file_name,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./old/logs6',
    histogram_freq=1,
    write_graph=True,
    write_images=False
)

try:
    model = tf.keras.models.load_model(checkpoint_file_name)
    print('Loaded model from checkpoint')
except (IOError, ValueError) as e:
    print(f"Checkpoint not found or invalid. Starting from scratch. Error: {e}")

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # optimizer = tf.keras.optimizers.AdamW(
    #     learning_rate=lr_schedule,
    #     weight_decay=1e-4,
    #     clipnorm=1.0,  # Start with conservative clipping
    #     epsilon=1e-8
    # )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        # loss='mean_squared_error',
        # loss=tf.keras.losses.Huber(delta=0.7),  # Combines L1 and L2 losses
        metrics=['mse', sdr_metric]
    )

callbacks = [
    early_stopping_callback,
    checkpoint_callback,
    tensorboard_callback,
    tf.keras.callbacks.CSVLogger('logs/training_log-v7.csv')
]

# model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping_callback, checkpoint_callback, tensorboard_callback, reduce_lr_callback])
model.fit(train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_dataset)


# Save the model
model.save('models/' + model_file_name + '.keras')

# Evaluate the model on the validation dataset
val_predictions = model.predict(val_dataset)
val_targets = []
for _, targets in val_dataset:
    val_targets.extend(targets.numpy())

# Flatten the predictions and targets if necessary
val_predictions = np.array(val_predictions).flatten()
val_targets = np.array(val_targets).flatten()

# Calculate regression metrics
mse = mean_squared_error(val_targets, val_predictions)
mae = mean_absolute_error(val_targets, val_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(val_targets, val_predictions)

# Print metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
