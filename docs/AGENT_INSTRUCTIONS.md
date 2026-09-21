# CNN Speech Denoiser - Agent Instructions

## Project Overview
This is a CNN-based speech audio denoiser that operates in the frequency domain (STFT). The project consists of three main scripts:

1. **1-gerar_tfrecord.py** - Dataset preprocessing: converts audio files to frequency domain features stored in TFRecords
2. **2-criar_modelo.py** - Model training: defines and trains the CNN architecture
3. **3-usar_modelo.py** - Inference: applies trained model to denoise audio files

## Hardware Specifications
- **GPU**: NVIDIA RTX 4070 with Tensor Cores (supports mixed precision FP16)
- **RAM**: 32GB system memory
- **CPU**: AMD Ryzen 5900x (12 cores, 24 threads)

## Dataset Structure
- **Clean speech**: `dataset/CleanSpeech_training/` (~383 WAV files)
- **Noisy speech**: `dataset/NoisySpeech_training/` (4 SNR levels: 0, 10, 20, 30 dB per clean file)
- **Sample rate**: Native 16kHz, resampled to 8kHz for processing
- **Format**: WAV files, 16-bit PCM

## Known Issues with Current Implementation

### 1. Data Pipeline Issues
- Per-sample min-max normalization breaks consistency across training/inference
- Log-magnitude scaling with different normalization params between training and inference
- Phase-aware scaling may introduce artifacts
- Predicting only magnitude, using noisy phase for reconstruction

### 2. Model Architecture Issues
- Using 8-frame context with single-frame prediction (inefficient)
- Very deep model with questionable skip connection implementation
- Final conv kernel of [129, 1] may be problematic
- L2 regularization on all layers may constrain learning

### 3. Training Issues
- Batch size of 64 may be suboptimal for RTX 4070
- CosineDecayRestarts may not be ideal for audio denoising
- Early stopping patience of 5 may be too aggressive
- Sharding-based validation split may leak information

### 4. Inference Issues
- Complex denormalization procedure prone to errors
- Scale factor calculation may distort signal
- RMS matching may introduce its own artifacts

## Correct Approach (Research-Based)

### Signal Processing
1. Use **Complex Ratio Mask (cRM)** approach - predict both magnitude AND phase
2. Use **global normalization** statistics computed over entire dataset
3. Use **512-point FFT** (better frequency resolution for speech at 16kHz)
4. Use **overlapping segments** (2-4 seconds) with proper overlap-add reconstruction
5. Keep audio at **16kHz** (better for speech quality)

### Model Architecture
1. **U-Net** style encoder-decoder with skip connections
2. **Two output channels**: real and imaginary parts of the mask
3. **Dilated convolutions** in bottleneck for larger receptive field
4. **Residual blocks** for stable training
5. Input: 2D spectrogram with 2 channels (real, imag)
6. Output: 2D complex mask with 2 channels (real, imag)

### Training Strategy
1. **Batch size**: 128-256 (leverage RTX 4070 tensor cores)
2. **Mixed precision**: Use FP16 for faster training
3. **Loss function**: Combination of:
   - MSE on mask components
   - SDR loss (perceptually meaningful)
   - Multi-resolution STFT loss (optional)
4. **Optimizer**: AdamW with cosine annealing
5. **Data augmentation**: Random gain, time-shift

### Reconstruction
1. Apply predicted complex mask to noisy STFT: `Clean = Noisy * Mask`
2. Use ISTFT with proper overlap-add
3. No post-processing needed if mask is well-trained

## Key Metrics
- **SDR (Signal-to-Distortion Ratio)**: Primary metric
- **PESQ**: Perceptual quality
- **STOI**: Intelligibility metric

## File Conventions
- Use `v2` suffix for new implementation files
- Keep old files for reference
- Store trained models in `models/` directory
- Store TFRecords in `tfrecords_v2/` directory

## Training Workflow
1. Generate dataset: `python 1-gerar_tfrecord_v2.py`
2. Train model: `python 2-criar_modelo_v2.py`
3. Test inference: `python 3-usar_modelo_v2.py`

## Performance Targets
- SNR improvement: > 5 dB
- Training time: < 4 hours per epoch
- Inference: Real-time or faster

## Common Pitfalls to Avoid
1. Don't use per-sample normalization
2. Don't ignore phase information
3. Don't use overly complex architectures
4. Don't forget to handle edge cases in audio segments
5. Don't use incompatible STFT parameters between training and inference
