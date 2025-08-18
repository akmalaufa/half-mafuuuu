# Half-MAFU-Net Training Guide

## Overview
This guide provides comprehensive instructions for training the enhanced Half-MAFU-Net with large kernel convolutions on your NVIDIA RTX A4000 GPU to achieve 95%+ F1 score.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the Model
```bash
python test_model.py
```

Windows note (optional DenseCRF):
- DenseCRF (pydensecrf) bersifat opsional. Di Windows sering perlu Microsoft Visual C++ Build Tools.
- Jika tidak terpasang, gunakan `--postprocess morph` saat validasi/test. Jika ingin CRF:
```bash
pip install -r requirements-optional-crf.txt
```

### 3. Start Training
```bash
python train.py --epochs 150 --batch_size 12 --lr 1e-4
```

## 📁 Project Structure
```
Half-MAFU-Net/
├── half_mafunet.py      # Enhanced model with large kernels
├── train.py             # Training script
├── test_model.py        # Model testing script
├── config.py            # Training configuration
├── requirements.txt     # Dependencies
├── dataset/             # Your dataset
│   ├── train/
│   │   ├── images/     # Training images
│   │   └── masks/      # Training masks
│   ├── val/
│   │   ├── images/     # Validation images
│   │   └── masks/      # Validation masks
│   └── test/
│       ├── images/     # Test images
│       └── masks/      # Test masks
└── checkpoints/         # Model checkpoints (created during training)
```

## 🎯 Training Configuration

### Optimal Settings for RTX A4000
- **Batch Size**: 12 (optimized for 16GB VRAM)
- **Image Size**: 384x288 (balanced performance/memory)
- **Learning Rate**: 1e-4
- **Epochs**: 150
- **Base Channels**: 16
- **MAF Depth**: 2

### Advanced Settings
```python
# From config.py
BATCH_SIZE = 12
IMAGE_SIZE = (384, 288)
EPOCHS = 150
LEARNING_RATE = 1e-4
BASE_CHANNELS = 16
MAF_DEPTH = 2
DROPOUT_RATE = 0.1
```

## 🚀 Training Commands

### Basic Training
```bash
python train.py
```

### Custom Parameters
```bash
python train.py \
    --epochs 150 \
    --batch_size 12 \
    --lr 1e-4 \
    --base_channels 16 \
    --maf_depth 2 \
    --dropout_rate 0.1 \
    --scheduler cosine
```

Parameter budget 500K–700K (disarankan):
```bash
python train.py --backbone mobilenetv3_small_075 --base_channels 8
```

### Resume Training
```bash
python train.py --resume ./checkpoints/latest_checkpoint.pth
```

### Custom Dataset Path
```bash
python train.py --data_dir /path/to/your/dataset
```

## 📊 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 8 | Batch size (use 12 for RTX A4000) |
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 1e-4 | Weight decay for regularization |
| `--base_channels` | 16 | Base channels for model |
| `--maf_depth` | 2 | MAF block depth |
| `--dropout_rate` | 0.1 | Dropout rate |
| `--scheduler` | cosine | Learning rate scheduler |
| `--image_size` | [384, 288] | Input image size (width, height) |
| `--data_dir` | ./dataset | Dataset directory |
| `--save_dir` | ./checkpoints | Checkpoint save directory |

## 🎨 Model Architecture Features

### Large Kernel Convolutions
- **7x7 and 9x9 kernels** for better spatial context
- **Multi-scale attention** with large receptive fields
- **Enhanced feature fusion** for improved segmentation

### Attention Mechanisms
- **ASA (Attention Spatial)**: Multi-scale spatial attention
- **ACA (Attention Channel)**: Multi-scale channel attention
- **Bridge Attention**: Enhanced feature refinement

### Optimization Features
- **Mixed Precision Training** (FP16) for RTX A4000
- **Gradient Clipping** for stable training
- **Learning Rate Scheduling** for better convergence

## 📈 Expected Performance

### Training Progress
- **Epoch 1-50**: Rapid F1 score improvement (60% → 80%)
- **Epoch 50-100**: Steady improvement (80% → 90%)
- **Epoch 100-150**: Fine-tuning (90% → 95%+)

### Target Metrics
- **F1 Score**: 95%+ (primary target)
- **IoU**: 90%+
- **Dice Score**: 95%+

## 🔧 Training Tips

### 1. Monitor GPU Memory
```bash
nvidia-smi -l 1
```

### 2. Use Mixed Precision
The training script automatically uses FP16 for RTX A4000.

### 3. Early Stopping
Training automatically stops if F1 score doesn't improve for 20 epochs.

### 4. Checkpoint Management
- **Best Model**: `best_model.pth` (highest F1 score)
- **Latest**: `latest_checkpoint.pth` (resume training)
- **Regular**: `checkpoint_epoch_N.pth` (every 10 epochs)

## 📊 Monitoring Training

### Real-time Metrics
- **Loss**: Training and validation loss
- **F1 Score**: Primary metric for segmentation
- **IoU**: Intersection over Union
- **Dice Score**: Dice coefficient

### Visualization
- Training curves are saved every 10 epochs
- Final training curves saved at completion
- TensorBoard support (optional)

## 🚨 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --batch_size 8

# Reduce image size
python train.py --image_size 256 256
```

#### 2. Slow Training
```bash
# Increase batch size if memory allows
python train.py --batch_size 16

# Use fewer workers
# Edit train.py: num_workers=2
```

#### 3. Poor Convergence
```bash
# Adjust learning rate
python train.py --lr 5e-5

# Change scheduler
python train.py --scheduler plateau
```

### GPU Memory Optimization
- **Batch Size**: Start with 12, adjust based on memory
- **Image Size**: 384x288 is optimal for RTX A4000
- **Mixed Precision**: Automatically enabled
- **Gradient Accumulation**: Available if needed

## 📋 Training Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test model: `python test_model.py`
- [ ] Verify dataset structure
- [ ] Set optimal batch size (12 for RTX A4000)
- [ ] Start training: `python train.py --epochs 150 --batch_size 12`
- [ ] Monitor GPU memory usage
- [ ] Check training curves every 10 epochs
- [ ] Save best model automatically
- [ ] Resume training if needed

## 🎯 Achieving 95% F1 Score

### Key Factors
1. **Large Kernels**: 7x7 and 9x9 convolutions for better context
2. **Multi-scale Attention**: Captures features at different scales
3. **Enhanced Fusion**: Better integration of multi-scale features
4. **Optimized Training**: 150 epochs with cosine scheduling
5. **Combined Loss**: BCE + Dice loss for better segmentation

### Training Strategy
- **Phase 1** (Epochs 1-50): High learning rate, rapid improvement
- **Phase 2** (Epochs 50-100): Moderate learning rate, steady improvement
- **Phase 3** (Epochs 100-150): Low learning rate, fine-tuning

## 📞 Support

If you encounter issues:
1. Check GPU memory usage
2. Verify dataset structure
3. Test with smaller batch size
4. Check error logs in terminal

## 🎉 Success Metrics

Your training is successful when:
- F1 Score reaches 95%+
- IoU reaches 90%+
- Training loss stabilizes
- Validation metrics improve consistently

Good luck with your training! 🚀
