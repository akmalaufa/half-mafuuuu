# Half-MAFU-Net Training di Kaggle Notebook

Panduan lengkap untuk training model Half-MAFU-Net di Kaggle dengan optimasi GPU.

## 🚀 Fitur Optimasi Kaggle

- **Automatic Mixed Precision (AMP)** - Training 2x lebih cepat
- **Gradient Accumulation** - Batch size efektif lebih besar
- **GPU Memory Management** - Otomatis handle OOM
- **Persistent Workers** - DataLoader lebih efisien
- **CUDA Optimizations** - cudnn.benchmark untuk performa maksimal

## 📋 Prerequisites

1. **Kaggle Account** dengan akses GPU
2. **Dataset** sudah diupload ke Kaggle
3. **Notebook** dengan GPU runtime

## 🛠️ Setup Awal

### 1. Upload Dataset
```bash
# Upload dataset ke Kaggle
# Pastikan struktur folder:
/kaggle/input/your-dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 2. Install Dependencies
```python
# Di cell pertama notebook
!pip install -r requirements-kaggle.txt

# Install pydensecrf (opsional)
!pip install scikit-build ninja cmake cython
!pip install "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git"
```

## 🎯 Training Commands

### Stage A: Initial Training (dengan MixUp)
```bash
python train.py \
  --epochs 200 \
  --batch_size 16 \
  --image_size 512 384 \
  --lr 1e-4 \
  --scheduler cosine \
  --backbone mobilenetv3_small_075 \
  --base_channels 16 \
  --postprocess crf \
  --threshold_search \
  --freeze_backbone_epochs 5 \
  --mixup_alpha 0.2 \
  --early_patience 30 \
  --num_workers 4 \
  --use_amp \
  --grad_accum_steps 2 \
  --persistent_workers \
  --data_dir /kaggle/input/your-dataset \
  --save_dir /kaggle/working/checkpoints/stage_a \
  --log_dir /kaggle/working/runs/stage_a
```

### Stage B: Fine-tuning (tanpa MixUp)
```bash
python train.py \
  --epochs 100 \
  --batch_size 16 \
  --image_size 512 384 \
  --lr 8e-5 \
  --scheduler cosine \
  --backbone mobilenetv3_small_075 \
  --base_channels 16 \
  --resume /kaggle/working/checkpoints/stage_a/best_model.pth \
  --postprocess crf \
  --threshold_search \
  --mixup_alpha 0.0 \
  --freeze_backbone_epochs 0 \
  --early_patience 30 \
  --num_workers 4 \
  --use_amp \
  --grad_accum_steps 2 \
  --persistent_workers \
  --data_dir /kaggle/input/your-dataset \
  --save_dir /kaggle/working/checkpoints/stage_b \
  --log_dir /kaggle/working/runs/stage_b
```

## 🔧 Parameter Optimasi

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| `--batch_size` | 16 | Batch size optimal untuk GPU Kaggle |
| `--num_workers` | 4 | Worker optimal untuk Kaggle |
| `--use_amp` | True | Enable Automatic Mixed Precision |
| `--grad_accum_steps` | 2 | Gradient accumulation untuk batch size efektif 32 |
| `--persistent_workers` | True | Keep workers alive antar epoch |
| `--image_size` | 512 384 | Resolusi optimal untuk performa |

## 📊 Monitoring Training

### GPU Usage
```bash
# Monitor GPU usage
!nvidia-smi

# Real-time monitoring
!watch -n 1 nvidia-smi
```

### TensorBoard
```python
# Start TensorBoard
!tensorboard --logdir=/kaggle/working/runs --host=0.0.0.0 --port=6006

# View di browser atau notebook
from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', '/kaggle/working/runs'])
url = tb.launch()
print(f"TensorBoard: {url}")
```

## 🚨 Troubleshooting

### GPU Out of Memory (OOM)
```bash
# Solusi otomatis sudah ada di script
# Jika manual, kurangi batch_size atau tambah grad_accum_steps

--batch_size 8 --grad_accum_steps 4  # Effective batch size = 32
```

### Slow Training
```python
# Pastikan AMP enabled
--use_amp

# Cek GPU utilization
!nvidia-smi

# Pastikan cudnn benchmark enabled
torch.backends.cudnn.benchmark = True
```

### Data Loading Issues
```bash
# Kurangi num_workers jika ada masalah
--num_workers 2

# Atau disable persistent workers
# (hapus --persistent_workers)
```

## 📁 Output Structure

Setelah training selesai, file akan tersimpan di:

```
/kaggle/working/
├── checkpoints/
│   ├── stage_a/
│   │   ├── best_model.pth
│   │   ├── latest_checkpoint.pth
│   │   └── final_model.pth
│   └── stage_b/
│       ├── best_model.pth
│       └── final_model.pth
├── runs/
│   ├── stage_a/
│   └── stage_b/
└── results/
    ├── training_curves.png
    └── final_training_curves.png
```

## 🎯 Target Metrics

Dengan optimasi ini, target yang diharapkan:

- **mIoU**: > 86%
- **F1-Score**: > 94%
- **Parameters**: 500K - 700K
- **Training Time**: 2-3x lebih cepat dengan AMP

## 🔄 Quick Start

Untuk training cepat dengan semua optimasi:

```python
# Jalankan script otomatis
!python kaggle_training_example.py

# Atau copy-paste command manual
!python train.py --epochs 200 --batch_size 16 --use_amp --grad_accum_steps 2 --data_dir /kaggle/input/your-dataset
```

## 📝 Notes

1. **Dataset Path**: Ganti `/kaggle/input/your-dataset` dengan path dataset yang sebenarnya
2. **GPU Runtime**: Pastikan notebook menggunakan GPU runtime
3. **Memory**: Script otomatis handle GPU memory management
4. **Resume**: Bisa resume training dari checkpoint jika terputus

## 🆘 Support

Jika ada masalah:
1. Cek GPU memory dengan `nvidia-smi`
2. Pastikan semua dependencies terinstall
3. Cek log error di console
4. Kurangi batch_size jika OOM
5. Gunakan `--help` untuk melihat semua options

Happy Training! 🚀
