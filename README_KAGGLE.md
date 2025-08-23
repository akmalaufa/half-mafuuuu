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

## 🎯 Cara Training

### **Option 1: Script Paling Sederhana (Recommended)**
```bash
# Jalankan script yang paling sederhana
!python kaggle_train_fast.py
```

### **Option 2: Script dengan Commands**
```bash
# Script yang akan tunjukkan commands yang bisa dijalankan
!python kaggle_commands.py
```

### **Option 3: Script Otomatis dengan Deteksi Dataset**
```bash
# Script yang akan mendeteksi dataset otomatis
!python kaggle_training_example.py
```

### **Option 4: Script Manual dengan Path Input**
```bash
# Jalankan dengan dataset path yang spesifik
!python kaggle_train_simple.py --data_dir /kaggle/input/your-dataset-name

# Atau dengan parameter custom
!python kaggle_train_simple.py \
  --data_dir /kaggle/input/your-dataset \
  --epochs 150 \
  --batch_size 12
```

### **Option 5: Command Manual Langsung**
```bash
# Copy-paste command ini langsung di notebook
!python train.py \
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
  --save_dir /kaggle/working/checkpoints \
  --log_dir /kaggle/working/runs
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

### Dataset Path Error
```bash
# Error: FileNotFoundError: No such file or directory
# Solusi: Gunakan script yang mendeteksi dataset otomatis
!python kaggle_training_example.py

# Atau berikan path yang benar
!python kaggle_train_simple.py --data_dir /kaggle/input/your-actual-dataset-name
```

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

### **Untuk Pemula (Recommended):**
```python
# 1. Upload dataset ke Kaggle
# 2. Jalankan script paling sederhana
!python kaggle_train_fast.py
```

### **Untuk User yang Mau Lihat Commands Dulu:**
```python
# Script akan tunjukkan commands yang tersedia
!python kaggle_commands.py
```

### **Untuk Advanced User:**
```python
# 1. Install dependencies
!pip install -r requirements-kaggle.txt

# 2. Jalankan dengan path spesifik
!python kaggle_train_simple.py --data_dir /kaggle/input/your-dataset

# 3. Atau command manual
!python train.py --epochs 200 --batch_size 16 --use_amp --grad_accum_steps 2 --data_dir /kaggle/input/your-dataset
```

## 📝 Notes

1. **Dataset Path**: Script otomatis akan mendeteksi dataset
2. **GPU Runtime**: Pastikan notebook menggunakan GPU runtime
3. **Memory**: Script otomatis handle GPU memory management
4. **Resume**: Bisa resume training dari checkpoint jika terputus
5. **Error Handling**: Script sudah include error handling untuk OOM dan path issues

## 🆘 Support

Jika ada masalah:
1. **Dataset Path**: Gunakan `kaggle_training_example.py` untuk deteksi otomatis
2. **GPU Memory**: Cek dengan `!nvidia-smi`
3. **Dependencies**: Pastikan semua packages terinstall
4. **Log Error**: Cek console output untuk error details
5. **Batch Size**: Kurangi jika OOM terjadi

## 📚 Script Files

- **`kaggle_train_fast.py`** - Script paling sederhana, langsung jalankan training
- **`kaggle_commands.py`** - Script yang tunjukkan commands yang bisa dijalankan
- **`kaggle_training_example.py`** - Script otomatis dengan deteksi dataset
- **`kaggle_train_simple.py`** - Script manual dengan command line arguments
- **`train.py`** - Script training utama dengan semua optimasi
- **`requirements-kaggle.txt`** - Dependencies untuk Kaggle

Happy Training! 🚀
