#!/usr/bin/env python3
"""
Kaggle Simple Training Script for Half-MAFU-Net
Usage: python kaggle_train_simple.py --data_dir /path/to/dataset
"""

import os
import subprocess
import sys
import argparse

def install_requirements():
    """Install required packages for Kaggle"""
    print("Installing requirements...")
    
    try:
        # Install core requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-kaggle.txt"])
        print("Core requirements installed successfully!")
    except Exception as e:
        print(f"Warning: Could not install from requirements-kaggle.txt: {e}")
        print("Trying to install packages individually...")
        
        packages = [
            "torch>=2.0.0", "torchvision>=0.15.0", "numpy>=1.21.0", 
            "Pillow>=8.3.0", "opencv-python>=4.6.0", "scikit-learn>=1.1.0",
            "matplotlib>=3.5.0", "seaborn>=0.12.0", "tensorboard>=2.10.0",
            "tqdm>=4.64.0", "albumentations>=1.3.0", "timm==1.0.19", 
            "thop>=0.1.1.post220907"
        ]
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Installed: {package}")
            except Exception as e:
                print(f"Failed to install {package}: {e}")
    
    # Try to install pydensecrf for CRF post-processing
    try:
        print("Installing pydensecrf...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "scikit-build", "ninja", "cmake", "cython"
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git"
        ])
        print("pydensecrf installed successfully!")
    except Exception as e:
        print(f"Warning: Could not install pydensecrf: {e}")
        print("CRF post-processing will not be available")

def setup_directories():
    """Setup Kaggle directories"""
    print("Setting up directories...")
    
    dirs = [
        "/kaggle/working/checkpoints",
        "/kaggle/working/runs", 
        "/kaggle/working/results"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")
    
    print("Directories setup completed!")

def validate_dataset(data_dir):
    """Validate dataset structure"""
    print(f"Validating dataset at: {data_dir}")
    
    required_dirs = [
        "train/images", "train/masks", 
        "val/images", "val/masks"
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing: {full_path}")
            return False
        else:
            # Count files
            file_count = len([f for f in os.listdir(full_path) if f.endswith('.png')])
            print(f"✅ {full_path}: {file_count} PNG files")
    
    return True

def run_training(data_dir, epochs=200, batch_size=16):
    """Run training with the given dataset path"""
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Stage A: Initial training
    print("\nStage A: Initial Training with MixUp")
    print("-" * 40)
    
    stage_a_cmd = [
        "python", "train.py",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--image_size", "512", "384",
        "--lr", "1e-4",
        "--scheduler", "cosine",
        "--backbone", "mobilenetv3_small_075",
        "--base_channels", "16",
        "--postprocess", "crf",
        "--threshold_search",
        "--freeze_backbone_epochs", "5",
        "--mixup_alpha", "0.2",
        "--early_patience", "30",
        "--num_workers", "4",
        "--use_amp",
        "--grad_accum_steps", "2",
        "--persistent_workers",
        "--data_dir", data_dir,
        "--save_dir", "/kaggle/working/checkpoints/stage_a",
        "--log_dir", "/kaggle/working/runs/stage_a"
    ]
    
    print("Command:")
    print(" ".join(stage_a_cmd))
    
    try:
        print("\nStarting Stage A training...")
        subprocess.run(stage_a_cmd, check=True)
        print("✅ Stage A completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Stage A failed: {e}")
        return False
    
    # Stage B: Fine-tuning
    print("\nStage B: Fine-tuning without MixUp")
    print("-" * 40)
    
    stage_b_cmd = [
        "python", "train.py",
        "--epochs", str(epochs // 2),  # Half the epochs for fine-tuning
        "--batch_size", str(batch_size),
        "--image_size", "512", "384",
        "--lr", "8e-5",
        "--scheduler", "cosine",
        "--backbone", "mobilenetv3_small_075",
        "--base_channels", "16",
        "--resume", "/kaggle/working/checkpoints/stage_a/best_model.pth",
        "--postprocess", "crf",
        "--threshold_search",
        "--mixup_alpha", "0.0",
        "--freeze_backbone_epochs", "0",
        "--early_patience", "30",
        "--num_workers", "4",
        "--use_amp",
        "--grad_accum_steps", "2",
        "--persistent_workers",
        "--data_dir", data_dir,
        "--save_dir", "/kaggle/working/checkpoints/stage_b",
        "--log_dir", "/kaggle/working/runs/stage_b"
    ]
    
    print("Command:")
    print(" ".join(stage_b_cmd))
    
    try:
        print("\nStarting Stage B training...")
        subprocess.run(stage_b_cmd, check=True)
        print("✅ Stage B completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Stage B failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Half-MAFU-Net Kaggle Training')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs for Stage A (default: 200)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--skip_install', action='store_true',
                       help='Skip package installation')
    
    args = parser.parse_args()
    
    print("Half-MAFU-Net Kaggle Training")
    print("="*50)
    
    # Check if we're in Kaggle
    if not os.path.exists("/kaggle"):
        print("Warning: This script is designed for Kaggle environment")
        print("Some paths may need to be adjusted for local use")
    
    # Install requirements (unless skipped)
    if not args.skip_install:
        install_requirements()
    else:
        print("Skipping package installation...")
    
    # Setup directories
    setup_directories()
    
    # Validate dataset
    if not validate_dataset(args.data_dir):
        print(f"❌ Invalid dataset structure at: {args.data_dir}")
        print("Expected structure:")
        print("  data_dir/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   └── masks/")
        print("  └── val/")
        print("      ├── images/")
        print("      └── masks/")
        return
    
    # Run training
    success = run_training(args.data_dir, args.epochs, args.batch_size)
    
    if success:
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Results saved to:")
        print("- Checkpoints: /kaggle/working/checkpoints/")
        print("- Logs: /kaggle/working/runs/")
        print("- Results: /kaggle/working/results/")
        
        print("\nTo view results:")
        print("!tensorboard --logdir=/kaggle/working/runs --host=0.0.0.0 --port=6006")
        
        print("\nBest models:")
        print("- Stage A: /kaggle/working/checkpoints/stage_a/best_model.pth")
        print("- Stage B: /kaggle/working/checkpoints/stage_b/best_model.pth")
    else:
        print("\n❌ Training failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Check if dataset path is correct")
        print("2. Verify dataset structure")
        print("3. Check GPU memory with: !nvidia-smi")
        print("4. Try reducing batch_size if OOM occurs")

if __name__ == "__main__":
    main()
