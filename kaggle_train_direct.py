#!/usr/bin/env python3
"""
Kaggle Direct Training Script for Half-MAFU-Net
Simple script that can be run directly without user input
"""

import os
import subprocess
import sys

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

def find_dataset():
    """Find dataset automatically"""
    print("Searching for dataset...")
    
    # Check current directory first
    if os.path.exists("train/images") and os.path.exists("train/masks") and \
       os.path.exists("val/images") and os.path.exists("val/masks"):
        print("✅ Found dataset in current directory")
        return "."
    
    # Check /kaggle/input
    if os.path.exists("/kaggle/input"):
        try:
            input_dirs = os.listdir("/kaggle/input")
            for dir_name in input_dirs:
                full_path = os.path.join("/kaggle/input", dir_name)
                if os.path.isdir(full_path):
                    # Check if this looks like our dataset
                    train_images = os.path.join(full_path, "train", "images")
                    train_masks = os.path.join(full_path, "train", "masks")
                    val_images = os.path.join(full_path, "val", "images")
                    val_masks = os.path.join(full_path, "val", "masks")
                    
                    if (os.path.exists(train_images) and os.path.exists(train_masks) and 
                        os.path.exists(val_images) and os.path.exists(val_masks)):
                        print(f"✅ Found dataset at: {full_path}")
                        return full_path
        except Exception as e:
            print(f"Error checking /kaggle/input: {e}")
    
    # Check /kaggle/working
    if os.path.exists("/kaggle/working"):
        try:
            working_dirs = os.listdir("/kaggle/working")
            for dir_name in working_dirs:
                full_path = os.path.join("/kaggle/working", dir_name)
                if os.path.isdir(full_path):
                    # Check if this looks like our dataset
                    train_images = os.path.join(full_path, "train", "images")
                    train_masks = os.path.join(full_path, "train", "masks")
                    val_images = os.path.join(full_path, "val", "images")
                    val_masks = os.path.join(full_path, "val", "masks")
                    
                    if (os.path.exists(train_images) and os.path.exists(train_masks) and 
                        os.path.exists(val_images) and os.path.exists(val_masks)):
                        print(f"✅ Found dataset at: {full_path}")
                        return full_path
        except Exception as e:
            print(f"Error checking /kaggle/working: {e}")
    
    print("❌ No dataset found automatically")
    return None

def run_single_training(dataset_path):
    """Run single training session (no two-stage)"""
    print("\n" + "="*60)
    print("STARTING SINGLE TRAINING SESSION")
    print("="*60)
    
    # Single training command with all optimizations
    training_cmd = [
        "python", "train.py",
        "--epochs", "200",
        "--batch_size", "16",
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
        "--data_dir", dataset_path,
        "--save_dir", "/kaggle/working/checkpoints",
        "--log_dir", "/kaggle/working/runs"
    ]
    
    print("Training command:")
    print(" ".join(training_cmd))
    
    try:
        print("\nStarting training...")
        subprocess.run(training_cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main function"""
    print("Half-MAFU-Net Kaggle Direct Training")
    print("="*50)
    
    # Check if we're in Kaggle
    if not os.path.exists("/kaggle"):
        print("Warning: This script is designed for Kaggle environment")
        print("Some paths may need to be adjusted for local use")
    
    # Install requirements
    install_requirements()
    
    # Setup directories
    setup_directories()
    
    # Find dataset
    dataset_path = find_dataset()
    
    if dataset_path is None:
        print("\n" + "="*60)
        print("DATASET NOT FOUND!")
        print("="*60)
        print("Please do one of the following:")
        print("1. Upload dataset to Kaggle and run this script again")
        print("2. Place dataset in current directory with structure:")
        print("   ./train/images/")
        print("   ./train/masks/")
        print("   ./val/images/")
        print("   ./val/masks/")
        print("3. Use Kaggle Dataset API")
        print("\nTo check available datasets:")
        print("!ls /kaggle/input/")
        return
    
    print(f"Using dataset at: {dataset_path}")
    
    # Run training
    success = run_single_training(dataset_path)
    
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
        
        print("\nBest model:")
        print("- /kaggle/working/checkpoints/best_model.pth")
    else:
        print("\n❌ Training failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Check if dataset path is correct")
        print("2. Verify dataset structure")
        print("3. Check GPU memory with: !nvidia-smi")
        print("4. Try reducing batch_size if OOM occurs")

if __name__ == "__main__":
    main()
