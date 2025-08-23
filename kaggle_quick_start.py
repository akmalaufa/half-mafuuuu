#!/usr/bin/env python3
"""
Kaggle Quick Start Script for Half-MAFU-Net
Simple script to run training with manual dataset path input
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

def list_available_datasets():
    """List available datasets in Kaggle"""
    print("\n" + "="*60)
    print("AVAILABLE DATASETS")
    print("="*60)
    
    # Check /kaggle/input
    if os.path.exists("/kaggle/input"):
        print("Datasets in /kaggle/input:")
        try:
            input_dirs = os.listdir("/kaggle/input")
            for dir_name in input_dirs:
                full_path = os.path.join("/kaggle/input", dir_name)
                if os.path.isdir(full_path):
                    # Check if it looks like our dataset
                    train_images = os.path.join(full_path, "train", "images")
                    train_masks = os.path.join(full_path, "train", "masks")
                    val_images = os.path.join(full_path, "val", "images")
                    val_masks = os.path.join(full_path, "val", "masks")
                    
                    if (os.path.exists(train_images) and os.path.exists(train_masks) and 
                        os.path.exists(val_images) and os.path.exists(val_masks)):
                        print(f"  ✅ {dir_name}: {full_path} (VALID DATASET)")
                    else:
                        print(f"  ❌ {dir_name}: {full_path} (INVALID STRUCTURE)")
        except Exception as e:
            print(f"Error listing /kaggle/input: {e}")
    else:
        print("No /kaggle/input directory found")
    
    # Check current directory
    print("\nCurrent directory structure:")
    current_dirs = ["train/images", "train/masks", "val/images", "val/masks"]
    for dir_path in current_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
    
    print("="*60)

def get_dataset_path():
    """Get dataset path from user"""
    print("\n" + "="*60)
    print("DATASET PATH INPUT")
    print("="*60)
    
    list_available_datasets()
    
    print("\nPlease enter the dataset path:")
    print("Examples:")
    print("  - /kaggle/input/your-dataset-name")
    print("  - . (current directory)")
    print("  - /kaggle/working/your-dataset")
    
    while True:
        dataset_path = input("\nEnter dataset path: ").strip()
        
        if not dataset_path:
            print("Path cannot be empty. Please try again.")
            continue
        
        # Check if path exists
        if not os.path.exists(dataset_path):
            print(f"Path '{dataset_path}' does not exist. Please try again.")
            continue
        
        # Check if it's a valid dataset
        train_images = os.path.join(dataset_path, "train", "images")
        train_masks = os.path.join(dataset_path, "train", "masks")
        val_images = os.path.join(dataset_path, "val", "images")
        val_masks = os.path.join(dataset_path, "val", "masks")
        
        if not all(os.path.exists(p) for p in [train_images, train_masks, val_images, val_masks]):
            print(f"Path '{dataset_path}' does not contain valid dataset structure.")
            print("Expected: train/images, train/masks, val/images, val/masks")
            continue
        
        print(f"✅ Valid dataset found at: {dataset_path}")
        return dataset_path

def run_training(dataset_path):
    """Run training with the given dataset path"""
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Stage A: Initial training
    print("\nStage A: Initial Training with MixUp")
    print("-" * 40)
    
    stage_a_cmd = [
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
        "--epochs", "100",
        "--batch_size", "16",
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
        "--data_dir", dataset_path,
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
    print("Half-MAFU-Net Kaggle Quick Start")
    print("="*50)
    
    # Check if we're in Kaggle
    if not os.path.exists("/kaggle"):
        print("Warning: This script is designed for Kaggle environment")
        print("Some paths may need to be adjusted for local use")
    
    # Install requirements
    install_requirements()
    
    # Setup directories
    setup_directories()
    
    # Get dataset path
    dataset_path = get_dataset_path()
    
    # Run training
    success = run_training(dataset_path)
    
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
    else:
        print("\n❌ Training failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Check if dataset path is correct")
        print("2. Verify dataset structure (train/images, train/masks, val/images, val/masks)")
        print("3. Check GPU memory with: !nvidia-smi")
        print("4. Try reducing batch_size if OOM occurs")

if __name__ == "__main__":
    main()
