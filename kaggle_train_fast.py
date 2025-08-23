#!/usr/bin/env python3
"""
Kaggle Fast Training Script for Half-MAFU-Net
Ultra-simple script for quick training
"""

import os
import subprocess
import sys

def main():
    """Main function - just run training directly"""
    print("Half-MAFU-Net Kaggle Fast Training")
    print("="*50)
    
    # Setup directories
    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    os.makedirs("/kaggle/working/runs", exist_ok=True)
    print("Directories created!")
    
    # Try to find dataset automatically
    dataset_path = None
    
    # Check current directory first
    if os.path.exists("train/images") and os.path.exists("train/masks") and \
       os.path.exists("val/images") and os.path.exists("val/masks"):
        dataset_path = "."
        print("✅ Found dataset in current directory")
    
    # Check /kaggle/input if not found
    elif os.path.exists("/kaggle/input"):
        try:
            input_dirs = os.listdir("/kaggle/input")
            for dir_name in input_dirs:
                full_path = os.path.join("/kaggle/input", dir_name)
                if os.path.isdir(full_path):
                    train_images = os.path.join(full_path, "train", "images")
                    train_masks = os.path.join(full_path, "train", "masks")
                    val_images = os.path.join(full_path, "val", "images")
                    val_masks = os.path.join(full_path, "val", "masks")
                    
                    if (os.path.exists(train_images) and os.path.exists(train_masks) and 
                        os.path.exists(val_images) and os.path.exists(val_masks)):
                        dataset_path = full_path
                        print(f"✅ Found dataset at: {full_path}")
                        break
        except Exception as e:
            print(f"Error checking /kaggle/input: {e}")
    
    if dataset_path is None:
        print("\n❌ No dataset found!")
        print("Please upload dataset to Kaggle or place in current directory")
        print("Expected structure: train/images, train/masks, val/images, val/masks")
        print("\nAvailable in /kaggle/input:")
        if os.path.exists("/kaggle/input"):
            try:
                for item in os.listdir("/kaggle/input"):
                    print(f"  - {item}")
            except:
                pass
        return
    
    print(f"Using dataset: {dataset_path}")
    
    # Run training directly
    print("\n🚀 Starting training...")
    
    cmd = [
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
    
    print("Command:")
    print(" ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("\n🎉 Training completed successfully!")
        print("Results saved to /kaggle/working/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    main()
