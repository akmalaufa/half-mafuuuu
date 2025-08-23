#!/usr/bin/env python3
"""
Kaggle Commands Script for Half-MAFU-Net
Just run the training commands directly
"""

import os
import subprocess

def main():
    print("Half-MAFU-Net Kaggle Commands")
    print("="*50)
    
    # Create directories
    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    os.makedirs("/kaggle/working/runs", exist_ok=True)
    print("Directories created!")
    
    # List available datasets
    print("\nAvailable datasets:")
    if os.path.exists("/kaggle/input"):
        try:
            for item in os.listdir("/kaggle/input"):
                print(f"  - /kaggle/input/{item}")
        except:
            pass
    
    # Check current directory
    if os.path.exists("train/images"):
        print("  - ./ (current directory)")
    
    print("\n" + "="*60)
    print("TRAINING COMMANDS")
    print("="*60)
    
    # Command 1: If dataset is in current directory
    if os.path.exists("train/images") and os.path.exists("train/masks"):
        print("\n1. Dataset in current directory:")
        cmd1 = [
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
            "--data_dir", ".",
            "--save_dir", "/kaggle/working/checkpoints",
            "--log_dir", "/kaggle/working/runs"
        ]
        print(" ".join(cmd1))
        
        # Ask if user wants to run this
        print("\nRun this command? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response == 'y':
                print("\n🚀 Starting training...")
                subprocess.run(cmd1, check=True)
                print("\n🎉 Training completed!")
                return
        except:
            pass
    
    # Command 2: If dataset is in /kaggle/input
    if os.path.exists("/kaggle/input"):
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
                        print(f"\n2. Dataset in /kaggle/input/{dir_name}:")
                        cmd2 = [
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
                            "--data_dir", full_path,
                            "--save_dir", "/kaggle/working/checkpoints",
                            "--log_dir", "/kaggle/working/runs"
                        ]
                        print(" ".join(cmd2))
                        
                        # Ask if user wants to run this
                        print(f"\nRun training with dataset {dir_name}? (y/n): ", end="")
                        try:
                            response = input().strip().lower()
                            if response == 'y':
                                print("\n🚀 Starting training...")
                                subprocess.run(cmd2, check=True)
                                print("\n🎉 Training completed!")
                                return
                        except:
                            pass
                        break
        except Exception as e:
            print(f"Error checking /kaggle/input: {e}")
    
    print("\n" + "="*60)
    print("MANUAL COMMANDS")
    print("="*60)
    print("If no dataset found automatically, use one of these:")
    print("\nFor dataset in current directory:")
    print("python train.py --epochs 200 --batch_size 16 --image_size 512 384 --lr 1e-4 --scheduler cosine --backbone mobilenetv3_small_075 --base_channels 16 --postprocess crf --threshold_search --freeze_backbone_epochs 5 --mixup_alpha 0.2 --early_patience 30 --num_workers 4 --use_amp --grad_accum_steps 2 --persistent_workers --data_dir . --save_dir /kaggle/working/checkpoints --log_dir /kaggle/working/runs")
    
    print("\nFor dataset in /kaggle/input/your-dataset:")
    print("python train.py --epochs 200 --batch_size 16 --image_size 512 384 --lr 1e-4 --scheduler cosine --backbone mobilenetv3_small_075 --base_channels 16 --postprocess crf --threshold_search --freeze_backbone_epochs 5 --mixup_alpha 0.2 --early_patience 30 --num_workers 4 --use_amp --grad_accum_steps 2 --persistent_workers --data_dir /kaggle/input/your-dataset --save_dir /kaggle/working/checkpoints --log_dir /kaggle/working/runs")

if __name__ == "__main__":
    main()
