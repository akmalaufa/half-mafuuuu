#!/usr/bin/env python3
"""
Training Launcher for Half-MAFU-Net
Quick start script with preset configurations
"""

import os
import subprocess
import sys
from config import TrainingConfig, GPUConfig

def print_banner():
    print("=" * 60)
    print("🚀 Half-MAFU-Net Training Launcher")
    print("=" * 60)
    print(f"🎯 Target: 95%+ F1 Score")
    print(f"🖥️  Optimized for: NVIDIA RTX A4000")
    print(f"📊 Base Channels: {TrainingConfig.BASE_CHANNELS}")
    print(f"🔧 MAF Depth: {TrainingConfig.MAF_DEPTH}")
    print("=" * 60)

def check_gpu():
    """Check GPU availability and memory"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Detected: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 12:  # RTX A4000 has 16GB
                print("✅ Sufficient GPU memory for optimal training")
                return True
            else:
                print("⚠️  GPU memory might be limited, consider reducing batch size")
                return True
        else:
            print("❌ No GPU detected, training will be slow on CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dataset():
    """Check if dataset exists and has correct structure"""
    data_dir = TrainingConfig.DATA_DIR
    required_dirs = [
        os.path.join(data_dir, "train", "images"),
        os.path.join(data_dir, "train", "masks"),
        os.path.join(data_dir, "val", "images"),
        os.path.join(data_dir, "val", "masks")
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
    
    # Count files
    train_images = len([f for f in os.listdir(required_dirs[0]) if f.endswith('.png')])
    train_masks = len([f for f in os.listdir(required_dirs[1]) if f.endswith('.png')])
    val_images = len([f for f in os.listdir(required_dirs[2]) if f.endswith('.png')])
    val_masks = len([f for f in os.listdir(required_dirs[3]) if f.endswith('.png')])
    
    print(f"📁 Dataset found:")
    print(f"   Train: {train_images} images, {train_masks} masks")
    print(f"   Val: {val_images} images, {val_masks} masks")
    
    if train_images == 0 or val_images == 0:
        print("❌ No images found in dataset")
        return False
    
    return True

def get_training_preset():
    """Get training preset from user"""
    print("\n🎯 Training Presets:")
    print("1. 🚀 Fast Training (50 epochs, batch_size=16)")
    print("2. ⚡ Standard Training (100 epochs, batch_size=12)")
    print("3. 🎯 High Performance (150 epochs, batch_size=12)")
    print("4. 🔧 Custom Configuration")
    
    while True:
        try:
            choice = input("\nSelect preset (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("❌ Invalid choice, please select 1-4")
        except KeyboardInterrupt:
            print("\n\n👋 Training cancelled")
            sys.exit(0)

def get_custom_config():
    """Get custom training configuration"""
    print("\n🔧 Custom Configuration:")
    
    try:
        epochs = int(input("Epochs (default 150): ") or "150")
        batch_size = int(input("Batch size (default 12): ") or "12")
        lr = float(input("Learning rate (default 1e-4): ") or "1e-4")
        base_channels = int(input("Base channels (default 16): ") or "16")
        maf_depth = int(input("MAF depth (default 2): ") or "2")
        
        return {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'base_channels': base_channels,
            'maf_depth': maf_depth
        }
    except ValueError:
        print("❌ Invalid input, using defaults")
        return {
            'epochs': 150,
            'batch_size': 12,
            'lr': 1e-4,
            'base_channels': 16,
            'maf_depth': 2
        }

def build_training_command(preset_choice):
    """Build training command based on preset"""
    base_cmd = ["python", "train.py"]
    
    if preset_choice == '1':  # Fast Training
        config = {
            'epochs': 50,
            'batch_size': 16,
            'lr': 1e-4,
            'base_channels': 16,
            'maf_depth': 2
        }
    elif preset_choice == '2':  # Standard Training
        config = {
            'epochs': 100,
            'batch_size': 12,
            'lr': 1e-4,
            'base_channels': 16,
            'maf_depth': 2
        }
    elif preset_choice == '3':  # High Performance
        config = {
            'epochs': 150,
            'batch_size': 12,
            'lr': 1e-4,
            'base_channels': 16,
            'maf_depth': 2
        }
    else:  # Custom
        config = get_custom_config()
    
    # Build command
    cmd = base_cmd + [
        '--epochs', str(config['epochs']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--base_channels', str(config['base_channels']),
        '--maf_depth', str(config['maf_depth']),
        '--scheduler', 'cosine'
    ]
    
    return cmd, config

def start_training(cmd, config):
    """Start the training process"""
    print(f"\n🚀 Starting Training with Configuration:")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Learning Rate: {config['lr']}")
    print(f"   Base Channels: {config['base_channels']}")
    print(f"   MAF Depth: {config['maf_depth']}")
    print(f"   Scheduler: cosine")
    
    print(f"\n📝 Command: {' '.join(cmd)}")
    
    # Confirm before starting
    confirm = input("\n✅ Start training? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("👋 Training cancelled")
        return
    
    print("\n🔥 Launching training...")
    print("💡 Monitor GPU memory with: nvidia-smi -l 1")
    print("💡 Training logs will appear below:")
    print("-" * 60)
    
    try:
        # Start training
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check GPU
    if not check_gpu():
        print("❌ GPU check failed")
        return
    
    # Check dataset
    if not check_dataset():
        print("❌ Dataset check failed")
        print(f"📁 Expected structure: {TrainingConfig.DATA_DIR}")
        return
    
    # Get training preset
    preset = get_training_preset()
    
    # Build command
    cmd, config = build_training_command(preset)
    
    # Start training
    start_training(cmd, config)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Training launcher cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Check your configuration and try again")
