#!/usr/bin/env python3
"""
Kaggle Training Example for Half-MAFU-Net
Optimized for Kaggle GPU environment with automatic mixed precision and gradient accumulation
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages for Kaggle"""
    print("Installing requirements...")
    
    # Install core requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-kaggle.txt"])
    
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

def setup_kaggle_environment():
    """Setup Kaggle-specific environment"""
    print("Setting up Kaggle environment...")
    
    # Create directories
    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    os.makedirs("/kaggle/working/runs", exist_ok=True)
    os.makedirs("/kaggle/working/results", exist_ok=True)
    
    print("Kaggle directories created successfully!")

def run_training():
    """Run the training with optimal Kaggle settings"""
    print("Starting training with Kaggle optimizations...")
    
    # Stage A: Initial training with MixUp
    print("\n" + "="*60)
    print("STAGE A: Initial Training with MixUp")
    print("="*60)
    
    stage_a_cmd = [
        "python", "train.py",
        "--epochs", "200",
        "--batch_size", "16",  # Larger batch size for GPU
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
        "--num_workers", "4",  # More workers for Kaggle
        "--use_amp",  # Enable Automatic Mixed Precision
        "--grad_accum_steps", "2",  # Gradient accumulation
        "--persistent_workers",  # Keep workers alive
        "--data_dir", "/kaggle/input/your-dataset",  # Update this path
        "--save_dir", "/kaggle/working/checkpoints/stage_a",
        "--log_dir", "/kaggle/working/runs/stage_a"
    ]
    
    print("Stage A command:")
    print(" ".join(stage_a_cmd))
    
    try:
        subprocess.run(stage_a_cmd, check=True)
        print("Stage A completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Stage A failed: {e}")
        return False
    
    # Stage B: Fine-tuning without MixUp
    print("\n" + "="*60)
    print("STAGE B: Fine-tuning without MixUp")
    print("="*60)
    
    stage_b_cmd = [
        "python", "train.py",
        "--epochs", "100",
        "--batch_size", "16",
        "--image_size", "512", "384",
        "--lr", "8e-5",  # Lower learning rate
        "--scheduler", "cosine",
        "--backbone", "mobilenetv3_small_075",
        "--base_channels", "16",
        "--resume", "/kaggle/working/checkpoints/stage_a/best_model.pth",
        "--postprocess", "crf",
        "--threshold_search",
        "--mixup_alpha", "0.0",  # No MixUp
        "--freeze_backbone_epochs", "0",  # No freezing
        "--early_patience", "30",
        "--num_workers", "4",
        "--use_amp",
        "--grad_accum_steps", "2",
        "--persistent_workers",
        "--data_dir", "/kaggle/input/your-dataset",  # Update this path
        "--save_dir", "/kaggle/working/checkpoints/stage_b",
        "--log_dir", "/kaggle/working/runs/stage_b"
    ]
    
    print("Stage B command:")
    print(" ".join(stage_b_cmd))
    
    try:
        subprocess.run(stage_b_cmd, check=True)
        print("Stage B completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Stage B failed: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("Half-MAFU-Net Kaggle Training Script")
    print("="*50)
    
    # Check if we're in Kaggle
    if not os.path.exists("/kaggle"):
        print("Warning: This script is designed for Kaggle environment")
        print("Some paths may need to be adjusted for local use")
    
    # Install requirements
    install_requirements()
    
    # Setup environment
    setup_kaggle_environment()
    
    # Run training
    success = run_training()
    
    if success:
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Results saved to:")
        print("- Checkpoints: /kaggle/working/checkpoints/")
        print("- Logs: /kaggle/working/runs/")
        print("- Results: /kaggle/working/results/")
    else:
        print("\nTraining failed. Check the error messages above.")

if __name__ == "__main__":
    main()
