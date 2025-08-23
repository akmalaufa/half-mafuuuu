#!/usr/bin/env python3
"""
Kaggle Robust Training Script for Half-MAFU-Net
Handles various dataset scenarios and provides clear guidance
"""

import os
import subprocess
import sys

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
        print(f"✅ Created: {dir_path}")
    
    print("Directories setup completed!")

def list_all_possible_locations():
    """List all possible dataset locations"""
    print("\n" + "="*60)
    print("SEARCHING FOR DATASET IN ALL LOCATIONS")
    print("="*60)
    
    locations = []
    
    # 1. Current directory
    print("\n1. Checking current directory (./):")
    if os.path.exists("train/images") and os.path.exists("train/masks") and \
       os.path.exists("val/images") and os.path.exists("val/masks"):
        locations.append(("Current Directory", "."))
        print("   ✅ Found valid dataset structure")
    else:
        print("   ❌ No valid dataset structure")
        if os.path.exists("train"):
            print("   📁 Found 'train' folder but missing required subfolders")
        if os.path.exists("val"):
            print("   📁 Found 'val' folder but missing required subfolders")
    
    # 2. /kaggle/input
    print("\n2. Checking /kaggle/input:")
    if os.path.exists("/kaggle/input"):
        try:
            input_dirs = os.listdir("/kaggle/input")
            if input_dirs:
                for dir_name in input_dirs:
                    full_path = os.path.join("/kaggle/input", dir_name)
                    if os.path.isdir(full_path):
                        print(f"   📁 Found: {dir_name}")
                        
                        # Check structure
                        train_images = os.path.join(full_path, "train", "images")
                        train_masks = os.path.join(full_path, "train", "masks")
                        val_images = os.path.join(full_path, "val", "images")
                        val_masks = os.path.join(full_path, "val", "masks")
                        
                        if (os.path.exists(train_images) and os.path.exists(train_masks) and 
                            os.path.exists(val_images) and os.path.exists(val_masks)):
                            locations.append((f"/kaggle/input/{dir_name}", full_path))
                            print(f"      ✅ Valid dataset structure")
                            
                            # Count files
                            try:
                                train_count = len([f for f in os.listdir(train_images) if f.endswith('.png')])
                                val_count = len([f for f in os.listdir(val_images) if f.endswith('.png')])
                                print(f"      📊 Train: {train_count} images, Val: {val_count} images")
                            except:
                                pass
                        else:
                            print(f"      ❌ Invalid structure")
                            if os.path.exists(os.path.join(full_path, "train")):
                                print(f"         📁 Has 'train' folder")
                            if os.path.exists(os.path.join(full_path, "val")):
                                print(f"         📁 Has 'val' folder")
            else:
                print("   ❌ No datasets found in /kaggle/input")
        except Exception as e:
            print(f"   ❌ Error checking /kaggle/input: {e}")
    else:
        print("   ❌ /kaggle/input directory does not exist")
    
    # 3. /kaggle/working
    print("\n3. Checking /kaggle/working:")
    if os.path.exists("/kaggle/working"):
        try:
            working_dirs = os.listdir("/kaggle/working")
            if working_dirs:
                for dir_name in working_dirs:
                    if dir_name not in ["checkpoints", "runs", "results"]:  # Skip our created dirs
                        full_path = os.path.join("/kaggle/working", dir_name)
                        if os.path.isdir(full_path):
                            print(f"   📁 Found: {dir_name}")
                            
                            # Check structure
                            train_images = os.path.join(full_path, "train", "images")
                            train_masks = os.path.join(full_path, "train", "masks")
                            val_images = os.path.join(full_path, "val", "images")
                            val_masks = os.path.join(full_path, "val", "masks")
                            
                            if (os.path.exists(train_images) and os.path.exists(train_masks) and 
                                os.path.exists(val_images) and os.path.exists(val_masks)):
                                locations.append((f"/kaggle/working/{dir_name}", full_path))
                                print(f"      ✅ Valid dataset structure")
                            else:
                                print(f"      ❌ Invalid structure")
            else:
                print("   ❌ No additional folders in /kaggle/working")
        except Exception as e:
            print(f"   ❌ Error checking /kaggle/working: {e}")
    else:
        print("   ❌ /kaggle/working directory does not exist")
    
    return locations

def provide_solutions(locations):
    """Provide solutions based on what was found"""
    print("\n" + "="*60)
    print("SOLUTIONS & RECOMMENDATIONS")
    print("="*60)
    
    if locations:
        print(f"\n🎉 Found {len(locations)} valid dataset(s)!")
        print("\nAvailable datasets:")
        for i, (name, path) in enumerate(locations, 1):
            print(f"  {i}. {name}")
            print(f"     Path: {path}")
        
        print("\nTo use these datasets, run one of these commands:")
        for i, (name, path) in enumerate(locations, 1):
            print(f"\n{i}. For dataset '{name}':")
            print(f"   !python train.py --epochs 200 --batch_size 16 --data_dir {path}")
        
        return locations[0][1]  # Return first valid dataset path
    else:
        print("\n❌ No valid datasets found!")
        print("\n🔧 SOLUTIONS:")
        print("\n1. UPLOAD DATASET TO KAGGLE:")
        print("   - Go to Kaggle > Datasets > Create Dataset")
        print("   - Upload your dataset with this structure:")
        print("     your-dataset/")
        print("     ├── train/")
        print("     │   ├── images/  (PNG files)")
        print("     │   └── masks/   (PNG files)")
        print("     └── val/")
        print("         ├── images/  (PNG files)")
        print("         └── masks/   (PNG files)")
        
        print("\n2. PLACE DATASET IN CURRENT DIRECTORY:")
        print("   - Upload dataset files directly to notebook")
        print("   - Ensure folder structure matches above")
        
        print("\n3. USE KAGGLE DATASET API:")
        print("   - Find existing polyp datasets on Kaggle")
        print("   - Use: !kaggle datasets download -d username/dataset-name")
        
        print("\n4. CHECK CURRENT STRUCTURE:")
        print("   Current directory contents:")
        try:
            for item in os.listdir("."):
                if os.path.isdir(item):
                    print(f"     📁 {item}/")
                    try:
                        subitems = os.listdir(item)
                        for subitem in subitems[:5]:  # Show first 5 items
                            print(f"        📄 {subitem}")
                        if len(subitems) > 5:
                            print(f"        ... and {len(subitems)-5} more items")
                    except:
                        pass
                else:
                    print(f"     📄 {item}")
        except Exception as e:
            print(f"     Error listing directory: {e}")
        
        return None

def run_training_with_path(dataset_path):
    """Run training with the given dataset path"""
    print(f"\n🚀 Starting training with dataset: {dataset_path}")
    
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
    
    print("\nTraining command:")
    print(" ".join(cmd))
    
    try:
        print("\nStarting training...")
        subprocess.run(cmd, check=True)
        print("\n🎉 Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False

def main():
    """Main function"""
    print("Half-MAFU-Net Kaggle Robust Training")
    print("="*50)
    
    # Check if we're in Kaggle
    if not os.path.exists("/kaggle"):
        print("⚠️  Warning: This script is designed for Kaggle environment")
        print("   Some paths may need to be adjusted for local use")
    
    # Setup directories
    setup_directories()
    
    # Search for datasets
    valid_locations = list_all_possible_locations()
    
    # Provide solutions
    dataset_path = provide_solutions(valid_locations)
    
    if dataset_path:
        print(f"\n✅ Using dataset: {dataset_path}")
        
        # Ask user if they want to proceed
        print("\nProceed with training? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response == 'y':
                success = run_training_with_path(dataset_path)
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
            else:
                print("\nTraining cancelled by user.")
        except:
            print("\n⚠️  Could not get user input. To run training manually, use:")
            print(f"!python train.py --epochs 200 --batch_size 16 --data_dir {dataset_path}")
    else:
        print("\n❌ Cannot proceed without valid dataset.")
        print("Please follow the solutions above to set up your dataset.")

if __name__ == "__main__":
    main()
