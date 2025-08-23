#!/usr/bin/env python3
"""
Kaggle Dataset Debug Script
Simple script to understand what's available and what's missing
"""

import os

def main():
    print("Half-MAFU-Net Dataset Debug")
    print("="*50)
    
    # Check current directory
    print("\n📁 CURRENT DIRECTORY:")
    try:
        items = os.listdir(".")
        if items:
            for item in items:
                if os.path.isdir(item):
                    print(f"  📁 {item}/")
                    try:
                        subitems = os.listdir(item)
                        if len(subitems) <= 10:
                            for subitem in subitems:
                                print(f"     📄 {subitem}")
                        else:
                            for subitem in subitems[:5]:
                                print(f"     📄 {subitem}")
                            print(f"     ... and {len(subitems)-5} more items")
                    except Exception as e:
                        print(f"     ❌ Error reading {item}: {e}")
                else:
                    print(f"  📄 {item}")
        else:
            print("  ❌ Empty directory")
    except Exception as e:
        print(f"  ❌ Error listing directory: {e}")
    
    # Check /kaggle/input
    print("\n📁 /kaggle/input:")
    if os.path.exists("/kaggle/input"):
        try:
            input_dirs = os.listdir("/kaggle/input")
            if input_dirs:
                for dir_name in input_dirs:
                    full_path = os.path.join("/kaggle/input", dir_name)
                    if os.path.isdir(full_path):
                        print(f"  📁 {dir_name}/")
                        try:
                            subitems = os.listdir(full_path)
                            if len(subitems) <= 10:
                                for subitem in subitems:
                                    print(f"     📄 {subitem}")
                            else:
                                for subitem in subitems[:5]:
                                    print(f"     📄 {subitem}")
                                print(f"     ... and {len(subitems)-5} more items")
                        except Exception as e:
                            print(f"     ❌ Error reading {dir_name}: {e}")
            else:
                print("  ❌ No datasets found")
        except Exception as e:
            print(f"  ❌ Error listing /kaggle/input: {e}")
    else:
        print("  ❌ /kaggle/input does not exist")
    
    # Check /kaggle/working
    print("\n📁 /kaggle/working:")
    if os.path.exists("/kaggle/working"):
        try:
            working_dirs = os.listdir("/kaggle/working")
            if working_dirs:
                for dir_name in working_dirs:
                    full_path = os.path.join("/kaggle/working", dir_name)
                    if os.path.isdir(full_path):
                        print(f"  📁 {dir_name}/")
                        try:
                            subitems = os.listdir(full_path)
                            if len(subitems) <= 10:
                                for subitem in subitems:
                                    print(f"     📄 {subitem}")
                            else:
                                for subitem in subitems[:5]:
                                    print(f"     📄 {subitem}")
                                print(f"     ... and {len(subitems)-5} more items")
                        except Exception as e:
                            print(f"     ❌ Error reading {dir_name}: {e}")
            else:
                print("  ❌ No additional folders")
        except Exception as e:
            print(f"  ❌ Error listing /kaggle/working: {e}")
    else:
        print("  ❌ /kaggle/working does not exist")
    
    # Check for dataset structure
    print("\n🔍 CHECKING FOR DATASET STRUCTURE:")
    
    # Check current directory
    print("\n1. Current directory (./):")
    train_images = "train/images"
    train_masks = "train/masks"
    val_images = "val/images"
    val_masks = "val/masks"
    
    if os.path.exists(train_images):
        print(f"   ✅ {train_images}")
        try:
            count = len([f for f in os.listdir(train_images) if f.endswith('.png')])
            print(f"      📊 {count} PNG files")
        except:
            print(f"      ❌ Error counting files")
    else:
        print(f"   ❌ {train_images}")
    
    if os.path.exists(train_masks):
        print(f"   ✅ {train_masks}")
        try:
            count = len([f for f in os.listdir(train_masks) if f.endswith('.png')])
            print(f"      📊 {count} PNG files")
        except:
            print(f"      ❌ Error counting files")
    else:
        print(f"   ❌ {train_masks}")
    
    if os.path.exists(val_images):
        print(f"   ✅ {val_images}")
        try:
            count = len([f for f in os.listdir(val_images) if f.endswith('.png')])
            print(f"      📊 {count} PNG files")
        except:
            print(f"      ❌ Error counting files")
    else:
        print(f"   ❌ {val_images}")
    
    if os.path.exists(val_masks):
        print(f"   ✅ {val_masks}")
        try:
            count = len([f for f in os.listdir(val_masks) if f.endswith('.png')])
            print(f"      📊 {count} PNG files")
        except:
            print(f"      ❌ Error counting files")
    else:
        print(f"   ❌ {val_masks}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    
    has_train_images = os.path.exists(train_images)
    has_train_masks = os.path.exists(train_masks)
    has_val_images = os.path.exists(val_images)
    has_val_masks = os.path.exists(val_masks)
    
    if has_train_images and has_train_masks and has_val_images and has_val_masks:
        print("✅ VALID DATASET STRUCTURE FOUND in current directory!")
        print("   You can run training with: --data_dir .")
    else:
        print("❌ INVALID DATASET STRUCTURE in current directory")
        print("   Missing:")
        if not has_train_images: print("   - train/images/")
        if not has_train_masks: print("   - train/masks/")
        if not has_val_images: print("   - val/images/")
        if not has_val_masks: print("   - val/masks/")
    
    print("\n💡 NEXT STEPS:")
    if has_train_images and has_train_masks and has_val_images and has_val_masks:
        print("1. Run training: !python train.py --data_dir .")
        print("2. Or use script: !python kaggle_train_fast.py")
    else:
        print("1. Upload dataset to Kaggle")
        print("2. Or place dataset in current directory")
        print("3. Or use existing dataset from /kaggle/input/")

if __name__ == "__main__":
    main()
