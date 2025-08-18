# Training Configuration for Half-MAFU-Net
# Optimized for NVIDIA RTX A4000 (16GB VRAM)

class TrainingConfig:
    # Data Configuration
    DATA_DIR = "./dataset"
    IMAGE_SIZE = (384, 288)  # (width, height) - optimized for RTX A4000
    BATCH_SIZE = 12  # Optimized for 16GB VRAM
    
    # Model Configuration
    BASE_CHANNELS = 16
    MAF_DEPTH = 2
    DROPOUT_RATE = 0.1
    
    # Training Configuration
    EPOCHS = 150  # Increased for better convergence
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    SCHEDULER = "cosine"  # cosine, step, or plateau
    
    # Loss Configuration
    BCE_WEIGHT = 0.3
    DICE_WEIGHT = 0.7
    
    # Optimization
    OPTIMIZER = "AdamW"
    GRADIENT_CLIP = 1.0
    
    # Data Augmentation (optional)
    USE_AUGMENTATION = True
    AUGMENTATION_PROB = 0.5
    
    # Checkpoint Configuration
    SAVE_DIR = "./checkpoints"
    SAVE_FREQUENCY = 10  # Save every N epochs
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 20
    MIN_DELTA = 1e-4
    
    # Mixed Precision (for RTX A4000)
    USE_AMP = True  # Automatic Mixed Precision
    
    # Validation
    VAL_FREQUENCY = 1  # Validate every N epochs
    
    # Logging
    LOG_FREQUENCY = 100  # Log every N batches
    TENSORBOARD = True

# GPU Memory Optimization
class GPUConfig:
    # For RTX A4000 (16GB VRAM)
    MAX_MEMORY = 15.5  # GB, leave some buffer
    GRADIENT_ACCUMULATION_STEPS = 1
    PIN_MEMORY = True
    NUM_WORKERS = 4
    
    # Mixed Precision
    FP16 = True
    BF16 = False  # RTX A4000 supports BF16
    
    # Memory Management
    EMPTY_CACHE_FREQUENCY = 5  # Empty cache every N epochs

# Advanced Training Options
class AdvancedConfig:
    # Learning Rate Scheduling
    WARMUP_EPOCHS = 5
    WARMUP_LR = 1e-6
    
    # Loss Functions
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    
    # Testing
    TEST_ON_TRAIN = False
    SAVE_PREDICTIONS = True
    PREDICTION_THRESHOLD = 0.5

# Export configurations
if __name__ == "__main__":
    print("Training Configuration:")
    print(f"Batch Size: {TrainingConfig.BATCH_SIZE}")
    print(f"Image Size: {TrainingConfig.IMAGE_SIZE}")
    print(f"Epochs: {TrainingConfig.EPOCHS}")
    print(f"Learning Rate: {TrainingConfig.LEARNING_RATE}")
    print(f"Base Channels: {TrainingConfig.BASE_CHANNELS}")
    print(f"MAF Depth: {TrainingConfig.MAF_DEPTH}")
    
    print("\nGPU Configuration:")
    print(f"Max Memory: {GPUConfig.MAX_MEMORY} GB")
    print(f"Mixed Precision: {GPUConfig.FP16}")
    print(f"BF16 Support: {GPUConfig.BF16}")
