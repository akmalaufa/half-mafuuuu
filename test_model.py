import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from half_mafunet import MAFUNet
import matplotlib.pyplot as plt

def test_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MAFUNet(in_ch=3, out_ch=1, base_c=16, maf_depth=2, dropout_rate=0.1)
    model = model.to(device)
    model.eval()
    
    # Test input
    batch_size = 2
    channels = 3
    height = 288
    width = 384
    
    # Create dummy input
    x = torch.randn(batch_size, channels, height, width).to(device)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output)
        print(f"Probability range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
        
        # Convert to binary mask
        mask = (probs > 0.5).float()
        print(f"Binary mask sum: {mask.sum().item()}")
    
    # Test with different input sizes
    test_sizes = [(256, 256), (512, 512), (640, 480)]
    for h, w in test_sizes:
        x_test = torch.randn(1, 3, h, w).to(device)
        with torch.no_grad():
            output_test = model(x_test)
            print(f"Input size ({h}, {w}) -> Output size {output_test.shape}")
    
    print("\nModel test completed successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Memory usage estimation
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

if __name__ == "__main__":
    test_model()
