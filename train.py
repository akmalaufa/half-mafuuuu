import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score
import time
from tqdm import tqdm
import argparse
from half_mafunet import MAFUNet

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Custom Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        assert len(self.images) == len(self.masks), "Number of images and masks must match"
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask

# Data Transforms
def get_transforms(image_size=(384, 288)):
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform

# Loss Functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# Metrics
def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    # Convert to numpy for sklearn metrics
    pred_np = pred_binary.cpu().numpy().flatten()
    target_np = target_binary.cpu().numpy().flatten()
    
    # F1 Score
    f1 = f1_score(target_np, pred_np, average='binary', zero_division=0)
    
    # IoU (Jaccard Score)
    iou = jaccard_score(target_np, pred_np, average='binary', zero_division=0)
    
    # Dice Score
    intersection = (pred_np * target_np).sum()
    dice = (2. * intersection) / (pred_np.sum() + target_np.sum() + 1e-6)
    
    return f1, iou, dice

# Training Function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_f1 = 0
    total_iou = 0
    total_dice = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        f1, iou, dice = calculate_metrics(outputs, masks)
        
        total_loss += loss.item()
        total_f1 += f1
        total_iou += iou
        total_dice += dice
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'F1': f'{f1:.4f}',
            'IoU': f'{iou:.4f}',
            'Dice': f'{dice:.4f}'
        })
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'f1': total_f1 / num_batches,
        'iou': total_iou / num_batches,
        'dice': total_dice / num_batches
    }

# Validation Function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_f1 = 0
    total_iou = 0
    total_dice = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            f1, iou, dice = calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_f1 += f1
            total_iou += iou
            total_dice += dice
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'F1': f'{f1:.4f}',
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}'
            })
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'f1': total_f1 / num_batches,
        'iou': total_iou / num_batches,
        'dice': total_dice / num_batches
    }

# Learning Rate Scheduler
def get_scheduler(optimizer, scheduler_type='cosine', num_epochs=100):
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    else:
        return None

# Save Checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_f1': metrics.get('f1', 0)
    }
    torch.save(checkpoint, save_path)

# Load Checkpoint
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

# Plot Training Curves
def plot_training_curves(train_metrics, val_metrics, save_path):
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, train_metrics['loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_metrics['loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score
    axes[0, 1].plot(epochs, train_metrics['f1'], 'b-', label='Train F1')
    axes[0, 1].plot(epochs, val_metrics['f1'], 'r-', label='Val F1')
    axes[0, 1].set_title('Training and Validation F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU
    axes[1, 0].plot(epochs, train_metrics['iou'], 'b-', label='Train IoU')
    axes[1, 0].plot(epochs, val_metrics['iou'], 'r-', label='Val IoU')
    axes[1, 0].set_title('Training and Validation IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Dice Score
    axes[1, 1].plot(epochs, train_metrics['dice'], 'b-', label='Train Dice')
    axes[1, 1].plot(epochs, val_metrics['dice'], 'r-', label='Val Dice')
    axes[1, 1].set_title('Training and Validation Dice Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Main Training Function
def main():
    parser = argparse.ArgumentParser(description='Train Half-MAFU-Net')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Dataset directory')
    parser.add_argument('--image_size', type=int, nargs=2, default=[384, 288], help='Image size (width, height)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--base_channels', type=int, default=16, help='Base channels for model')
    parser.add_argument('--maf_depth', type=int, default=2, help='MAF block depth')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau'], help='Learning rate scheduler')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data transforms
    image_transform, mask_transform = get_transforms(args.image_size)
    
    # Datasets
    train_dataset = SegmentationDataset(
        os.path.join(args.data_dir, 'train', 'images'),
        os.path.join(args.data_dir, 'train', 'masks'),
        transform=image_transform,
        mask_transform=mask_transform
    )
    
    val_dataset = SegmentationDataset(
        os.path.join(args.data_dir, 'val', 'images'),
        os.path.join(args.data_dir, 'val', 'masks'),
        transform=image_transform,
        mask_transform=mask_transform
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model = MAFUNet(
        in_ch=3,
        out_ch=1,
        base_c=args.base_channels,
        maf_depth=args.maf_depth,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = CombinedLoss(bce_weight=0.3, dice_weight=0.7)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Training history
    train_metrics = {'loss': [], 'f1': [], 'iou': [], 'dice': []}
    val_metrics = {'loss': [], 'f1': [], 'iou': [], 'dice': []}
    
    start_epoch = 0
    best_f1 = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, checkpoint_metrics = load_checkpoint(model, optimizer, scheduler, args.resume)
        best_f1 = checkpoint_metrics.get('f1', 0)
        print(f"Resumed from epoch {start_epoch} with best F1: {best_f1:.4f}")
    
    print(f"Starting training from epoch {start_epoch + 1}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_results = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_results = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_results['loss'])
            else:
                scheduler.step()
        
        # Store metrics
        for key in train_metrics:
            train_metrics[key].append(train_results[key])
            val_metrics[key].append(val_results[key])
        
        # Print epoch summary
        print(f"Train - Loss: {train_results['loss']:.4f}, F1: {train_results['f1']:.4f}, IoU: {train_results['iou']:.4f}, Dice: {train_results['dice']:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f}, F1: {val_results['f1']:.4f}, IoU: {val_results['iou']:.4f}, Dice: {val_results['dice']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            best_checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_results, best_checkpoint_path)
            print(f"New best F1: {best_f1:.4f} - Model saved!")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_results, checkpoint_path)
        
        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(args.save_dir, 'latest_checkpoint.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, val_results, latest_checkpoint_path)
        
        # Plot training curves
        if (epoch + 1) % 10 == 0:
            plot_path = os.path.join(args.save_dir, 'training_curves.png')
            plot_training_curves(train_metrics, val_metrics, plot_path)
    
    print(f"\nTraining completed! Best F1: {best_f1:.4f}")
    
    # Final plot
    plot_path = os.path.join(args.save_dir, 'final_training_curves.png')
    plot_training_curves(train_metrics, val_metrics, plot_path)
    
    # Save final model
    final_checkpoint_path = os.path.join(args.save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, val_results, final_checkpoint_path)

if __name__ == '__main__':
    main()
