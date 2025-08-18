import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from sklearn.metrics import f1_score, jaccard_score, confusion_matrix

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except Exception:
    dcrf = None

from tqdm import tqdm
from half_mafunet import MAFUNet
try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Custom Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, aug=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.aug = aug

        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

        assert len(self.images) == len(self.masks), "Number of images and masks must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 0).astype(np.uint8)

        if self.aug is not None:
            augmented = self.aug(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if mask.ndim == 2:
            mask = mask[None, ...]

        return image.float(), mask.float()

# Data Transforms
def get_train_aug(image_size=(384, 288), strong=True, clahe=True):
    h, w = image_size[1], image_size[0]
    transforms_list = []
    if strong:
        if clahe:
            transforms_list.append(A.CLAHE(p=0.3))
        transforms_list += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT101),
            A.ElasticTransform(alpha=50, sigma=7, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.GaussNoise(p=0.2),
        ]
    transforms_list += [
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms_list)


def get_val_aug(image_size=(384, 288)):
    h, w = image_size[1], image_size[0]
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

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


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

class CombinedLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.25, iou_w=0.25):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.iou_w = iou_w
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        iou_loss = self.iou(pred, target)
        return self.bce_w * bce_loss + self.dice_w * dice_loss + self.iou_w * iou_loss

# Metrics
def calculate_metrics(pred, target, threshold=0.5):
    # Accept logits or probabilities/binary. If in [0,1], skip sigmoid.
    if torch.is_floating_point(pred) and pred.min().item() >= 0.0 and pred.max().item() <= 1.0:
        pred_binary = (pred > threshold).float()
    else:
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


def apply_morph_closing(mask, ksize=3, iters=1, threshold=0.5):
    mask_np = (mask.squeeze(0).cpu().numpy() > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=iters)
    closed = torch.from_numpy((closed > 127).astype(np.float32))[None, ...]
    return closed


def apply_densecrf(image_tensor, prob_map):
    if dcrf is None:
        return (prob_map > 0.5).float()
    im = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    H, W = im.shape[:2]
    probs = np.vstack([1 - prob_map.cpu().numpy().reshape(1, H, W), prob_map.cpu().numpy().reshape(1, H, W)])
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=im, compat=5)
    Q = d.inference(5)
    pred = np.argmax(np.array(Q), axis=0).reshape(H, W)
    return torch.from_numpy(pred.astype(np.float32))[None, ...]

# Training Function
def train_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.0):
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
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_indices = torch.randperm(images.size(0)).to(device)
            images_mix = lam * images + (1 - lam) * images[batch_indices]
            masks_mix = lam * masks + (1 - lam) * masks[batch_indices]
            outputs = model(images_mix)
            loss = criterion(outputs, masks_mix)
        else:
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
def validate_epoch(model, dataloader, criterion, device, postprocess=None, vis_dir=None, epoch=0, writer: SummaryWriter = None, num_visualize=4, threshold_search: bool = False, morph_ksize: int = 3):
    model.eval()
    total_loss = 0
    total_f1 = 0
    total_iou = 0
    total_dice = 0
    all_targets = []
    all_preds = []
    probs_all = []
    masks_all = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            probs = torch.sigmoid(outputs)
            probs_all.append(probs.cpu())
            masks_all.append(masks.cpu())
            if postprocess == 'morph':
                bin_preds = []
                for b in range(probs.size(0)):
                    bin_pred = apply_morph_closing(probs[b], ksize=morph_ksize, threshold=0.5)
                    bin_preds.append(bin_pred)
                preds_bin = torch.stack(bin_preds, dim=0).to(masks.device)
            elif postprocess == 'crf':
                preds_bin = []
                for b in range(probs.size(0)):
                    pred_b = apply_densecrf(images[b].cpu(), probs[b].cpu())
                    preds_bin.append(pred_b)
                preds_bin = torch.stack(preds_bin, dim=0).to(masks.device)
            else:
                preds_bin = (probs > 0.5).float()

            f1, iou, dice = calculate_metrics(preds_bin, masks)
            
            total_loss += loss.item()
            total_f1 += f1
            total_iou += iou
            total_dice += dice
            all_targets.append(masks.cpu().numpy().astype(np.uint8))
            all_preds.append(preds_bin.cpu().numpy().astype(np.uint8))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'F1': f'{f1:.4f}',
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}'
            })
    
    # Visualization samples
    if writer is not None and vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
        imgs = images[:num_visualize].cpu()
        gts = masks[:num_visualize].cpu()
        prb = torch.sigmoid(outputs[:num_visualize].cpu())
        prd = (prb > 0.5).float()
        grid = make_grid(torch.cat([imgs, gts.repeat(1,3,1,1), prb.repeat(1,3,1,1), prd.repeat(1,3,1,1)], dim=0), nrow=num_visualize)
        writer.add_image('val/samples', grid, epoch)

    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'f1': total_f1 / num_batches,
        'iou': total_iou / num_batches,
        'dice': total_dice / num_batches
    }

    y_true = np.concatenate([t.reshape(-1) for t in all_targets], axis=0)
    y_pred = np.concatenate([p.reshape(-1) for p in all_preds], axis=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BG', 'Polyp'], yticklabels=['BG', 'Polyp'])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    if writer is not None:
        writer.add_figure('val/confusion_matrix', fig, epoch)
    plt.close(fig)

    # Derive metrics from confusion matrix (binary)
    tn, fp, fn, tp = cm.ravel().tolist()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)  # sensitivity
    specificity = tn / (tn + fp + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    metrics.update({'precision': precision, 'sen': recall, 'spe': specificity, 'acc': accuracy})

    # Threshold search (maximize IoU on raw probabilities)
    if threshold_search and len(probs_all) > 0:
        probs_cat = torch.cat(probs_all, dim=0)
        masks_cat = torch.cat(masks_all, dim=0)
        best_thr, best_iou = 0.5, -1.0
        for thr in [x/100.0 for x in range(30, 71, 2)]:  # 0.30..0.70 step 0.02
            preds_thr = (probs_cat > thr).float()
            inter = (preds_thr * masks_cat).sum().item()
            union = (preds_thr + masks_cat - preds_thr * masks_cat).sum().item()
            iou_thr = inter / (union + 1e-6)
            if iou_thr > best_iou:
                best_iou = iou_thr
                best_thr = thr
        metrics['best_threshold'] = best_thr

    return metrics

def compute_model_flops_params(model: nn.Module, image_size):
    # image_size is [W, H]
    H, W = int(image_size[1]), int(image_size[0])
    if thop_profile is None:
        return None, sum(p.numel() for p in model.parameters())
    device_tmp = next(model.parameters()).device
    with torch.no_grad():
        dummy = torch.randn(1, 3, H, W, device=device_tmp)
        flops, params = thop_profile(model, inputs=(dummy,), verbose=False)
    return flops, params

def print_paper_results(params, flops, val_metrics):
    # params in number, flops in number of operations
    params_m = params / 1e6 if params is not None else None
    flops_g = (flops / 1e9) if (flops is not None) else None
    lines = []
    lines.append("================ Results (Paper) ===============")
    if params_m is not None:
        lines.append(f"Params(M)  : {params_m:.3f}")
    if flops_g is not None:
        lines.append(f"FLOPs(G)   : {flops_g:.3f} (GMAcs)")
    if 'iou' in val_metrics:
        lines.append(f"mIoU       : {val_metrics['iou']*100:.2f}")
    if 'f1' in val_metrics:
        lines.append(f"F1/Dice (FG): {val_metrics['f1']*100:.2f}")
    if 'precision' in val_metrics:
        lines.append(f"Precision  : {val_metrics['precision']*100:.2f}")
    if 'sen' in val_metrics:
        lines.append(f"SEN        : {val_metrics['sen']*100:.2f}")
    if 'acc' in val_metrics:
        lines.append(f"ACC        : {val_metrics['acc']*100:.2f}")
    if 'spe' in val_metrics:
        lines.append(f"SPE        : {val_metrics['spe']*100:.2f}")
    print("\n".join(lines))

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

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0

    def step(self, value):
        improved = value > (self.best + self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return improved, self.counter >= self.patience

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
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--base_channels', type=int, default=16, help='Base channels for model')
    parser.add_argument('--maf_depth', type=int, default=2, help='MAF block depth')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--mixup_alpha', type=float, default=0.0, help='MixUp alpha (0 disables)')
    parser.add_argument('--postprocess', type=str, default='none', choices=['none', 'morph', 'crf'], help='Validation post-processing')
    parser.add_argument('--morph_ksize', type=int, default=3, help='Kernel size for morphological closing (odd number)')
    parser.add_argument('--threshold_search', action='store_true', help='Search best probability threshold on validation')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0, help='Freeze backbone for N initial epochs')
    parser.add_argument('--early_patience', type=int, default=20, help='Early stopping patience on mIoU')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--backbone', type=str, default='mobilenetv3_small_050', help='timm backbone variant')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau'], help='Learning rate scheduler')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='./runs', help='TensorBoard log dir')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data transforms (Albumentations)
    train_aug = get_train_aug(args.image_size, strong=True, clahe=True)
    val_aug = get_val_aug(args.image_size)

    # Datasets
    train_dataset = SegmentationDataset(
        os.path.join(args.data_dir, 'train', 'images'),
        os.path.join(args.data_dir, 'train', 'masks'),
        aug=train_aug
    )

    val_dataset = SegmentationDataset(
        os.path.join(args.data_dir, 'val', 'images'),
        os.path.join(args.data_dir, 'val', 'masks'),
        aug=val_aug
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
        dropout_rate=args.dropout_rate,
        backbone_name=args.backbone
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if not (500_000 <= total_params <= 700_000):
        print(f"[Warning] Parameter count {total_params:,} is outside target [500K, 700K].")
        print("          Try: --backbone mobilenetv3_small_050 and lower --base_channels (e.g., 8) to reduce params.")
    
    # Loss function
    criterion = CombinedLoss(bce_w=0.5, dice_w=0.25, iou_w=0.25)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Training history
    train_metrics = {'loss': [], 'f1': [], 'iou': [], 'dice': []}
    val_metrics = {'loss': [], 'f1': [], 'iou': [], 'dice': []}
    
    writer = SummaryWriter(args.log_dir)
    start_epoch = 0
    best_f1 = 0
    best_iou = 0
    early = EarlyStopping(patience=args.early_patience, min_delta=args.min_delta)
    
    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, checkpoint_metrics = load_checkpoint(model, optimizer, scheduler, args.resume)
        best_f1 = checkpoint_metrics.get('f1', 0)
        best_iou = checkpoint_metrics.get('iou', 0)
        print(f"Resumed from epoch {start_epoch} with best F1: {best_f1:.4f}, best IoU: {best_iou:.4f}")
    
    print(f"Starting training from epoch {start_epoch + 1}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Optionally freeze backbone for warmup epochs
        if args.freeze_backbone_epochs > 0 and epoch < args.freeze_backbone_epochs:
            if hasattr(model, 'freeze_backbone'):
                model.freeze_backbone()
        else:
            if hasattr(model, 'unfreeze_backbone'):
                model.unfreeze_backbone()

        # Train
        train_results = train_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=args.mixup_alpha)
        
        # Validate
        vis_dir = os.path.join(args.save_dir, 'viz')
        val_results = validate_epoch(
            model, val_loader, criterion, device,
            postprocess=args.postprocess, vis_dir=vis_dir, epoch=epoch, writer=writer,
            threshold_search=args.threshold_search, morph_ksize=args.morph_ksize
        )
        
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

        # TensorBoard logging
        writer.add_scalar('train/loss', train_results['loss'], epoch)
        writer.add_scalar('train/f1', train_results['f1'], epoch)
        writer.add_scalar('train/iou', train_results['iou'], epoch)
        writer.add_scalar('train/dice', train_results['dice'], epoch)
        writer.add_scalar('val/loss', val_results['loss'], epoch)
        writer.add_scalar('val/f1', val_results['f1'], epoch)
        writer.add_scalar('val/iou', val_results['iou'], epoch)
        writer.add_scalar('val/dice', val_results['dice'], epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            best_checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_results, best_checkpoint_path)
            print(f"New best F1: {best_f1:.4f} - Model saved!")
        
        # Early stopping on mIoU
        improved, should_stop = early.step(val_results['iou'])
        if improved:
            best_iou = val_results['iou']
        if should_stop:
            print(f"Early stopping triggered on mIoU. Best IoU: {best_iou:.4f}")
            break

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
    
    print(f"\nTraining completed! Best F1: {best_f1:.4f}, Best IoU: {best_iou:.4f}")
    
    # Final plot
    plot_path = os.path.join(args.save_dir, 'final_training_curves.png')
    plot_training_curves(train_metrics, val_metrics, plot_path)
    
    # Save final model
    final_checkpoint_path = os.path.join(args.save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, scheduler, epoch, val_results, final_checkpoint_path)
    writer.close()

    # Print paper-style summary on final validation
    flops, params = compute_model_flops_params(model, args.image_size)
    # Use last val metrics to approximate summary
    try:
        last_val = {k: v[-1] for k, v in val_metrics.items()}
    except Exception:
        last_val = val_results
    print_paper_results(params=params, flops=flops, val_metrics=last_val)

if __name__ == '__main__':
    main()
