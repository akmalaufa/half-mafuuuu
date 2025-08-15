# # train_half_mafunet.py
# import argparse, os, time, random
# from pathlib import Path
# from glob import glob

# import numpy as np
# from PIL import Image, ImageOps

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import torchvision.transforms.functional as TF

# # ====== import model (file kamu) ======
# from half_mafunet import MAFUNet as MAFUNet


# # -------------------- Utils --------------------
# def seed_all(seed=42):
#     random.seed(seed); np.random.seed(seed)
#     torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True


# def dice_coeff(probs, target, eps=1e-7):
#     # probs, target: (B,1,H,W) in [0,1]
#     p = probs.view(probs.size(0), -1)
#     t = target.view(target.size(0), -1)
#     inter = (p * t).sum(1)
#     union = p.sum(1) + t.sum(1)
#     dice = (2 * inter + eps) / (union + eps)
#     return dice.mean()


# def iou_coeff(probs, target, eps=1e-7):
#     p = (probs > 0.5).float()
#     t = (target > 0.5).float()
#     p = p.view(p.size(0), -1)
#     t = t.view(t.size(0), -1)
#     inter = (p * t).sum(1)
#     union = (p + t - p * t).sum(1)
#     iou = (inter + eps) / (union + eps)
#     return iou.mean()


# class BCEDiceLoss(nn.Module):
#     def __init__(self, bce_weight=0.5):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.bce_w = bce_weight
#     def forward(self, logits, targets):
#         bce = self.bce(logits, targets)
#         probs = torch.sigmoid(logits)
#         dice = 1.0 - dice_coeff(probs, targets)
#         return self.bce_w * bce + (1 - self.bce_w) * dice


# # -------------------- Dataset --------------------
# class SegFolder(Dataset):
#     """
#     Pair images and masks by filename stem (e.g., image_001.jpg <-> image_001.png)
#     """
#     IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

#     def __init__(self, root, size=(288, 384), augment=False):
#         self.root = Path(root)
#         self.img_dir = self.root / 'images'
#         self.mask_dir = self.root / 'masks'
#         self.size = size
#         self.augment = augment

#         img_paths = []
#         for ext in self.IMG_EXTS:
#             img_paths += glob(str(self.img_dir / f'*{ext}'))
#         img_paths = sorted(img_paths)

#         # build pairs
#         self.pairs = []
#         for ip in img_paths:
#             stem = Path(ip).stem
#             mp = None
#             for ext in self.IMG_EXTS:
#                 candidate = self.mask_dir / f'{stem}{ext}'
#                 if candidate.exists():
#                     mp = str(candidate); break
#             if mp is None:
#                 # fallback: any file starting with the stem
#                 for m in self.mask_dir.glob(f'{stem}*'):
#                     if m.suffix.lower() in self.IMG_EXTS:
#                         mp = str(m); break
#             if mp is not None:
#                 self.pairs.append((ip, mp))

#         if len(self.pairs) == 0:
#             raise RuntimeError(f'No image/mask pairs found under {self.root}')

#         self.to_tensor = transforms.ToTensor()

#     def __len__(self): return len(self.pairs)

#     def _load_image(self, path):
#         img = Image.open(path).convert('RGB')  # force 3ch
#         return img

#     def _load_mask(self, path):
#         return Image.open(path).convert('L')   # grayscale

#     def _augment(self, img, mask):
#         if random.random() < 0.5:
#             img = ImageOps.mirror(img); mask = ImageOps.mirror(mask)
#         if random.random() < 0.5:
#             img = ImageOps.flip(img); mask = ImageOps.flip(mask)
#         if random.random() < 0.3:
#             angle = random.uniform(-15, 15)
#             img = img.rotate(angle, resample=Image.BILINEAR)
#             mask = mask.rotate(angle, resample=Image.NEAREST)
#         return img, mask

#     def __getitem__(self, idx):
#         ip, mp = self.pairs[idx]
#         img = self._load_image(ip)
#         mask = self._load_mask(mp)

#         if self.augment:
#             img, mask = self._augment(img, mask)

#         if self.size is not None:
#             img = img.resize(self.size, Image.BILINEAR)
#             mask = mask.resize(self.size, Image.NEAREST)

#         img_t = self.to_tensor(img)  # [0,1]
#         mask_np = np.array(mask, dtype=np.uint8)
#         mask_np = (mask_np > 127).astype('float32')  # 0/1 dari B/W
#         mask_t = torch.from_numpy(mask_np)[None, ...]  # (1,H,W)

#         return img_t, mask_t, Path(ip).name


# # -------------------- Train/Eval --------------------
# def train_one_epoch(model, loader, opt, scaler, loss_fn, device, amp=True):
#     model.train()
#     run_loss = run_dice = run_iou = 0.0

#     for imgs, masks, _ in loader:
#         imgs = imgs.to(device)
#         masks = masks.to(device)

#         opt.zero_grad(set_to_none=True)
#         with torch.amp.autocast('cuda', enabled=amp):
#             logits = model(imgs)
#             loss = loss_fn(logits, masks)

#         # --- urutan AMP yang benar ---
#         scaler.scale(loss).backward()
#         scaler.step(opt)
#         scaler.update()

#         with torch.no_grad():
#             probs = torch.sigmoid(logits)
#             run_loss += loss.item() * imgs.size(0)
#             run_dice += dice_coeff(probs, masks).item() * imgs.size(0)
#             run_iou  += iou_coeff(probs, masks).item() * imgs.size(0)

#     n = len(loader.dataset)
#     return run_loss / n, run_dice / n, run_iou / n


# @torch.no_grad()
# def evaluate(model, loader, loss_fn, device, epoch, out_dir, save_samples=6, amp=True):
#     model.eval()
#     run_loss = run_dice = run_iou = 0.0

#     sample_dir = Path(out_dir) / 'samples'
#     sample_dir.mkdir(parents=True, exist_ok=True)
#     saved = 0

#     for imgs, masks, names in loader:
#         imgs = imgs.to(device)
#         masks = masks.to(device)
#         with torch.amp.autocast('cuda', enabled=amp):
#             logits = model(imgs)
#             loss = loss_fn(logits, masks)
#             probs = torch.sigmoid(logits)

#         run_loss += loss.item() * imgs.size(0)
#         run_dice += dice_coeff(probs, masks).item() * imgs.size(0)
#         run_iou  += iou_coeff(probs, masks).item() * imgs.size(0)

#         # save a few B/W masks
#         if saved < save_samples:
#             preds = (probs > 0.5).float().cpu().numpy()
#             for b in range(min(imgs.size(0), save_samples - saved)):
#                 pm = (preds[b, 0] * 255).astype(np.uint8)
#                 Image.fromarray(pm).save(sample_dir / f'val_epoch{epoch:02d}_{names[b]}')
#                 saved += 1
#                 if saved >= save_samples: break

#     n = len(loader.dataset)
#     return run_loss / n, run_dice / n, run_iou / n


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='dataset', help='folder with train/ val/ test/')
#     parser.add_argument('--img_size', type=int, nargs=2, default=[288, 384])
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--batch_size', type=int, default=8)
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--base_c', type=int, default=16)
#     parser.add_argument('--maf_depth', type=int, default=2)
#     parser.add_argument('--num_workers', type=int, default=2)
#     parser.add_argument('--amp', action='store_true', help='use mixed precision (CUDA)')
#     parser.add_argument('--out_dir', type=str, default='outputs')
#     args = parser.parse_args()

#     seed_all(42)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     use_amp = bool(args.amp and device.type == 'cuda')

#     # Datasets & loaders
#     train_set = SegFolder(Path(args.data_root) / 'train', size=tuple(args.img_size), augment=True)
#     val_set   = SegFolder(Path(args.data_root) / 'val',   size=tuple(args.img_size), augment=False)

#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
#                               num_workers=args.num_workers, pin_memory=True, drop_last=False)
#     val_loader   = DataLoader(val_set, batch_size=max(1, args.batch_size//2), shuffle=False,
#                               num_workers=args.num_workers, pin_memory=True, drop_last=False)

#     # Model
#     model = MAFUNet(in_ch=3, out_ch=1, base_c=args.base_c, maf_depth=args.maf_depth).to(device)

#     # Optim, loss
#     opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     loss_fn = BCEDiceLoss(bce_weight=0.5)

#     # AMP scaler (API baru)
#     scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

#     # Output dirs
#     ckpt_dir = Path(args.out_dir) / 'checkpoints'
#     ckpt_dir.mkdir(parents=True, exist_ok=True)

#     best_dice = 0.0
#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         tr_loss, tr_dice, tr_iou = train_one_epoch(model, train_loader, opt, scaler, loss_fn, device, amp=use_amp)
#         va_loss, va_dice, va_iou = evaluate(model, val_loader, loss_fn, device, epoch, args.out_dir, amp=use_amp)

#         dt = time.time() - t0
#         print(f"[{epoch:03d}/{args.epochs}] "
#               f"train_loss={tr_loss:.4f} dice={tr_dice:.4f} iou={tr_iou:.4f} | "
#               f"val_loss={va_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f} | "
#               f"lr={opt.param_groups[0]['lr']:.2e} time={dt:.1f}s")

#         # save best
#         if va_dice > best_dice:
#             best_dice = va_dice
#             torch.save({
#                 'epoch': epoch,
#                 'model_state': model.state_dict(),
#                 'opt_state': opt.state_dict(),
#                 'dice': best_dice,
#                 'args': vars(args)
#             }, ckpt_dir / 'best.pt')

#         # save last
#         torch.save({'epoch': epoch, 'model_state': model.state_dict()}, ckpt_dir / 'last.pt')

#     print(f"Training done. Best val Dice = {best_dice:.4f}. "
#           f"Checkpoints @ {ckpt_dir} and samples @ {Path(args.out_dir) / 'samples'}")


# if __name__ == '__main__':
#     main()



import argparse, os, time, random
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# ====== import model (file kamu) ======
from half_mafunet import MAFUNet as MAFUNet


# -------------------- Utils --------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def dice_coeff(probs, target, eps=1e-7):
    """
    Soft Dice untuk monitoring (pakai probabilitas 0..1)
    probs, target: (B,1,H,W)
    """
    p = probs.view(probs.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(1)
    union = p.sum(1) + t.sum(1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def iou_coeff(probs, target, eps=1e-7):
    """
    IoU pakai threshold 0.5 untuk monitoring
    probs, target: (B,1,H,W)
    """
    p = (probs > 0.5).float()
    t = (target > 0.5).float()
    p = p.view(p.size(0), -1)
    t = t.view(t.size(0), -1)
    inter = (p * t).sum(1)
    union = (p + t - p * t).sum(1)
    iou = (inter + eps) / (union + eps)
    return iou.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice = 1.0 - dice_coeff(probs, targets)
        return self.bce_w * bce + (1 - self.bce_w) * dice


# -------------------- Dataset --------------------
class SegFolder(Dataset):
    """
    Pair images and masks by filename stem (e.g., image_001.jpg <-> image_001.png)
    Struktur dataset:
        root/
          images/*.png|jpg|...   (gambar)
          masks/*.png|jpg|...    (mask biner)
    """
    IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

    def __init__(self, root, size=(288, 384), augment=False):
        self.root = Path(root)
        self.img_dir = self.root / 'images'
        self.mask_dir = self.root / 'masks'
        self.size = size
        self.augment = augment

        img_paths = []
        for ext in self.IMG_EXTS:
            img_paths += glob(str(self.img_dir / f'*{ext}'))
        img_paths = sorted(img_paths)

        # build pairs
        self.pairs = []
        for ip in img_paths:
            stem = Path(ip).stem
            mp = None
            for ext in self.IMG_EXTS:
                candidate = self.mask_dir / f'{stem}{ext}'
                if candidate.exists():
                    mp = str(candidate)
                    break
            if mp is None:
                # fallback: any file starting with the stem
                for m in self.mask_dir.glob(f'{stem}*'):
                    if m.suffix.lower() in self.IMG_EXTS:
                        mp = str(m)
                        break
            if mp is not None:
                self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            raise RuntimeError(f'No image/mask pairs found under {self.root}')

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')  # force 3 channel
        return img

    def _load_mask(self, path):
        return Image.open(path).convert('L')   # grayscale

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random.random() < 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        return img, mask

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = self._load_image(ip)
        mask = self._load_mask(mp)

        if self.augment:
            img, mask = self._augment(img, mask)

        if self.size is not None:
            img = img.resize(self.size, Image.BILINEAR)
            mask = mask.resize(self.size, Image.NEAREST)

        img_t = self.to_tensor(img)  # [0,1]
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 127).astype('float32')  # 0/1 dari B/W
        mask_t = torch.from_numpy(mask_np)[None, ...]  # (1,H,W)

        return img_t, mask_t, Path(ip).name


# -------------------- Train/Eval --------------------
def train_one_epoch(model, loader, opt, scaler, loss_fn, device, amp=True):
    model.train()
    run_loss = run_dice = run_iou = 0.0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=amp):
            logits = model(imgs)
            loss = loss_fn(logits, masks)

        # urutan AMP yang benar
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            run_loss += loss.item() * imgs.size(0)
            run_dice += dice_coeff(probs, masks).item() * imgs.size(0)
            run_iou  += iou_coeff(probs, masks).item() * imgs.size(0)

    n = len(loader.dataset)
    return run_loss / n, run_dice / n, run_iou / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, epoch, out_dir, save_samples=6, amp=True):
    model.eval()
    run_loss = run_dice = run_iou = 0.0

    sample_dir = Path(out_dir) / 'samples'
    sample_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for imgs, masks, names in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        with torch.amp.autocast('cuda', enabled=amp):
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            probs = torch.sigmoid(logits)

        run_loss += loss.item() * imgs.size(0)
        run_dice += dice_coeff(probs, masks).item() * imgs.size(0)
        run_iou  += iou_coeff(probs, masks).item() * imgs.size(0)

        # save a few predicted masks for visual check
        if saved < save_samples:
            preds = (probs > 0.5).float().cpu().numpy()
            for b in range(min(imgs.size(0), save_samples - saved)):
                pm = (preds[b, 0] * 255).astype(np.uint8)
                Image.fromarray(pm).save(sample_dir / f'val_epoch{epoch:02d}_{names[b]}')
                saved += 1
                if saved >= save_samples:
                    break

    n = len(loader.dataset)
    return run_loss / n, run_dice / n, run_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset', help='folder with train/ val/ test/')
    parser.add_argument('--img_size', type=int, nargs=2, default=[288, 384])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--base_c', type=int, default=16)
    parser.add_argument('--maf_depth', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true', help='use mixed precision (CUDA)')
    parser.add_argument('--out_dir', type=str, default='outputs')
    args = parser.parse_args()

    seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.amp and device.type == 'cuda')

    # Datasets & loaders
    train_set = SegFolder(Path(args.data_root) / 'train', size=tuple(args.img_size), augment=True)
    val_set   = SegFolder(Path(args.data_root) / 'val',   size=tuple(args.img_size), augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = MAFUNet(in_ch=3, out_ch=1, base_c=args.base_c, maf_depth=args.maf_depth).to(device)

    # Optimizer & Loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = BCEDiceLoss(bce_weight=0.5)

    # LR scheduler: turunkan LR x0.1 bila val loss stagnan 10 epoch
    # Kompatibel untuk PyTorch lama (tanpa argumen 'verbose' / 'threshold_mode')
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=0.1,      # bagi 10
            patience=10,     # setelah 10 epoch tidak membaik
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0.0,
            verbose=True,
        )
    except TypeError:
        # Fallback untuk versi PyTorch lama
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=0.1,
            patience=10,
            threshold=1e-4,
            cooldown=0,
            min_lr=0.0,
        )

    # AMP scaler (API baru)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Output dirs
    ckpt_dir = Path(args.out_dir) / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_dice, tr_iou = train_one_epoch(
            model, train_loader, opt, scaler, loss_fn, device, amp=use_amp
        )
        va_loss, va_dice, va_iou = evaluate(
            model, val_loader, loss_fn, device, epoch, args.out_dir, amp=use_amp
        )

        # Step scheduler pakai validation loss (urutan benar: setelah 1 epoch selesai)
        scheduler.step(va_loss)

        dt = time.time() - t0
        cur_lr = opt.param_groups[0]['lr']
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} dice={tr_dice:.4f} iou={tr_iou:.4f} | "
            f"val_loss={va_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f} | "
            f"lr={cur_lr:.2e} time={dt:.1f}s"
        )

        # save best (berdasarkan Dice)
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(
                {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'opt_state': opt.state_dict(),
                    'sched_state': scheduler.state_dict(),
                    'dice': best_dice,
                    'args': vars(args),
                },
                ckpt_dir / 'best.pt',
            )

        # save last
        torch.save(
            {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'sched_state': scheduler.state_dict(),
            },
            ckpt_dir / 'last.pt',
        )

    print(
        f"Training done. Best val Dice = {best_dice:.4f}. "
        f"Checkpoints @ {ckpt_dir} and samples @ {Path(args.out_dir) / 'samples'}"
    )


if __name__ == '__main__':
    main()
