# eval_metrics.py — disesuaikan dengan paper (mIoU = TP/(TP+FP+FN))
import argparse, copy
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from half_mafunet import MAFUNet as MAFUNet
from train_half_mafunet import SegFolder  # reuse dataset class

# ---------- helpers ----------
def safe_div(a, b, eps=1e-7):
    return float(a) / float(b + eps)

@torch.no_grad()
def confusion_counts(pred_bin, target_bin):
    """
    pred_bin, target_bin: (B,1,H,W) in {0,1}
    Returns summed TP, TN, FP, FN over the batch (micro-averaged).
    """
    p = pred_bin.long()
    t = target_bin.long()
    tp = (p & t).sum().item()
    tn = ((1 - p) & (1 - t)).sum().item()
    fp = (p & (1 - t)).sum().item()
    fn = ((1 - p) & t).sum().item()
    return tp, tn, fp, fn

def params_in_million(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def flops_in_gmacs(model, in_ch, H, W, device):
    """Hitung FLOPs pada salinan model agar hook THOP tidak mengotori model asli."""
    try:
        from thop import profile
    except ImportError:
        raise SystemExit("Package 'thop' belum terpasang. Install dengan: pip install thop")
    m = copy.deepcopy(model).to(device).eval()
    dummy = torch.randn(1, in_ch, H, W, device=device)
    macs, _ = profile(m, inputs=(dummy,), verbose=False)
    return macs / 1e9  # GMACs

def counts_to_metrics_paper(tp, tn, fp, fn):
    """
    Mengembalikan metrik sesuai paper:
    - mIoU (paper): TP/(TP+FP+FN)  [= IoU foreground]
    - F1/Dice (FG), Precision, SEN/Recall, ACC, SPE
    Tambahan info opsional: IoU_BG dan mIoU_macro (FG+BG)/2 (tidak dipakai di paper).
    """
    # definisi utama (sesuai paper)
    miou_paper = safe_div(tp, tp + fp + fn)       # disebut "mIoU" di paper

    # metrik pendukung
    dice_fg = safe_div(2*tp, 2*tp + fp + fn)      # = F1 (FG)
    prec    = safe_div(tp, tp + fp)
    sen     = safe_div(tp, tp + fn)               # recall / TPR
    acc     = safe_div(tp + tn, tp + tn + fp + fn)
    spe     = safe_div(tn, tn + fp)

    # info opsional (tidak memengaruhi mIoU paper)
    iou_bg        = safe_div(tn, tn + fp + fn)
    miou_macro    = 0.5 * (miou_paper + iou_bg)   # rata-rata FG+BG (standar umum)

    return {
        "mIoU_paper": miou_paper,
        "Dice_FG": dice_fg,
        "Precision": prec,
        "SEN": sen,
        "ACC": acc,
        "SPE": spe,
        "IoU_BG_info": iou_bg,
        "mIoU_macro_info": miou_macro,
    }

@torch.no_grad()
def evaluate(model, loader, device, thr=0.5, amp=True, save_samples=0, out_dir="metrics_samples"):
    model.eval()
    tot_tp = tot_tn = tot_fp = tot_fn = 0

    if save_samples > 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    saved = 0
    for imgs, masks, names in loader:
        imgs   = imgs.to(device, non_blocking=True)
        masks  = (masks > 0.5).float().to(device, non_blocking=True)  # pastikan biner

        with torch.amp.autocast(device_type='cuda', enabled=amp and device.type == 'cuda'):
            logits = model(imgs)                 # (B,1,H,W)
            probs  = torch.sigmoid(logits)
        preds = (probs >= thr).float()

        tp, tn, fp, fn = confusion_counts(preds, masks)
        tot_tp += tp; tot_tn += tn; tot_fp += fp; tot_fn += fn

        if save_samples and saved < save_samples:
            pr = preds.detach().cpu().numpy()
            for b in range(min(imgs.size(0), save_samples - saved)):
                pm = (pr[b, 0] * 255).astype(np.uint8)
                Image.fromarray(pm).save(Path(out_dir) / f"pred_{names[b]}")
                saved += 1
                if saved >= save_samples:
                    break

    return counts_to_metrics_paper(tot_tp, tot_tn, tot_fp, tot_fn)

def load_state_dict_flexible(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    for k in ["model_state", "state_dict", "model", "net"]:
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            state = state[k]; break
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="dataset/val")
    ap.add_argument("--img_size", type=int, nargs=2, default=[288, 384])  # H W
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt")
    ap.add_argument("--in_ch", type=int, default=3)
    ap.add_argument("--out_ch", type=int, default=1)
    ap.add_argument("--base_c", type=int, default=16)
    ap.add_argument("--maf_depth", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_samples", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset & loader
    ds = SegFolder(args.data_root, size=tuple(args.img_size), augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    # model
    model = MAFUNet(in_ch=args.in_ch, out_ch=args.out_ch,
                    base_c=args.base_c, maf_depth=args.maf_depth).to(device)

    # load checkpoint
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        load_state_dict_flexible(model, ckpt_path, device)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: checkpoint tidak ditemukan, evaluasi memakai bobot random.")

    # Params & FLOPs
    params_m = params_in_million(model)
    gmacs    = flops_in_gmacs(model, args.in_ch, args.img_size[0], args.img_size[1], device)

    # Metrics (sesuai paper)
    res = evaluate(model, dl, device, thr=args.thr, amp=args.amp,
                   save_samples=args.save_samples, out_dir="outputs/metrics_samples")

    # Tampilkan: mIoU (paper) + metrik pendukung, info tambahan di akhir
    print("\n================ Results (Paper) ================")
    print(f"Params(M)     : {params_m:.3f}")
    print(f"FLOPs(G)      : {gmacs:.3f} (GMACs)")
    print(f"mIoU          : {100*res['mIoU_paper']:.2f}")       # = IoU(FG)
    print(f"F1/Dice (FG)  : {100*res['Dice_FG']:.2f}")
    print(f"Precision     : {100*res['Precision']:.2f}")
    print(f"SEN           : {100*res['SEN']:.2f}")
    print(f"ACC           : {100*res['ACC']:.2f}")
    print(f"SPE           : {100*res['SPE']:.2f}")
    print("-----------------------------------------------")
    print(f"[Info] IoU_BG : {100*res['IoU_BG_info']:.2f}")
    print(f"[Info] mIoU_macro(FG+BG)/2 : {100*res['mIoU_macro_info']:.2f}")
    print("=================================================\n")

if __name__ == "__main__":
    main()
