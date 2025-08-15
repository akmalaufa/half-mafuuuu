# eval_metrics.py  (+ F1 & Precision, FLOPs via deepcopy + SEN)
import argparse
import copy
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from half_mafunet import MAFUNet as MAFUNet
from train_half_mafunet import SegFolder  # reuse dataset class


# ---------- helpers ----------
@torch.no_grad()
def confusion_counts(pred, target):
    """
    pred, target: (B,1,H,W) in {0,1}
    Returns summed TP, TN, FP, FN over the batch.
    """
    pred = pred.long()
    target = target.long()
    tp = (pred & target).sum().item()
    tn = ((1 - pred) & (1 - target)).sum().item()
    fp = (pred & (1 - target)).sum().item()
    fn = ((1 - pred) & target).sum().item()
    return tp, tn, fp, fn

def safe_div(a, b, eps=1e-7):
    return float(a) / float(b + eps)

def compute_scores(tp, tn, fp, fn):
    """
    Return:
      mIoU, Dice(=DSC), ACC, SPE, SEN, PREC, F1  (all in [0,1])
    """
    # precision & recall (SEN)
    prec = safe_div(tp, tp + fp)
    sen  = safe_div(tp, tp + fn)          # sensitivity/recall (TPR)

    # F1 (identik dengan Dice utk biner)
    f1 = safe_div(2 * prec * sen, prec + sen)

    # Dice dihitung langsung dari TP/FP/FN (sama dengan F1)
    dice_fg = safe_div(2 * tp, 2 * tp + fp + fn)

    # foreground IoU
    iou_fg = safe_div(tp, tp + fp + fn)
    # background IoU for mIoU
    iou_bg = safe_div(tn, tn + fp + fn)
    miou = 0.5 * (iou_fg + iou_bg)

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    spe = safe_div(tn, tn + fp)           # specificity (TNR)

    return miou, dice_fg, acc, spe, sen, prec, f1

def params_in_million(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def flops_in_gmacs(model, in_ch, H, W, device):
    """
    Hitung FLOPs pada salinan model agar hook THOP tidak mempengaruhi model asli.
    """
    try:
        from thop import profile
    except ImportError:
        raise SystemExit("Package 'thop' belum terpasang. Install dengan: pip install thop")

    m = copy.deepcopy(model).to(device).eval()
    dummy = torch.randn(1, in_ch, H, W, device=device)
    macs, _ = profile(m, inputs=(dummy,), verbose=False)
    return macs / 1e9  # GMACs


@torch.no_grad()
def evaluate(model, loader, device, amp=True, save_samples=0, out_dir="metrics_samples"):
    model.eval()
    tot_tp = tot_tn = tot_fp = tot_fn = 0

    if save_samples > 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    saved = 0
    for imgs, masks, names in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
            logits = model(imgs)
            probs = torch.sigmoid(logits)

        preds = (probs > 0.5).float()

        tp, tn, fp, fn = confusion_counts(preds, masks)
        tot_tp += tp; tot_tn += tn; tot_fp += fp; tot_fn += fn

        # optional: save a few predictions (B/W)
        if save_samples and saved < save_samples:
            pr = preds.cpu().numpy()
            for b in range(min(imgs.size(0), save_samples - saved)):
                pm = (pr[b, 0] * 255).astype(np.uint8)
                Image.fromarray(pm).save(Path(out_dir) / f"pred_{names[b]}")
                saved += 1
                if saved >= save_samples:
                    break

    return compute_scores(tot_tp, tot_tn, tot_fp, tot_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="dataset/val",
                    help="folder subset yang mau dievaluasi (val atau test)")
    ap.add_argument("--img_size", type=int, nargs=2, default=[288, 384])
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt")
    ap.add_argument("--in_ch", type=int, default=3)
    ap.add_argument("--out_ch", type=int, default=1)
    ap.add_argument("--base_c", type=int, default=16)
    ap.add_argument("--maf_depth", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_samples", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset & loader
    ds = SegFolder(args.data_root, size=tuple(args.img_size), augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # model
    model = MAFUNet(in_ch=args.in_ch, out_ch=args.out_ch,
                    base_c=args.base_c, maf_depth=args.maf_depth).to(device)

    # load checkpoint if provided
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("model_state", state)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: checkpoint tidak ditemukan, evaluasi memakai bobot random.")

    # --- Params & FLOPs (FLOPs via deepcopy agar aman dari hook) ---
    params_m = params_in_million(model)
    gmacs = flops_in_gmacs(model, args.in_ch, args.img_size[0], args.img_size[1], device)

    # --- Metrics on dataset ---
    miou, dice, acc, spe, sen, prec, f1 = evaluate(
        model, dl, device, amp=args.amp, save_samples=args.save_samples,
        out_dir="outputs/metrics_samples"
    )

    # pretty print
    print("\n================ Results ================ ")
    print(f"Params(M) : {params_m:.3f}")
    print(f"FLOPs(G)  : {gmacs:.3f} (GMACs)")
    print(f"mIoU      : {miou*100:.2f}")
    print(f"F1 Score  : {f1*100:.2f}")
    print(f"Precision : {prec*100:.2f}")
    print(f"ACC       : {acc*100:.2f}")
    print(f"SPE       : {spe*100:.2f}")
    print(f"SEN       : {sen*100:.2f}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
