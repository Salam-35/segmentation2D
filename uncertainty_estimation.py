"""
Uncertainty Estimation for 2D Medical Image Segmentation
=========================================================
Two complementary methods:
  1. Test-Time Augmentation (TTA)  — works on every saved model, no retraining
  2. Monte Carlo Dropout (MCD)     — enabled automatically when Dropout layers exist

Outputs
-------
Results/uncertainty/
  uncertainty_summary_TTA.csv      model × fold table (one row per checkpoint)
  uncertainty_summary_MCD.csv      same format, MCD method
  calibration_plots/               reliability diagram PNGs
  confidence_maps/<model>/<fold>/  entropy + variance heatmaps per test image
"""

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ── Project imports ───────────────────────────────────────────────────────────
from utils import SegData, Createlabels
import config as cfg_module

# ── Suppress the IPython terminal check from utils ────────────────────────────
try:
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = 'all'
except Exception:
    pass

# ── Configuration ─────────────────────────────────────────────────────────────
config = cfg_module.config

parentdir    = config['parentdir']
Results_path = config['Results_path']
in_channels  = config['in_channels']
out_channels = config['out_channels']
Resize_h     = config['Resize_h']
Resize_w     = config['Resize_w']
input_mean   = config['input_mean']
input_std    = config['input_std']
seg_threshold = config['seg_threshold']
ONN          = config['ONN']
batch_size   = 1   # process one image at a time for per-image uncertainty maps

testdir = parentdir + 'Data/Test/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on: {device}')

# Output directories
unc_dir       = os.path.join(Results_path, 'uncertainty')
cal_plots_dir = os.path.join(unc_dir, 'calibration_plots')
maps_dir      = os.path.join(unc_dir, 'confidence_maps')
for d in [unc_dir, cal_plots_dir, maps_dir]:
    os.makedirs(d, exist_ok=True)

EPS = 1e-8
TTA_PASSES = 6    # number of TTA augmented views (including original)
MCD_PASSES = 20   # number of MC Dropout stochastic forward passes


# ══════════════════════════════════════════════════════════════════════════════
# Helper: model utilities
# ══════════════════════════════════════════════════════════════════════════════

def get_model_out_channels(model, in_ch, h, w, dev):
    with torch.no_grad():
        dummy = torch.zeros(1, in_ch, h, w).to(dev)
        out = model(dummy)
    return out.shape[1]


def has_dropout(model):
    return any(isinstance(m, nn.Dropout) for m in model.modules())


def enable_dropout(model):
    """Activate only Dropout modules while keeping BatchNorm in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def get_probs(logits, n_out):
    """Convert raw logits to probability tensor."""
    if n_out == 1:
        return torch.sigmoid(logits)          # (B, 1, H, W)
    return F.softmax(logits, dim=1)           # (B, C, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# Test-Time Augmentation
# ══════════════════════════════════════════════════════════════════════════════

def tta_augment(x):
    """Return list of TTA_PASSES augmented views of x (B, C, H, W)."""
    return [
        x,
        torch.flip(x, dims=[3]),                      # H-flip
        torch.flip(x, dims=[2]),                      # V-flip
        torch.rot90(x, k=1, dims=[2, 3]),             # 90°
        torch.rot90(x, k=2, dims=[2, 3]),             # 180°
        torch.rot90(x, k=3, dims=[2, 3]),             # 270°
    ]


def tta_deaugment(preds):
    """Invert augmentations so all predictions are in original orientation."""
    return [
        preds[0],
        torch.flip(preds[1], dims=[3]),
        torch.flip(preds[2], dims=[2]),
        torch.rot90(preds[3], k=3, dims=[2, 3]),
        torch.rot90(preds[4], k=2, dims=[2, 3]),
        torch.rot90(preds[5], k=1, dims=[2, 3]),
    ]


@torch.no_grad()
def run_tta(model, data, n_out):
    """
    Returns
    -------
    mean_probs : (B, C, H, W)
    variance   : (B, H, W)  – averaged over classes
    entropy    : (B, H, W)
    """
    augmented = tta_augment(data)
    raw_preds = [model(aug) for aug in augmented]
    probs     = [get_probs(p, n_out) for p in raw_preds]
    aligned   = tta_deaugment(probs)

    stack = torch.stack(aligned, dim=0)        # (T, B, C, H, W)
    mean_probs = stack.mean(dim=0)             # (B, C, H, W)
    variance   = stack.var(dim=0).mean(dim=1)  # (B, H, W)
    entropy    = -(mean_probs * torch.log(mean_probs + EPS)).sum(dim=1)  # (B, H, W)
    return mean_probs, variance, entropy


@torch.no_grad()
def run_mcd(model, data, n_out):
    """
    Returns
    -------
    mean_probs : (B, C, H, W)
    variance   : (B, H, W)
    entropy    : (B, H, W)
    """
    probs_list = []
    for _ in range(MCD_PASSES):
        logits = model(data)
        probs_list.append(get_probs(logits, n_out))

    stack = torch.stack(probs_list, dim=0)     # (T, B, C, H, W)
    mean_probs = stack.mean(dim=0)
    variance   = stack.var(dim=0).mean(dim=1)
    entropy    = -(mean_probs * torch.log(mean_probs + EPS)).sum(dim=1)
    return mean_probs, variance, entropy


# ══════════════════════════════════════════════════════════════════════════════
# Uncertainty metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_ece(conf_flat, acc_flat, n_bins=15):
    """Expected Calibration Error. conf and acc are 1-D numpy arrays."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    N    = len(conf_flat)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf_flat >= lo) & (conf_flat < hi)
        if mask.sum() == 0:
            continue
        avg_conf = conf_flat[mask].mean()
        avg_acc  = acc_flat[mask].mean()
        ece     += (mask.sum() / N) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_brier(probs_flat, gt_oh_flat):
    """
    Multi-class Brier Score.
    probs_flat  : (N, C) float
    gt_oh_flat  : (N, C) float one-hot
    """
    return float(np.mean(np.sum((probs_flat - gt_oh_flat) ** 2, axis=1)))


def compute_auroc_ue(entropy_flat, pred_flat, gt_flat):
    """AUROC of entropy predicting per-pixel misclassification."""
    is_error = (pred_flat != gt_flat).astype(float)
    if is_error.sum() == 0 or is_error.sum() == len(is_error):
        return float('nan')
    return float(roc_auc_score(is_error, entropy_flat))


def pixel_accuracy_on_mask(pred_flat, gt_flat, mask):
    if mask.sum() == 0:
        return float('nan')
    return float((pred_flat[mask] == gt_flat[mask]).mean())


# ══════════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_heatmap(arr2d, save_path, cmap='hot'):
    """Save a (H, W) numpy array as a coloured heatmap PNG."""
    norm = (arr2d - arr2d.min()) / (arr2d.max() - arr2d.min() + EPS)
    colormap = plt.get_cmap(cmap)
    rgba = colormap(norm)
    img  = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
    img.save(save_path)


def plot_reliability_diagram(conf_flat, acc_flat, title, save_path, n_bins=15):
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_c  = (bins[:-1] + bins[1:]) / 2
    avg_acc, avg_conf = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf_flat >= lo) & (conf_flat < hi)
        if mask.sum() == 0:
            avg_acc.append(float('nan'))
            avg_conf.append((lo + hi) / 2)
        else:
            avg_acc.append(float(acc_flat[mask].mean()))
            avg_conf.append(float(conf_flat[mask].mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.bar(bin_c, avg_acc, width=1 / n_bins, alpha=0.6, label='Accuracy', align='center')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Per-image processing
# ══════════════════════════════════════════════════════════════════════════════

def process_batch(mean_probs, variance, entropy, target, n_out,
                  img_name, maps_out_dir):
    """
    Compute uncertainty metrics for one image and save heatmaps.
    All tensors on CPU.

    Returns dict with per-image metrics.
    """
    B = mean_probs.shape[0]
    assert B == 1, "process_batch expects batch_size=1"

    p  = mean_probs[0]          # (C, H, W)
    v  = variance[0]            # (H, W)
    h  = entropy[0]             # (H, W)
    gt = target[0].squeeze(0)   # (H, W) integer labels

    p_np  = p.numpy()           # (C, H, W)
    v_np  = v.numpy()           # (H, W)
    h_np  = h.numpy()           # (H, W)
    gt_np = gt.numpy().astype(int)  # (H, W)

    # Predicted class
    if n_out == 1:
        pred_np = (p_np[0] >= seg_threshold).astype(int)   # (H, W)
        conf_np = p_np[0]                                   # (H, W) confidence map
    else:
        pred_np = p_np.argmax(axis=0).astype(int)          # (H, W)
        conf_np = p_np.max(axis=0)                         # (H, W)

    # Flatten for metric computation
    H, W       = h_np.shape
    N          = H * W
    h_flat     = h_np.flatten()
    v_flat     = v_np.flatten()
    conf_flat  = conf_np.flatten()
    pred_flat  = pred_np.flatten()
    gt_flat    = gt_np.flatten()
    acc_flat   = (pred_flat == gt_flat).astype(float)

    # One-hot for Brier
    C = n_out
    gt_oh = np.eye(C, dtype=np.float32)[gt_flat]   # (N, C)
    p_flat = p_np.reshape(C, -1).T                  # (N, C)

    # Metrics
    mean_ent  = float(h_flat.mean())
    mean_var  = float(v_flat.mean())
    ece       = compute_ece(conf_flat, acc_flat)
    brier     = compute_brier(p_flat, gt_oh)
    auroc_ue  = compute_auroc_ue(h_flat, pred_flat, gt_flat)

    # Accuracy on certain vs uncertain pixels (split at median entropy)
    median_h  = np.median(h_flat)
    cert_mask = h_flat < median_h
    acc_cert  = pixel_accuracy_on_mask(pred_flat, gt_flat, cert_mask)
    acc_uncer = pixel_accuracy_on_mask(pred_flat, gt_flat, ~cert_mask)

    # Save heatmaps
    base = os.path.splitext(img_name)[0]
    save_heatmap(h_np,    os.path.join(maps_out_dir, f'{base}_entropy.png'),  cmap='hot')
    save_heatmap(v_np,    os.path.join(maps_out_dir, f'{base}_variance.png'), cmap='YlOrRd')
    save_heatmap(conf_np, os.path.join(maps_out_dir, f'{base}_confidence.png'), cmap='viridis')

    return {
        'img':           img_name,
        'mean_entropy':  mean_ent,
        'mean_variance': mean_var,
        'ece':           ece,
        'brier_score':   brier,
        'auroc_ue':      auroc_ue,
        'acc_certain':   acc_cert,
        'acc_uncertain': acc_uncer,
        # kept for global ECE/reliability diagram
        '_conf_flat':    conf_flat,
        '_acc_flat':     acc_flat,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per model-fold aggregation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_rows(rows, model_name, fold_idx, method):
    """Aggregate per-image dicts to a single model-fold summary row."""
    metrics = ['mean_entropy', 'mean_variance', 'ece', 'brier_score',
               'auroc_ue', 'acc_certain', 'acc_uncertain']
    agg = {'model': model_name, 'fold': fold_idx, 'method': method,
           'n_images': len(rows)}
    for m in metrics:
        vals = [r[m] for r in rows if not np.isnan(r[m])]
        agg[m] = float(np.mean(vals)) if vals else float('nan')
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Discover available model checkpoints
# ══════════════════════════════════════════════════════════════════════════════

def discover_models(results_path):
    """Scan Results directory and return {model_name: [fold_idx, ...]}."""
    pt_files = glob.glob(os.path.join(results_path, '*', '*.pt'))
    models   = {}
    for pt in pt_files:
        model_name = os.path.basename(os.path.dirname(pt))
        fname      = os.path.basename(pt)
        if '_fold_' not in fname:
            continue
        try:
            fold_idx = int(fname.split('_fold_')[1].split('.pt')[0])
        except (ValueError, IndexError):
            continue
        models.setdefault(model_name, []).append(fold_idx)
    # Sort folds for deterministic order
    for k in models:
        models[k] = sorted(set(models[k]))
    return models


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    models = discover_models(Results_path)
    if not models:
        print(f'No model checkpoints found under: {Results_path}')
        return
    print(f'Found {sum(len(v) for v in models.values())} checkpoint(s) '
          f'across {len(models)} model(s).')

    all_tta, all_mcd = [], []

    for model_name, folds in models.items():
        for fold_idx in folds:
            pt_file = os.path.join(
                Results_path, model_name,
                f'{model_name}_fold_{fold_idx}.pt'
            )
            if not os.path.isfile(pt_file):
                print(f'  [SKIP] checkpoint not found: {pt_file}')
                continue

            print(f'\n{"="*60}')
            print(f'  Model : {model_name}  Fold : {fold_idx}')

            # ── Load checkpoint ───────────────────────────────────────────
            checkpoint = torch.load(pt_file, weights_only=False, map_location=device)
            model      = checkpoint['model'].to(device)
            del checkpoint
            model.eval()

            # Detect actual output channels from dummy forward pass
            n_out = get_model_out_channels(model, in_channels, Resize_h, Resize_w, device)
            print(f'  out_channels detected: {n_out}')

            # ── Build data loader ─────────────────────────────────────────
            testdir_fold = testdir + f'fold_{fold_idx}/'
            if not os.path.isdir(testdir_fold):
                print(f'  [SKIP] test directory not found: {testdir_fold}')
                continue

            _, _, img_names_test = Createlabels(testdir_fold, Seg_state=True)
            test_ds = SegData(
                root_dir=testdir_fold, images_path='images', masks_path='masks',
                img_names=img_names_test, h=Resize_h, w=Resize_w,
                mean=input_mean, std=input_std,
                in_channels=in_channels, out_channels=n_out,
                return_path=True, ONN=ONN
            )
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 pin_memory=(device.type == 'cuda'), num_workers=0)

            # Output directories for confidence maps
            maps_tta_dir = os.path.join(maps_dir, model_name, f'fold_{fold_idx}', 'TTA')
            maps_mcd_dir = os.path.join(maps_dir, model_name, f'fold_{fold_idx}', 'MCD')
            os.makedirs(maps_tta_dir, exist_ok=True)

            rows_tta, rows_mcd = [], []
            use_mcd = has_dropout(model)

            if use_mcd:
                os.makedirs(maps_mcd_dir, exist_ok=True)
                print(f'  MC Dropout: enabled ({MCD_PASSES} passes)')
            else:
                print('  MC Dropout: no Dropout layers found — TTA only')

            # ── Per-image loop ────────────────────────────────────────────
            pbar = tqdm(test_dl, desc=f'  fold_{fold_idx}', ncols=90, leave=False)
            for data, target, img_name_tuple in pbar:
                img_name = img_name_tuple[0] if isinstance(img_name_tuple, (list, tuple)) else img_name_tuple
                data   = data.to(device)
                target_cpu = target.cpu()

                # TTA
                mean_p, var, ent = run_tta(model, data, n_out)
                rows_tta.append(process_batch(
                    mean_p.cpu(), var.cpu(), ent.cpu(), target_cpu,
                    n_out, img_name, maps_tta_dir
                ))

                # MCD (only if model has dropout)
                if use_mcd:
                    enable_dropout(model)
                    mean_p_mcd, var_mcd, ent_mcd = run_mcd(model, data, n_out)
                    model.eval()  # restore eval for next TTA iteration
                    rows_mcd.append(process_batch(
                        mean_p_mcd.cpu(), var_mcd.cpu(), ent_mcd.cpu(), target_cpu,
                        n_out, img_name, maps_mcd_dir
                    ))

            # ── Reliability diagram (TTA) ─────────────────────────────────
            all_conf = np.concatenate([r['_conf_flat'] for r in rows_tta])
            all_acc  = np.concatenate([r['_acc_flat']  for r in rows_tta])
            diag_title = f'{model_name}\nfold {fold_idx} — TTA'
            diag_path  = os.path.join(cal_plots_dir,
                                      f'{model_name}_fold_{fold_idx}_TTA_reliability.png')
            plot_reliability_diagram(all_conf, all_acc, diag_title, diag_path)

            # ── Aggregate ─────────────────────────────────────────────────
            agg_tta = aggregate_rows(rows_tta, model_name, fold_idx, 'TTA')
            all_tta.append(agg_tta)
            print(f'  TTA  → entropy={agg_tta["mean_entropy"]:.4f}  '
                  f'ECE={agg_tta["ece"]:.4f}  '
                  f'AUROC_UE={agg_tta["auroc_ue"]:.4f}')

            if rows_mcd:
                # Reliability diagram (MCD)
                all_conf_mcd = np.concatenate([r['_conf_flat'] for r in rows_mcd])
                all_acc_mcd  = np.concatenate([r['_acc_flat']  for r in rows_mcd])
                diag_path_mcd = os.path.join(cal_plots_dir,
                                             f'{model_name}_fold_{fold_idx}_MCD_reliability.png')
                plot_reliability_diagram(all_conf_mcd, all_acc_mcd,
                                         f'{model_name}\nfold {fold_idx} — MCD',
                                         diag_path_mcd)
                agg_mcd = aggregate_rows(rows_mcd, model_name, fold_idx, 'MCD')
                all_mcd.append(agg_mcd)
                print(f'  MCD  → entropy={agg_mcd["mean_entropy"]:.4f}  '
                      f'ECE={agg_mcd["ece"]:.4f}  '
                      f'AUROC_UE={agg_mcd["auroc_ue"]:.4f}')

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

    # ── Save summary tables ───────────────────────────────────────────────────
    col_order = ['model', 'fold', 'method', 'n_images',
                 'mean_entropy', 'mean_variance', 'ece', 'brier_score',
                 'auroc_ue', 'acc_certain', 'acc_uncertain']

    if all_tta:
        df_tta = pd.DataFrame(all_tta)[col_order]
        out_tta = os.path.join(unc_dir, 'uncertainty_summary_TTA.csv')
        df_tta.to_csv(out_tta, index=False, float_format='%.6f')
        print(f'\nTTA summary saved  → {out_tta}')
        print(df_tta.to_string(index=False))

    if all_mcd:
        df_mcd = pd.DataFrame(all_mcd)[col_order]
        out_mcd = os.path.join(unc_dir, 'uncertainty_summary_MCD.csv')
        df_mcd.to_csv(out_mcd, index=False, float_format='%.6f')
        print(f'\nMCD summary saved  → {out_mcd}')
        print(df_mcd.to_string(index=False))

    print('\nDone.')


if __name__ == '__main__':
    main()
