"""
evaluate_3d.py
==============
Full 3D evaluation pipeline for AMOS segmentation.

Steps:
  1. Load one NIfTI CT volume + ground truth label
  2. Extract 2D axial slices (same body-crop + resize as extraction pipeline)
  3. Run model inference slice by slice
  4. Reconstruct 3D prediction volume
  5. Compute 3D metrics per class: DSC, HD95, NSD

Usage:
    python evaluate_3d.py \
        --image   C:/Salam/AMOS/3D/amos/imagesTs/amos_0001.nii.gz \
        --label   C:/Salam/AMOS/3D/amos/labelsTs/amos_0001.nii.gz \
        --model   C:/Salam/AMOS/checkpoints/best_model.pth \
        --method  multi_window_clahe \
        --out_dir C:/Salam/AMOS/3D_eval/

    python evaluate_3d.py --help
"""

import argparse
import json
import logging
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import exposure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONSTANTS — must match extraction pipeline
# ─────────────────────────────────────────────
HU_OFFSET      = 1024
OUTPUT_SIZE    = (512, 512)
ROI_PADDING    = 20
BODY_THRESHOLD = 10
N_CLASSES      = 16

ORGAN_MAP = {
    0:  "background",
    1:  "spleen",
    2:  "right_kidney",
    3:  "left_kidney",
    4:  "gallbladder",
    5:  "esophagus",
    6:  "liver",
    7:  "stomach",
    8:  "aorta",
    9:  "inferior_vena_cava",
    10: "pancreas",
    11: "right_adrenal_gland",
    12: "left_adrenal_gland",
    13: "duodenum",
    14: "bladder",
    15: "prostate_uterus",
}

HU_WINDOWS = {
    "soft_tissue": (-60,  400),
    "organ":       ( 40,  400),
    "vessel":      (200,  700),
}
CLAHE_CLIP      = 0.02
CLAHE_TILE_GRID = (8, 8)


# ──────────────────────────────────────────────────────────────
#  PREPROCESSING (same as preprocessing.py)
# ──────────────────────────────────────────────────────────────

def hu_window(hu, center, width):
    lo, hi = center - width / 2, center + width / 2
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def apply_clahe(ch, shape):
    H, W = shape
    th = max(1, H // CLAHE_TILE_GRID[0])
    tw = max(1, W // CLAHE_TILE_GRID[1])
    u8 = (ch * 255).clip(0, 255).astype(np.uint8)
    return exposure.equalize_adapthist(u8, kernel_size=(th, tw),
                                       clip_limit=CLAHE_CLIP, nbins=256).astype(np.float32)


def apply_unsharp(ch, sigma=1.0, strength=0.5):
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(ch, sigma=sigma)
    return np.clip(ch + strength * (ch - blurred), 0.0, 1.0).astype(np.float32)


def apply_gamma(ch, gamma=0.8):
    return np.power(np.clip(ch, 0.0, 1.0), gamma).astype(np.float32)


PREPROCESS_METHODS = {
    # ── 1-channel (use when model trained with in_channels=1) ──────────────
    "grayscale": lambda hu: np.stack([
        np.clip((hu + HU_OFFSET) / 4095.0, 0.0, 1.0).astype(np.float32)
    ], axis=0),

    # ── 3-channel (use when model trained with in_channels=3) ──────────────
    "single_window": lambda hu: np.stack([
        hu_window(hu, *HU_WINDOWS["organ"])] * 3, axis=0),

    "multi_window": lambda hu: np.stack([
        hu_window(hu, *HU_WINDOWS["soft_tissue"]),
        hu_window(hu, *HU_WINDOWS["organ"]),
        hu_window(hu, *HU_WINDOWS["vessel"]),
    ], axis=0),

    "multi_window_clahe": lambda hu: np.stack([
        apply_clahe(hu_window(hu, *HU_WINDOWS["soft_tissue"]), hu.shape),
        apply_clahe(hu_window(hu, *HU_WINDOWS["organ"]),       hu.shape),
        apply_clahe(hu_window(hu, *HU_WINDOWS["vessel"]),      hu.shape),
    ], axis=0),

    "multi_window_clahe_unsharp": lambda hu: np.stack([
        apply_clahe(hu_window(hu, *HU_WINDOWS["soft_tissue"]), hu.shape),
        apply_unsharp(apply_clahe(hu_window(hu, *HU_WINDOWS["organ"]), hu.shape)),
        apply_clahe(hu_window(hu, *HU_WINDOWS["vessel"]), hu.shape),
    ], axis=0),

    "multi_window_gamma": lambda hu: np.stack([
        apply_gamma(hu_window(hu, *HU_WINDOWS["soft_tissue"])),
        apply_gamma(hu_window(hu, *HU_WINDOWS["organ"])),
        apply_gamma(hu_window(hu, *HU_WINDOWS["vessel"])),
    ], axis=0),

    "multi_window_clahe_gamma_unsharp": lambda hu: np.stack([
        apply_gamma(apply_clahe(hu_window(hu, *HU_WINDOWS["soft_tissue"]), hu.shape)),
        apply_unsharp(apply_gamma(apply_clahe(hu_window(hu, *HU_WINDOWS["organ"]), hu.shape))),
        apply_gamma(apply_clahe(hu_window(hu, *HU_WINDOWS["vessel"]), hu.shape)),
    ], axis=0),
}


def preprocess_slice(slice_hu: np.ndarray, method: str,
                     in_channels: int, mean: list, std: list,
                     input_size: int) -> torch.Tensor:
    """
    Raw HU (H, W) → normalized tensor (1, C, input_size, input_size) ready for model.

    in_channels : must match the model's expected input channels (1 or 3)
    mean / std  : per-channel normalization stats used during training
    input_size  : spatial size the model was trained on (e.g. 256)
    """
    fn      = PREPROCESS_METHODS[method]
    img_chw = fn(slice_hu)                        # (C, H, W) float [0,1]

    # Resize each channel to the model's training input size
    C, H, W = img_chw.shape
    if H != input_size or W != input_size:
        resized = np.zeros((C, input_size, input_size), dtype=np.float32)
        for c in range(C):
            u8  = (img_chw[c] * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(u8, mode='L').resize((input_size, input_size), Image.BILINEAR)
            resized[c] = np.array(pil, dtype=np.float32) / 255.0
        img_chw = resized

    # Normalize using the same stats as training
    mean_arr = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
    std_arr  = np.array(std,  dtype=np.float32).reshape(-1, 1, 1)
    img_chw  = (img_chw - mean_arr) / std_arr

    return torch.from_numpy(img_chw).unsqueeze(0).float()  # (1, C, input_size, input_size)


# ──────────────────────────────────────────────────────────────
#  BODY CROP (same as extraction pipeline)
# ──────────────────────────────────────────────────────────────

def get_body_crop(slice_hu: np.ndarray, H: int, W: int):
    body = slice_hu > BODY_THRESHOLD
    rows = np.any(body, axis=1)
    cols = np.any(body, axis=0)
    if not rows.any():
        return 0, H, 0, W
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    r1 = max(0, rmin - ROI_PADDING)
    r2 = min(H, rmax + ROI_PADDING + 1)
    c1 = max(0, cmin - ROI_PADDING)
    c2 = min(W, cmax + ROI_PADDING + 1)
    return r1, r2, c1, c2


def resize_slice(slice_hu: np.ndarray) -> np.ndarray:
    """Resize (H', W') HU slice to OUTPUT_SIZE preserving HU scale."""
    hu_min, hu_max = slice_hu.min(), slice_hu.max()
    if hu_max == hu_min:
        return np.zeros(OUTPUT_SIZE, dtype=np.float32)
    norm  = ((slice_hu - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
    pil   = Image.fromarray(norm, mode='L').resize((OUTPUT_SIZE[1], OUTPUT_SIZE[0]), Image.BILINEAR)
    arr   = np.array(pil).astype(np.float32) / 255.0
    return arr * (hu_max - hu_min) + hu_min


def resize_mask_back(mask: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Resize predicted mask (OUTPUT_SIZE) back to original crop size."""
    pil = Image.fromarray(mask.astype(np.uint8), mode='L')
    pil = pil.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return np.array(pil, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────
#  3D METRICS
# ──────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary DSC."""
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float(2 * inter / denom)


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray,
                           spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """
    95th percentile Hausdorff Distance (HD95) in mm.
    spacing: (z_spacing, y_spacing, x_spacing) in mm from NIfTI header.
    """
    pred_bool = pred.astype(bool)
    gt_bool   = gt.astype(bool)

    if pred_bool.sum() == 0 and gt_bool.sum() == 0:
        return 0.0
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return np.inf

    # Surface voxels
    pred_border = pred_bool ^ ndimage.binary_erosion(pred_bool)
    gt_border   = gt_bool   ^ ndimage.binary_erosion(gt_bool)

    # Distance transforms (in voxel units, then scale by spacing)
    dt_pred = distance_transform_edt(~pred_bool, sampling=spacing)
    dt_gt   = distance_transform_edt(~gt_bool,   sampling=spacing)

    d1 = dt_gt[pred_border]
    d2 = dt_pred[gt_border]

    if len(d1) == 0 or len(d2) == 0:
        return np.inf

    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95))


def normalized_surface_distance(pred: np.ndarray, gt: np.ndarray,
                                 spacing: tuple = (1.0, 1.0, 1.0),
                                 tau: float = 1.0) -> float:
    """
    Normalized Surface Distance (NSD).
    Fraction of surface within distance tau (mm) of the other surface.
    tau=1.0mm is standard for AMOS evaluation.
    """
    pred_bool = pred.astype(bool)
    gt_bool   = gt.astype(bool)

    if pred_bool.sum() == 0 and gt_bool.sum() == 0:
        return 1.0
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return 0.0

    pred_border = pred_bool ^ ndimage.binary_erosion(pred_bool)
    gt_border   = gt_bool   ^ ndimage.binary_erosion(gt_bool)

    dt_pred = distance_transform_edt(~pred_bool, sampling=spacing)
    dt_gt   = distance_transform_edt(~gt_bool,   sampling=spacing)

    # Fraction of gt border within tau of pred surface
    gt_close   = (dt_pred[gt_border]   <= tau).sum()
    pred_close = (dt_gt[pred_border] <= tau).sum()

    denom = gt_border.sum() + pred_border.sum()
    if denom == 0:
        return 1.0

    return float((gt_close + pred_close) / denom)


def compute_all_metrics(pred_vol: np.ndarray, gt_vol: np.ndarray,
                        spacing: tuple, n_classes: int = N_CLASSES) -> dict:
    """
    Compute DSC, HD95, NSD for each class.
    pred_vol, gt_vol: (H, W, D) uint8, values 0..n_classes-1
    spacing: (z_mm, y_mm, x_mm)
    """
    results = {}
    for c in range(1, n_classes):
        pred_c = (pred_vol == c)
        gt_c   = (gt_vol   == c)

        # Skip classes not present in GT (don't penalize)
        if gt_c.sum() == 0:
            if pred_c.sum() == 0:
                results[c] = {"dsc": 1.0, "hd95": 0.0, "nsd": 1.0, "present": False}
            else:
                results[c] = {"dsc": 0.0, "hd95": np.inf, "nsd": 0.0, "present": False}
            continue

        dsc  = dice_score(pred_c, gt_c)
        hd95 = hausdorff_distance_95(pred_c, gt_c, spacing)
        nsd  = normalized_surface_distance(pred_c, gt_c, spacing)

        results[c] = {
            "dsc":     round(dsc * 100, 4),
            "hd95":    round(hd95, 4),
            "nsd":     round(nsd * 100, 4),
            "present": True,
            "gt_voxels":   int(gt_c.sum()),
            "pred_voxels": int(pred_c.sum()),
        }

    return results


# ──────────────────────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """
    Load model saved by TrainUnet.py.
    Checkpoint format: {'model': <full model object>, 'history': [...]}
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and 'model' in ckpt:
        model = ckpt['model']
    elif hasattr(ckpt, 'parameters'):
        # Raw model object (not wrapped in dict)
        model = ckpt
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {checkpoint_path}.\n"
            "TrainUnet.py saves: torch.save({{'model': model, 'history': history}}, path)\n"
            "If your checkpoint differs, extract the model manually before calling this."
        )

    model.to(device)
    model.eval()
    log.info(f"Model loaded from {checkpoint_path}")
    return model


# ──────────────────────────────────────────────────────────────
#  MAIN INFERENCE + EVALUATION
# ──────────────────────────────────────────────────────────────

def infer_volume(nifti_image_path: str,
                 nifti_label_path: str,
                 model,
                 device: torch.device,
                 preprocess_method: str,
                 out_dir: Path,
                 in_channels: int = 1,
                 mean: list = None,
                 std: list = None,
                 input_size: int = 256) -> dict:

    # ── Load volume ──
    nii_img  = nib.load(nifti_image_path)
    nii_lbl  = nib.load(nifti_label_path)
    img_vol  = nii_img.get_fdata(dtype=np.float32)   # (H, W, D) raw HU
    gt_vol   = nii_lbl.get_fdata(dtype=np.float32)   # (H, W, D) labels

    # Voxel spacing in mm: (x, y, z) → we use (z, y, x) for distance transforms
    zooms    = nii_img.header.get_zooms()             # (x_mm, y_mm, z_mm)
    spacing  = (float(zooms[2]), float(zooms[1]), float(zooms[0]))  # (z, y, x)

    H, W, D = img_vol.shape
    log.info(f"Volume shape : {img_vol.shape}")
    log.info(f"Voxel spacing: {spacing} mm (z, y, x)")

    # ── Inference slice by slice ──
    pred_vol_original = np.zeros_like(gt_vol, dtype=np.uint8)
    t_start = time.time()

    for z in tqdm(range(D), desc="Inference", unit="slice"):
        slice_hu = img_vol[:, :, z]

        # Body crop (same as extraction)
        r1, r2, c1, c2  = get_body_crop(slice_hu, H, W)
        cropped_hu       = slice_hu[r1:r2, c1:c2]
        crop_hw          = (r2 - r1, c2 - c1)

        # Resize to model input size
        resized_hu = resize_slice(cropped_hu)

        # Preprocess → tensor  (resizes to input_size, applies mean/std)
        _mean = mean if mean is not None else [0.5] * in_channels
        _std  = std  if std  is not None else [0.5] * in_channels
        tensor = preprocess_slice(
            resized_hu, preprocess_method, in_channels, _mean, _std, input_size
        ).to(device)

        # Model forward
        with torch.no_grad():
            logits = model(tensor)                   # (1, C, input_size, input_size)
            # If model returns tuple (deep supervision), take first
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred_model = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize prediction back to crop size
        pred_crop = resize_mask_back(pred_model, crop_hw)

        # Place back into full-size slice
        pred_slice = np.zeros((H, W), dtype=np.uint8)
        pred_slice[r1:r2, c1:c2] = pred_crop
        pred_vol_original[:, :, z] = pred_slice

    infer_time = time.time() - t_start
    log.info(f"Inference done in {infer_time:.1f}s  ({infer_time/D*1000:.1f}ms/slice)")

    # ── Compute 3D metrics ──
    gt_u8 = gt_vol.astype(np.uint8)
    log.info("Computing 3D metrics...")
    metrics = compute_all_metrics(pred_vol_original, gt_u8, spacing)

    # ── Summary ──
    present_classes = [c for c, v in metrics.items() if v["present"]]
    mean_dsc  = np.mean([metrics[c]["dsc"]  for c in present_classes]) if present_classes else 0.0
    mean_hd95 = np.mean([metrics[c]["hd95"] for c in present_classes
                         if metrics[c]["hd95"] != np.inf]) if present_classes else 0.0
    mean_nsd  = np.mean([metrics[c]["nsd"]  for c in present_classes]) if present_classes else 0.0

    log.info(f"\n{'='*55}")
    log.info(f"{'Class':<25} {'DSC %':>8} {'HD95 mm':>10} {'NSD %':>8}")
    log.info(f"{'-'*55}")
    for c in range(1, N_CLASSES):
        name = ORGAN_MAP.get(c, f"class_{c}")
        if c in metrics:
            m = metrics[c]
            hd_str = f"{m['hd95']:>10.2f}" if m['hd95'] != np.inf else f"{'inf':>10}"
            present_str = "" if m["present"] else "  (absent)"
            log.info(f"{name:<25} {m['dsc']:>8.2f} {hd_str} {m['nsd']:>8.2f}{present_str}")
    log.info(f"{'-'*55}")
    log.info(f"{'Mean (present classes)':<25} {mean_dsc:>8.2f} {mean_hd95:>10.2f} {mean_nsd:>8.2f}")
    log.info(f"{'='*55}")

    # ── Save outputs ──
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predicted NIfTI (same header as input)
    pred_nii = nib.Nifti1Image(pred_vol_original.astype(np.uint8), nii_img.affine, nii_img.header)
    stem     = Path(nifti_image_path).name.replace(".nii.gz", "").replace(".nii", "")
    nib.save(pred_nii, str(out_dir / f"{stem}_pred.nii.gz"))
    log.info(f"Saved predicted NIfTI → {out_dir / f'{stem}_pred.nii.gz'}")

    # Save metrics JSON
    report = {
        "volume":           stem,
        "preprocess_method":preprocess_method,
        "spacing_mm_zyx":   list(spacing),
        "inference_time_s": round(infer_time, 2),
        "mean_dsc":         round(mean_dsc,  4),
        "mean_hd95":        round(mean_hd95, 4),
        "mean_nsd":         round(mean_nsd,  4),
        "per_class": {
            ORGAN_MAP.get(c, f"class_{c}"): metrics[c]
            for c in range(1, N_CLASSES) if c in metrics
        }
    }
    # Replace inf with null for JSON
    def clean(obj):
        if isinstance(obj, float) and obj == np.inf:
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        return obj

    json_path = out_dir / f"{stem}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(clean(report), f, indent=2)
    log.info(f"Saved metrics JSON → {json_path}")

    return report


# ──────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    # Load config_2d3d.py from the same directory
    import sys
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).parent))
    cfg = import_module('config_2d3d').config

    parser = argparse.ArgumentParser(description="3D Evaluation — AMOS Segmentation")
    parser.add_argument("--image",   required=True,  help="Path to CT NIfTI (.nii.gz)")
    parser.add_argument("--label",   required=True,  help="Path to GT label NIfTI (.nii.gz)")
    parser.add_argument("--model",   default=None,   help="Path to model checkpoint (.pt). Defaults to config['model_path']")
    parser.add_argument("--method",  default=None,
                        choices=list(PREPROCESS_METHODS.keys()),
                        help="Preprocessing method. Defaults to config['preprocess_method']")
    parser.add_argument("--out_dir", default=None,   help="Output directory. Defaults to config['eval_output_dir']")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path  = args.model   or cfg['model_path']
    method      = args.method  or cfg['preprocess_method']
    out_dir     = Path(args.out_dir or cfg['eval_output_dir'])
    in_ch       = cfg['in_channels']
    mean        = cfg['input_mean']
    std         = cfg['input_std']
    inp_size    = cfg['model_input_size']

    device = torch.device(args.device)
    log.info(f"Device    : {device}")
    log.info(f"Image     : {args.image}")
    log.info(f"Label     : {args.label}")
    log.info(f"Checkpoint: {model_path}")
    log.info(f"Method    : {method}")
    log.info(f"in_channels={in_ch}, input_size={inp_size}")

    model = load_model(model_path, device)

    infer_volume(
        nifti_image_path  = args.image,
        nifti_label_path  = args.label,
        model             = model,
        device            = device,
        preprocess_method = method,
        out_dir           = out_dir,
        in_channels       = in_ch,
        mean              = mean,
        std               = std,
        input_size        = inp_size,
    )


if __name__ == "__main__":
    main()
