"""
preprocessing.py
================
Apply a named preprocessing method to raw uint16 PNG images.
Creates a new copy of the dataset with the method applied.

Usage:
    python preprocessing.py --method single_window
    python preprocessing.py --method multi_window
    python preprocessing.py --method multi_window_clahe
    python preprocessing.py --method multi_window_clahe_unsharp
    python preprocessing.py --method multi_window_gamma
    python preprocessing.py --method multi_window_clahe_gamma_unsharp
    python preprocessing.py --list

Output structure mirrors input:
    Data/Train/fold_1_<method>/images/
    Data/Train/fold_1_<method>/masks/   ← masks are just symlinked/copied unchanged
"""

import os
import shutil
import argparse
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import exposure
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

HU_OFFSET   = 1024   # same as extraction: stored_pixel = HU + 1024

# HU windows (center, width)
WINDOWS = {
    "soft_tissue": (-60,  400),
    "organ":       ( 40,  400),
    "vessel":      (200,  700),
}

# CLAHE settings
CLAHE_CLIP      = 0.02
CLAHE_TILE_GRID = (8, 8)

# Gamma
GAMMA_VALUE = 0.8

# Unsharp mask
UNSHARP_SIGMA    = 1.0
UNSHARP_STRENGTH = 0.5

# ImageNet stats (applied last, always)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  LOW-LEVEL OPS
# ──────────────────────────────────────────────────────────────

def load_hu(path: Path) -> np.ndarray:
    """Load uint16 PNG → float32 HU."""
    raw = np.array(Image.open(str(path)), dtype=np.int32)
    return (raw - HU_OFFSET).astype(np.float32)


def hu_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """HU slice → float32 [0, 1]."""
    lo = center - width / 2
    hi = center + width / 2
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def apply_clahe(ch: np.ndarray, shape) -> np.ndarray:
    """Float [0,1] channel → CLAHE → float [0,1]."""
    H, W  = shape
    tile_h = max(1, H // CLAHE_TILE_GRID[0])
    tile_w = max(1, W // CLAHE_TILE_GRID[1])
    u8 = (ch * 255).clip(0, 255).astype(np.uint8)
    result = exposure.equalize_adapthist(
        u8, kernel_size=(tile_h, tile_w),
        clip_limit=CLAHE_CLIP, nbins=256
    )
    return result.astype(np.float32)


def apply_unsharp(ch: np.ndarray) -> np.ndarray:
    """Unsharp masking on float [0,1] channel."""
    blurred = gaussian_filter(ch, sigma=UNSHARP_STRENGTH)
    sharpened = ch + UNSHARP_STRENGTH * (ch - blurred)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


def apply_gamma(ch: np.ndarray, gamma: float = GAMMA_VALUE) -> np.ndarray:
    """Gamma correction on float [0,1] channel."""
    return np.power(np.clip(ch, 0.0, 1.0), gamma).astype(np.float32)


def imagenet_normalize(img_chw: np.ndarray) -> np.ndarray:
    """
    (3, H, W) float [0,1] → ImageNet normalized float32.
    Stored as float32 [0,1] PNG via rescale — normalization is
    stored as metadata offset; actual norm applied in Dataset.
    We save the pre-norm image so Dataset can still choose.
    Just return as-is here; normalization lives in Dataset.
    """
    return img_chw


def to_rgb_png(img_chw: np.ndarray) -> np.ndarray:
    """(3, H, W) float [0,1] → (H, W, 3) uint8 for saving as RGB PNG."""
    hwc = np.transpose(img_chw, (1, 2, 0))
    return (hwc * 255).clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
#  PREPROCESSING METHODS
#  Each method: hu (H,W) float32 → (3,H,W) float32 [0,1]
# ──────────────────────────────────────────────────────────────

def method_single_window(hu: np.ndarray) -> np.ndarray:
    """Single soft-tissue window, replicated to 3 channels."""
    ch = hu_window(hu, *WINDOWS["organ"])
    return np.stack([ch, ch, ch], axis=0)


def method_multi_window(hu: np.ndarray) -> np.ndarray:
    """3 different HU windows → 3 channels."""
    ch1 = hu_window(hu, *WINDOWS["soft_tissue"])
    ch2 = hu_window(hu, *WINDOWS["organ"])
    ch3 = hu_window(hu, *WINDOWS["vessel"])
    return np.stack([ch1, ch2, ch3], axis=0)


def method_multi_window_clahe(hu: np.ndarray) -> np.ndarray:
    """3-window + CLAHE on each channel."""
    ch1 = apply_clahe(hu_window(hu, *WINDOWS["soft_tissue"]), hu.shape)
    ch2 = apply_clahe(hu_window(hu, *WINDOWS["organ"]),       hu.shape)
    ch3 = apply_clahe(hu_window(hu, *WINDOWS["vessel"]),      hu.shape)
    return np.stack([ch1, ch2, ch3], axis=0)


def method_multi_window_clahe_unsharp(hu: np.ndarray) -> np.ndarray:
    """3-window + CLAHE + unsharp mask on organ channel only."""
    ch1 = apply_clahe(hu_window(hu, *WINDOWS["soft_tissue"]), hu.shape)
    ch2 = apply_unsharp(apply_clahe(hu_window(hu, *WINDOWS["organ"]), hu.shape))
    ch3 = apply_clahe(hu_window(hu, *WINDOWS["vessel"]), hu.shape)
    return np.stack([ch1, ch2, ch3], axis=0)


def method_multi_window_gamma(hu: np.ndarray) -> np.ndarray:
    """3-window + gamma correction."""
    ch1 = apply_gamma(hu_window(hu, *WINDOWS["soft_tissue"]))
    ch2 = apply_gamma(hu_window(hu, *WINDOWS["organ"]))
    ch3 = apply_gamma(hu_window(hu, *WINDOWS["vessel"]))
    return np.stack([ch1, ch2, ch3], axis=0)


def method_multi_window_clahe_gamma_unsharp(hu: np.ndarray) -> np.ndarray:
    """Full pipeline: 3-window + CLAHE + gamma + unsharp."""
    ch1 = apply_gamma(apply_clahe(hu_window(hu, *WINDOWS["soft_tissue"]), hu.shape))
    ch2 = apply_unsharp(apply_gamma(apply_clahe(hu_window(hu, *WINDOWS["organ"]), hu.shape)))
    ch3 = apply_gamma(apply_clahe(hu_window(hu, *WINDOWS["vessel"]), hu.shape))
    return np.stack([ch1, ch2, ch3], axis=0)


# ─────────────────────────────────────────
#  REGISTRY — add new methods here
# ─────────────────────────────────────────
METHODS = {
    "single_window":                    method_single_window,
    "multi_window":                     method_multi_window,
    "multi_window_clahe":               method_multi_window_clahe,
    "multi_window_clahe_unsharp":       method_multi_window_clahe_unsharp,
    "multi_window_gamma":               method_multi_window_gamma,
    "multi_window_clahe_gamma_unsharp": method_multi_window_clahe_gamma_unsharp,
}


# ──────────────────────────────────────────────────────────────
#  PIPELINE
# ──────────────────────────────────────────────────────────────

def process_split(subdir: str, split_name: str, method_name: str, fn, data_root: Path):
    src_img_dir = data_root / subdir / split_name / "images"
    src_msk_dir = data_root / subdir / split_name / "masks"
    out_split   = f"{split_name}_{method_name}"
    dst_img_dir = data_root / subdir / out_split / "images"
    dst_msk_dir = data_root / subdir / out_split / "masks"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_msk_dir.mkdir(parents=True, exist_ok=True)

    if not src_img_dir.exists():
        log.warning(f"Source not found, skipping: {src_img_dir}")
        return

    img_files = sorted(src_img_dir.glob("*.png"))
    if not img_files:
        log.warning(f"No PNG files in {src_img_dir}")
        return

    log.info(f"[{subdir}/{split_name}] {len(img_files)} slices → {out_split}")

    for img_path in tqdm(img_files, desc=f"{subdir}/{split_name}", unit="slice"):
        # ── Load raw HU ──
        hu = load_hu(img_path)

        # ── Apply method ──
        img_chw = fn(hu)          # (3, H, W) float [0,1]

        # ── Save as RGB PNG ──
        rgb = to_rgb_png(img_chw)  # (H, W, 3) uint8
        Image.fromarray(rgb, mode='RGB').save(str(dst_img_dir / img_path.name))

        # ── Copy mask unchanged ──
        msk_path = src_msk_dir / img_path.name
        if msk_path.exists():
            shutil.copy2(str(msk_path), str(dst_msk_dir / img_path.name))

    # Copy metadata JSON if exists
    meta_src = data_root / subdir / split_name / "slice_metadata.json"
    if meta_src.exists():
        shutil.copy2(str(meta_src), str(data_root / subdir / out_split / "slice_metadata.json"))


def main():
    # Load config_2d3d.py from the same directory
    import sys
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).parent))
    cfg       = import_module('config_2d3d').config
    data_root = Path(cfg['data_root'])
    splits    = cfg['splits']

    parser = argparse.ArgumentParser(description="AMOS Preprocessing Pipeline")
    parser.add_argument("--method", type=str, default=None,
                        help="Preprocessing method name. Defaults to config['preprocess_method']")
    parser.add_argument("--list", action="store_true",
                        help="List all available methods")
    args = parser.parse_args()

    if args.list or args.method is None:
        if args.method is None and not args.list:
            args.method = cfg.get('preprocess_method')
        if args.list or args.method is None:
            print("\nAvailable preprocessing methods:")
            for name in METHODS:
                print(f"  {name}")
            print("\nUsage: python preprocessing.py --method <name>")
            return

    if args.method not in METHODS:
        log.error(f"Unknown method: '{args.method}'")
        log.error(f"Available: {list(METHODS.keys())}")
        return

    fn = METHODS[args.method]
    log.info(f"Method   : {args.method}")
    log.info(f"Data root: {data_root}")

    for subdir, split_name in splits:
        process_split(subdir, split_name, args.method, fn, data_root)

    log.info("Done.")


if __name__ == "__main__":
    main()
