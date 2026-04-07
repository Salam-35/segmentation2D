"""
postprocessing.py
=================
Apply named postprocessing to raw model prediction masks.
Input:  directory of predicted uint8 PNG masks (values 0-15)
Output: new directory with postprocessed masks

Usage:
    python postprocessing.py --method cca
    python postprocessing.py --method morphological
    python postprocessing.py --method cca_morph
    python postprocessing.py --method slice_consistency
    python postprocessing.py --method full
    python postprocessing.py --list

    Optional flags:
    --pred_dir  path to predicted masks directory
    --out_dir   output directory
    --n_classes number of classes (default 16)
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 16   # 0 background + 15 organs


# ──────────────────────────────────────────────────────────────
#  LOW-LEVEL OPS
# ──────────────────────────────────────────────────────────────

def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert('L'), dtype=np.uint8)


def save_mask(mask: np.ndarray, path: Path):
    Image.fromarray(mask.astype(np.uint8), mode='L').save(str(path))


# ──────────────────────────────────────────────────────────────
#  POSTPROCESSING OPERATIONS
# ──────────────────────────────────────────────────────────────

def connected_component_analysis(mask: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """
    For each organ class, keep only the largest connected component.
    Removes floating false positive blobs.
    """
    out = np.zeros_like(mask)
    for c in range(1, n_classes):
        binary = (mask == c).astype(np.uint8)
        if binary.sum() == 0:
            continue
        labeled, n_comp = ndimage.label(binary)
        if n_comp == 0:
            continue
        # Keep largest component
        sizes = ndimage.sum(binary, labeled, range(1, n_comp + 1))
        largest = np.argmax(sizes) + 1
        out[labeled == largest] = c
    return out


def morphological_closing(mask: np.ndarray, n_classes: int = N_CLASSES,
                           kernel_size: int = 3) -> np.ndarray:
    """
    Binary closing per class — fills small holes in predictions.
    """
    out = np.zeros_like(mask)
    struct = ndimage.generate_binary_structure(2, 1)
    # Expand structure to kernel_size
    struct = ndimage.iterate_structure(struct, kernel_size // 2)

    for c in range(1, n_classes):
        binary = (mask == c).astype(bool)
        if binary.sum() == 0:
            continue
        closed = ndimage.binary_closing(binary, structure=struct)
        out[closed] = c
    return out


def slice_consistency_filter(masks: list[np.ndarray],
                              n_classes: int = N_CLASSES,
                              window: int = 3) -> list[np.ndarray]:
    """
    2.5D: apply per-class mode filter across adjacent slices.
    For each slice z, for each pixel (i,j), the output label is
    the most common label among slices [z-w, ..., z, ..., z+w].
    window=3 means 1 neighbor each side.

    Input:  list of (H, W) uint8 masks ordered by z
    Output: list of (H, W) uint8 masks, smoothed
    """
    D = len(masks)
    if D == 0:
        return masks

    H, W = masks[0].shape
    stack = np.stack(masks, axis=0).astype(np.uint8)  # (D, H, W)
    out   = np.zeros_like(stack)

    half = window // 2
    for z in range(D):
        z0 = max(0, z - half)
        z1 = min(D, z + half + 1)
        slab = stack[z0:z1]   # (window, H, W)

        # Mode per pixel across the slab
        # Fast: for each class vote
        votes = np.zeros((n_classes, H, W), dtype=np.uint8)
        for zi in range(slab.shape[0]):
            for c in range(n_classes):
                votes[c] += (slab[zi] == c).astype(np.uint8)
        out[z] = np.argmax(votes, axis=0).astype(np.uint8)

    return [out[z] for z in range(D)]


# ──────────────────────────────────────────────────────────────
#  METHODS (operate on a single mask unless stated)
# ──────────────────────────────────────────────────────────────

def method_cca(mask, **_):
    return connected_component_analysis(mask)


def method_morphological(mask, **_):
    return morphological_closing(mask)


def method_cca_morph(mask, **_):
    m = connected_component_analysis(mask)
    m = morphological_closing(m)
    return m


# slice_consistency is volume-level — handled separately in pipeline
def method_slice_consistency(mask, **_):
    # No-op at single-slice level; applied at volume level in pipeline
    return mask


def method_full(mask, **_):
    m = connected_component_analysis(mask)
    m = morphological_closing(m)
    return m   # slice_consistency applied at volume level in pipeline


METHODS = {
    "cca":               method_cca,
    "morphological":     method_morphological,
    "cca_morph":         method_cca_morph,
    "slice_consistency": method_slice_consistency,
    "full":              method_full,
}

# Methods that also need volume-level slice_consistency pass
NEEDS_SLICE_CONSISTENCY = {"slice_consistency", "full"}


# ──────────────────────────────────────────────────────────────
#  PIPELINE
# ──────────────────────────────────────────────────────────────

def process_directory(pred_dir: Path, out_dir: Path, method_name: str, fn):
    """
    Process all predicted masks in pred_dir.
    Groups slices by volume stem for slice_consistency.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_files = sorted(pred_dir.glob("*.png"))
    if not mask_files:
        log.error(f"No PNG files found in {pred_dir}")
        return

    log.info(f"Processing {len(mask_files)} masks → {out_dir}")

    # ── Step 1: per-slice processing ──
    temp_results = {}
    for mpath in tqdm(mask_files, desc="Per-slice", unit="slice"):
        mask = load_mask(mpath)
        processed = fn(mask)
        temp_results[mpath.stem] = processed

    # ── Step 2: volume-level slice_consistency (if needed) ──
    if method_name in NEEDS_SLICE_CONSISTENCY:
        log.info("Applying 2.5D slice consistency...")

        # Group by volume (filename format: {stem}_z{zzzz})
        volumes = defaultdict(list)
        for fname_stem in sorted(temp_results.keys()):
            # Extract volume stem: everything before _z followed by 4 digits
            parts = fname_stem.rsplit('_z', 1)
            if len(parts) == 2 and parts[1].isdigit():
                vol_stem = parts[0]
                z_idx    = int(parts[1])
                volumes[vol_stem].append((z_idx, fname_stem))
            else:
                # Can't group — just pass through
                volumes[fname_stem].append((0, fname_stem))

        for vol_stem, entries in tqdm(volumes.items(), desc="Volume consistency"):
            entries.sort(key=lambda x: x[0])
            ordered_stems  = [e[1] for e in entries]
            ordered_masks  = [temp_results[s] for s in ordered_stems]
            smoothed       = slice_consistency_filter(ordered_masks)
            for s, m in zip(ordered_stems, smoothed):
                temp_results[s] = m

    # ── Step 3: save ──
    for fname_stem, mask in temp_results.items():
        save_mask(mask, out_dir / f"{fname_stem}.png")

    log.info(f"Saved {len(temp_results)} masks to {out_dir}")


def main():
    # Load config_2d3d.py from the same directory
    import sys
    from importlib import import_module
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent))
    cfg = import_module('config_2d3d').config

    parser = argparse.ArgumentParser(description="AMOS Postprocessing Pipeline")
    parser.add_argument("--method",    type=str, default=None)
    parser.add_argument("--pred_dir",  type=str, default=cfg.get('pred_dir', ''),
                        help="Directory of raw predicted masks. Defaults to config['pred_dir']")
    parser.add_argument("--out_dir",   type=str, default=None,
                        help="Output directory. Defaults to config['post_output_dir'] or pred_dir/../<method>")
    parser.add_argument("--list",      action="store_true")
    args = parser.parse_args()

    if args.list or args.method is None:
        print("\nAvailable postprocessing methods:")
        for name in METHODS:
            print(f"  {name}")
        print("\nUsage: python postprocessing.py --method <name> --pred_dir <path>")
        return

    if args.method not in METHODS:
        log.error(f"Unknown method: '{args.method}'")
        return

    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.out_dir) if args.out_dir \
               else Path(cfg.get('post_output_dir', '')) if cfg.get('post_output_dir') \
               else pred_dir.parent / args.method

    fn = METHODS[args.method]
    log.info(f"Method      : {args.method}")
    log.info(f"Predictions : {pred_dir}")
    log.info(f"Output      : {out_dir}")

    process_directory(pred_dir, out_dir, args.method, fn)
    log.info("Done.")


if __name__ == "__main__":
    main()
