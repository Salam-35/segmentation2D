# ============================================================
#  config_2d3d.py  —  2D→3D Pipeline Configuration
#  Preprocessing · 3D Evaluation · Postprocessing
# ============================================================

##### DO NOT EDIT THESE LINES #####
config = {}
####################################


# ────────────────────────────────────────────────────────────
#  PATHS
# ────────────────────────────────────────────────────────────

# Root of the extracted 2D dataset (contains Train/ Test/ Val/ subdirectories)
config['data_root'] = r'C:/Salam/AMOS/2D-Seg/Data'

# Root of the original 3D NIfTI volumes (used for 3D evaluation)
config['nifti_images_dir'] = r'C:/Salam/AMOS/3D/amos/imagesTs'   # CT volumes
config['nifti_labels_dir'] = r'C:/Salam/AMOS/3D/amos/labelsTs'   # GT labels

# Trained model checkpoint — output of TrainUnet.py
# Format: Results/<model_name>/<model_name>_fold_1.pt
config['model_path'] = r'C:/Salam/AMOS/2D-Seg/Results/mobilenet_v2_UnetPlusPlus/mobilenet_v2_UnetPlusPlus_fold_1.pt'

# Output directories
config['eval_output_dir'] = r'C:/Salam/AMOS/2D-Seg/Eval3D'           # 3D eval results (NIfTI + JSON)
config['pred_dir']        = r'C:/Salam/AMOS/2D-Seg/Predictions/raw'  # raw 2D predicted masks
config['post_output_dir'] = r'C:/Salam/AMOS/2D-Seg/Predictions/postprocessed'  # postprocessed masks


# ────────────────────────────────────────────────────────────
#  MODEL — must match exactly what was used in TrainUnet.py
# ────────────────────────────────────────────────────────────

config['in_channels']      = 1      # 1 for grayscale, 3 for RGB / multi-window
config['out_channels']     = 16     # number of classes (background + 15 organs)
config['model_input_size'] = 256    # spatial size the model was trained on (Resize_h in config.py)

# Normalization stats used during training (from config.py)
# For in_channels=1, provide 1 value. For in_channels=3, provide 3 values.
config['input_mean'] = [0.2277]
config['input_std']  = [0.2317]


# ────────────────────────────────────────────────────────────
#  PREPROCESSING  (preprocessing.py)
# ────────────────────────────────────────────────────────────

# Which HU windowing method to apply to create preprocessed dataset copies.
# For in_channels=1 models use 'grayscale' (no windowing, just normalization).
# For in_channels=3 models choose a multi-window variant.
#
# Available:
#   'grayscale'                       — 1ch: pixel = (HU+1024)/4095
#   'single_window'                   — 3ch: organ window replicated x3
#   'multi_window'                    — 3ch: soft-tissue / organ / vessel windows
#   'multi_window_clahe'              — 3ch: multi_window + CLAHE on each channel
#   'multi_window_clahe_unsharp'      — 3ch: multi_window_clahe + unsharp on organ channel
#   'multi_window_gamma'              — 3ch: multi_window + gamma correction
#   'multi_window_clahe_gamma_unsharp'— 3ch: full pipeline
config['preprocess_method'] = 'grayscale'

# Splits to preprocess: list of (subdirectory, fold_name) tuples
config['splits'] = [
    ('Train', 'fold_1'),
    ('Test',  'fold_1'),
    ('Val',   'fold_1'),
]


# ────────────────────────────────────────────────────────────
#  POSTPROCESSING  (postprocessing.py)
# ────────────────────────────────────────────────────────────

# Which postprocessing method to apply to raw predicted 2D masks.
#
# Available:
#   'cca'               — keep largest connected component per organ (removes blobs)
#   'morphological'     — binary closing per class (fills small holes)
#   'cca_morph'         — CCA then morphological closing (recommended)
#   'slice_consistency' — 2.5D mode filter across adjacent slices
#   'full'              — CCA + morphological closing + slice consistency
config['postprocess_method'] = 'cca_morph'
