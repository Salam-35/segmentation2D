import os
import re
import cv2
import torch
import warnings
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from tqdm import tqdm
from PIL import Image
from medpy.metric.binary import hd95
from torchvision import transforms

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    keyfn = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=keyfn)

def evaluation(
    gnd,
    predict,
    save=False,
    reduction='micro',
    classes=1,
    class_map=None,
    disregard_background=True
):
    """
    gnd:  path to ground-truth folder (fold*/masks)
    predict: path to predictions root (fold* subfolders)
    classes: number of classes (1 = binary)
    class_map: list of class names, length == classes
    """
    seg_type = 'binary' if classes == 1 else 'multiclass'
    thresh   = 0.5 if seg_type == 'binary' else None

    # which class‐indices to include
    cls_idxs = list(range(classes))
    if seg_type == 'multiclass' and disregard_background:
        cls_idxs = cls_idxs[1:]

    records = []

    for fold in sorted(os.listdir(predict)):
        if 'fold' not in fold:
            continue

        gt_dir = os.path.join(gnd, fold, 'masks')
        pr_dir = os.path.join(predict, fold)

        # HD95 storage
        if seg_type == 'binary':
            hd95_store = []
        else:
            hd95_store = {c: [] for c in cls_idxs}

        all_tp = all_fp = all_fn = all_tn = None

        for fname in tqdm(sorted_alphanumeric(os.listdir(pr_dir)), desc=fold):
            # ---- load masks ----
            if seg_type == 'binary':
                gt = cv2.imread(os.path.join(gt_dir,   fname), cv2.IMREAD_GRAYSCALE)
                pr = cv2.imread(os.path.join(pr_dir, fname), cv2.IMREAD_GRAYSCALE)
                pr = (pr/255).astype(np.uint8)
                gt = (gt/255).astype(np.uint8)
                if gt.shape != pr.shape:
                    gt = cv2.resize(gt, (pr.shape[1], pr.shape[0]), interpolation=cv2.INTER_NEAREST)

                # HD95
                try:
                    hd_val = hd95(gt.astype(bool), pr.astype(bool))
                except:
                    hd_val = np.nan
                hd95_store.append(hd_val)

                y_true = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
                y_pred = torch.from_numpy(pr).unsqueeze(0).unsqueeze(0)

            else:
                P = Image.open(os.path.join(pr_dir, fname))
                G = Image.open(os.path.join(gt_dir, fname))
                W,H = P.size

                to_lbl = transforms.Compose([
                    transforms.Resize((H,W), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor()
                ])
                pr_t = (255*to_lbl(P)).long().squeeze(0)  # H x W
                gt_t = (255*to_lbl(G)).long().squeeze(0)

                # HD95 per class
                pr_np = pr_t.cpu().numpy()
                gt_np = gt_t.cpu().numpy()
                for c in cls_idxs:
                    try:
                        hd_val = hd95((gt_np==c), (pr_np==c))
                    except:
                        hd_val = np.nan
                    hd95_store[c].append(hd_val)

                y_pred = pr_t.unsqueeze(0)
                y_true = gt_t.unsqueeze(0)

            # get stats
            tp, fp, fn, tn = smp.metrics.get_stats(
                y_pred, y_true,
                mode=seg_type,
                threshold=thresh,
                num_classes=classes
            )
            if seg_type == 'multiclass' and disregard_background:
                tp, fp, fn, tn = tp[:,1:], fp[:,1:], fn[:,1:], tn[:,1:]

            if all_tp is None:
                all_tp, all_fp, all_fn, all_tn = tp, fp, fn, tn
            else:
                all_tp = torch.cat((all_tp, tp), dim=0)
                all_fp = torch.cat((all_fp, fp), dim=0)
                all_fn = torch.cat((all_fn, fn), dim=0)
                all_tn = torch.cat((all_tn, tn), dim=0)

        # ---- handle reduction=None separately ----
        if reduction is None or reduction == 'none':
            # aggregate stats over images per class
            tp_s = all_tp.sum(0)
            fp_s = all_fp.sum(0)
            fn_s = all_fn.sum(0)
            tn_s = all_tn.sum(0)

            # per-class metrics
            acc_v  = smp.metrics.accuracy( tp_s, fp_s, fn_s, tn_s, reduction=None)*100
            iou_v  = smp.metrics.iou_score(tp_s, fp_s, fn_s, tn_s, reduction=None)*100
            f1_v   = smp.metrics.f1_score(  tp_s, fp_s, fn_s, tn_s, reduction=None)*100
            prec_v = smp.metrics.precision(tp_s, fp_s, fn_s, tn_s, reduction=None)*100
            sens_v = smp.metrics.sensitivity(tp_s, fp_s, fn_s, tn_s, reduction=None)*100
            spec_v = smp.metrics.specificity(tp_s, fp_s, fn_s, tn_s, reduction=None)*100
            fnr_v  = smp.metrics.false_negative_rate(tp_s, fp_s, fn_s, tn_s, reduction=None)*100
            fpr_v  = smp.metrics.false_positive_rate(tp_s, fp_s, fn_s, tn_s, reduction=None)*100

            # per-class HD95 means
            if seg_type == 'binary':
                hd_vals = [float(np.nanmean(hd95_store))]
            else:
                hd_vals = [float(np.nanmean(hd95_store[c])) for c in cls_idxs]

            # assemble rows per class
            for idx, c in enumerate(cls_idxs):
                records.append([
                    fold,
                    class_map[c],
                    acc_v[idx].item(),
                    iou_v[idx].item(),
                    f1_v[idx].item(),
                    prec_v[idx].item(),
                    sens_v[idx].item(),
                    spec_v[idx].item(),
                    fnr_v[idx].item(),
                    fpr_v[idx].item(),
                    hd_vals[idx]
                ])
            continue

        # ---- all other reductions ----
        acc   = smp.metrics.accuracy( all_tp, all_fp, all_fn, all_tn, reduction=reduction)*100
        iou   = smp.metrics.iou_score(all_tp, all_fp, all_fn, all_tn, reduction=reduction)*100
        f1    = smp.metrics.f1_score(  all_tp, all_fp, all_fn, all_tn, reduction=reduction)*100
        prec  = smp.metrics.precision(all_tp, all_fp, all_fn, all_tn, reduction=reduction)*100
        sens  = smp.metrics.sensitivity(all_tp, all_fp, all_fn, all_tn, reduction=reduction)*100
        spec  = smp.metrics.specificity(all_tp, all_fp, all_fn, all_tn, reduction=reduction)*100
        fnr   = smp.metrics.false_negative_rate(all_tp, all_fp, all_fn, all_tn, reduction=reduction)*100
        fpr   = smp.metrics.false_positive_rate(all_tp, all_fp, all_fn, all_tn, reduction=reduction)*100

        # HD95 flatten/mean exactly like SMP’s micro/macro logic
        if seg_type == 'binary':
            hd_all = hd95_store
        else:
            hd_all = sum([hd95_store[c] for c in cls_idxs], [])

        if reduction == 'micro':
            hd_val = float(np.nanmean(hd_all))
        elif reduction == 'macro':
            per_cls = [np.nanmean(hd95_store[c]) for c in cls_idxs]
            hd_val  = float(np.nanmean(per_cls))
        elif reduction in ['micro-imagewise','macro-imagewise','weighted-imagewise']:
            # average over classes per image, then over images
            n_imgs = len(hd95_store[cls_idxs[0]]) if seg_type!='binary' else len(hd_all)
            img_means = []
            for i in range(n_imgs):
                vals = ([hd_all[i]] if seg_type=='binary'
                        else [hd95_store[c][i] for c in cls_idxs])
                img_means.append(np.nanmean(vals))
            hd_val = float(np.nanmean(img_means))
        else:
            # fallback micro
            hd_val = float(np.nanmean(hd_all))

        # single row per fold
        records.append([
            fold,
            acc.item(),
            iou.item(),
            f1.item(),
            prec.item(),
            sens.item(),
            spec.item(),
            fnr.item(),
            fpr.item(),
            hd_val
        ])

    # build DataFrame
    if reduction is None or reduction=='none':
        cols = [
            'Fold','Class','Accuracy','IoU','Dice_Score','Precision',
            'Sensitivity','Specificity','False Negative Rate',
            'False Positive Rate','HD95'
        ]
        df = pd.DataFrame(records, columns=cols)

        # only numeric columns:
        num_cols = [c for c in df.columns if c not in ('Fold','Class')]
        # compute per-class means over numeric columns
        mean_df = df.groupby('Class')[num_cols].mean().reset_index()
        mean_df.insert(0, 'Fold', 'Mean')

        df = pd.concat([df, mean_df], ignore_index=True)
    else:
        cols = [
            'Fold','Accuracy','IoU','Dice_Score','Precision',
            'Sensitivity','Specificity','False Negative Rate',
            'False Positive Rate','HD95'
        ]
        df = pd.DataFrame(records, columns=cols)

        # only numeric cols (we know 'Fold' is non-numeric here)
        num_cols = df.columns.drop('Fold')
        mean_vals = df[num_cols].mean()

        # one-row DataFrame for the mean
        mean_row = pd.Series(['Mean'] + mean_vals.tolist(), index=cols)
        mean_df  = pd.DataFrame([mean_row], columns=cols)

        # concat instead of append
        df = pd.concat([df, mean_df], ignore_index=True)


    if save:
        out_dir = predict.replace('Generated_mask','Results')
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir,'additional_metrics.csv'), index=False)

    display(df)
