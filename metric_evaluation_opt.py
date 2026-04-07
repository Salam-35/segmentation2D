import cv2
import os
import numpy as np
from sklearn.metrics import jaccard_score, accuracy_score
import argparse
import re
import segmentation_models_pytorch as smp
import torch
from importlib import import_module 
import pandas as pd
from tqdm import tqdm
import warnings
from PIL import Image
from torchvision import transforms

def _handle_zero_division(x, zero_division):
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x


#natural Sort
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def dice_coef(tp, fp, fn, tn, reduction="micro"):
    #calculating according to formula
    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = (2*tp / (2*tp + fp + fn))

        return score
    
    elif reduction == "micro-imagewise":
        zero_division = 'warn'
        tp = tp.sum(1)
        fp = fp.sum(1)
        fn = fn.sum(1)
        tn = tn.sum(1)
        x = tp + fp + fn
        #print((x==0).any())
        #print(tp, fp, fn)
        score = (2*tp / (2*tp + fp + fn))
        #score = _handle_zero_division(score, zero_division)
        score = score.mean()

        return score
    
    


def evaluation(gnd, predict, save = False, reduction = 'micro', classes = 1, class_map = [], disregard_background = True):
    seg_type = 'binary'
    thresh = 0.5
    num_classes = None

    if classes > 1:
       seg_type = 'multiclass'
       thresh = None
       num_classes = classes
       #reduction = None\

    metrics = []
    for fold in os.listdir(predict):
      if 'fold' in fold:
        gt = os.path.join(os.path.join(gnd, fold), 'masks')
        pred = os.path.join(predict, fold)
        
        # print('Predicting from:', pred)

        #loading mask paths
        gt_images = sorted_alphanumeric([os.path.join(gt, x) for x in os.listdir(gt)])
        pred_images = sorted_alphanumeric([os.path.join(pred, x) for x in os.listdir(pred)])
        TP = FP = FN = TN = flag = 0
        for mask_name in tqdm(os.listdir(pred), desc = fold):
            if seg_type == 'binary':
              pred_img = cv2.imread(os.path.join(pred, mask_name), cv2.IMREAD_GRAYSCALE)
              gt_img = cv2.imread(os.path.join(gt, mask_name), cv2.IMREAD_GRAYSCALE)
              image_shape = pred_img.shape
              gt_img = cv2.resize(gt_img, (image_shape[1], image_shape[0]))
              gt_img = gt_img/255
              pred_img = pred_img/255
              gt_img = gt_img.astype(int)
              pred_img = pred_img.astype(int)
              y_true = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0)
              #print(y_true.size())
              y_pred = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0)

            else:
              pred_img = Image.open(os.path.join(pred, mask_name))
              #pred_img = np.array(pred_img)
              gt_img = Image.open(os.path.join(gt, mask_name))
              #gt_img = np.array(gt_img)
              image_shape = pred_img.size
              mask_resize = transforms.Compose([
                    transforms.Resize((image_shape[1], image_shape[0]),  interpolation = transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor(),  
                    ]) 
              pred_img = 255*mask_resize(pred_img)
              gt_img = 255*mask_resize(gt_img)
              y_pred = pred_img.long()
              y_true = gt_img.long()
             
            

            tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_true, mode = seg_type, threshold = thresh, num_classes = num_classes)
            if disregard_background and seg_type == 'multiclass':
              tp = tp[:, 1:]
              fp = fp[:, 1:]
              fn = fn[:, 1:]
              tn = tn[:, 1:]

            if flag == 0:
                TP = tp
                FP = fp
                FN = fn
                TN = tn
                flag = 1             
            else:
               TP = torch.cat((TP, tp), dim=0)
               FP = torch.cat((FP, fp), dim=0)
               FN = torch.cat((FN, fn), dim=0)
               TN = torch.cat((TN, tn), dim=0)
        #print(TP.size())
        accuracy = smp.metrics.accuracy(TP, FP, FN, TN, reduction=reduction)*100
        iou_score = smp.metrics.iou_score(TP, FP, FN, TN, reduction=reduction)*100
        pre = smp.metrics.precision(TP, FP, FN, TN, reduction=reduction)*100
        #dice = dice_coef(TP, FP, FN, TN, reduction=reduction)*100
        dice = smp.metrics.f1_score(TP, FP, FN, TN, reduction=reduction)*100
        sens = smp.metrics.sensitivity(TP, FP, FN, TN, reduction=reduction)*100
        spec = smp.metrics.specificity(TP, FP, FN, TN, reduction=reduction)*100
        fnr= smp.metrics.false_negative_rate(TP, FP, FN, TN, reduction=reduction)*100
        fpr= smp.metrics.false_positive_rate(TP, FP, FN, TN, reduction=reduction)*100

        #print(dice)

        if reduction == None:
           accuracy = accuracy.mean(dim=0)
           iou_score = iou_score.mean(dim=0)
           pre = pre.mean(dim=0)
           dice = dice.mean(dim=0)
           sens = sens.mean(dim=0)
           spec = spec.mean(dim=0)
           fnr = fnr.mean(dim=0)
           fpr = fpr.mean(dim=0)
           
           disregard_factor = 0
           if disregard_background:
              disregard_factor = 1
           for i in range(0, classes-disregard_factor):
              metrics.append([fold, class_map[i+disregard_factor], accuracy[i].item(), iou_score[i].item(), dice[i].item(), pre[i].item(), sens[i].item(), spec[i].item(), fnr[i].item(), fpr[i].item()])
              
        else:
          metrics.append([fold, accuracy.item(), iou_score.item(), dice.item(), pre.item(), sens.item(), spec.item(), fnr.item(), fpr.item()])
    
    if reduction == None:
      metric_df = pd.DataFrame(metrics, columns=['Fold', 'Class', 'Accuracy', 'IoU', 'Dice_Score', 'Precision', 'Sensitivity', 'Specificity', 'False Negative Rate', 'False Positive Rate'])
      mean_metrics = metric_df.groupby('Class').mean(numeric_only=True).reset_index()
      mean_metrics['Fold'] = 'Mean'
      metric_df = pd.concat([metric_df, mean_metrics], axis=0, ignore_index=True)
    else:
      metric_df = pd.DataFrame(metrics, columns=['Fold', 'Accuracy', 'IoU', 'Dice_Score', 'Precision', 'Sensitivity', 'Specificity', 'False Negative Rate', 'False Positive Rate'])
      mean = metric_df.mean(axis = 0, numeric_only=True).tolist()
      last_row = ['Mean']
      last_row.extend(mean)
      metric_df.loc[len(metric_df)] = last_row

    if save:
      save_dir = predict.replace('Generated_mask', 'Results')
      save_dir_alter = predict.replace('Genereated_mask', 'Results')
      if os.path.exists(save_dir):
        metric_df.to_csv(os.path.join(save_dir, 'additional_metrics.csv'), index = False)
      elif os.path.exists(save_dir_alter):
        metric_df.to_csv(os.path.join(save_dir_alter, 'additional_metrics.csv'), index = False)
      else:
        metric_df.to_csv(os.path.join(predict, 'additional_metrics.csv'), index = False)
    display(metric_df)
    
    