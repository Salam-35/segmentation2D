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

#natural Sort
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def dice_coef(y_true, y_pred):
    #calculating according to formula
    intersection = np.logical_and(y_true, y_pred)
    im_sum = y_true.sum() + y_pred.sum()
    return 2. * intersection.sum() / im_sum

def iou(y_true, y_pred):
    #converting to 1-D np array
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    jac = jaccard_score(y_true, y_pred, average="binary")
    return jac

def acc(y_true, y_pred):
    #converting to 1-D np array
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    return acc

def smp_metrics(y_true, y_pred):
    y_true = torch.from_numpy(y_true)
    y_pred = torch.from_numpy(y_pred)
    tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_true, mode='binary', threshold=0.5)
    pre = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
    sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro")
    specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro")
    fnr= smp.metrics.false_negative_rate(tp, fp, fn, tn, reduction="micro")
    fpr= smp.metrics.false_positive_rate(tp, fp, fn, tn, reduction="micro")
    return pre*100, sensitivity*100, specificity*100, fnr*100, fpr*100

def evaluation(gnd, predict, save = False):
    metrics = []
    for fold in os.listdir(predict):
      if 'fold' in fold:
        gt = os.path.join(os.path.join(gnd, fold), 'masks')
        pred = os.path.join(predict, fold)
        
        # print('Predicting from:', pred)

        #loading mask paths
        gt_images = sorted_alphanumeric([os.path.join(gt, x) for x in os.listdir(gt)])
        pred_images = sorted_alphanumeric([os.path.join(pred, x) for x in os.listdir(pred)])

        gt_img_list = []
        pred_img_list = []

        img = cv2.imread(pred_images[0], cv2.IMREAD_GRAYSCALE)
        image_shape = img.shape

        #storing ground truth and prediction masks in separate lists
        for gt_img in tqdm(gt_images):
            img = cv2.imread(gt_img, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (image_shape[1], image_shape[0]))
            img = img/255.0
            gt_img_list.append(img)

        for pred_img in tqdm(pred_images):
            img = cv2.imread(pred_img, cv2.IMREAD_GRAYSCALE)
            img = img/255.0
            pred_img_list.append(img)

        #converting to np array
        gt_img_list = np.asarray(gt_img_list)
        pred_img_list = np.asarray(pred_img_list)
        #converting to int
        gt_img_list = gt_img_list.astype(int)
        pred_img_list = pred_img_list.astype(int)

        accuracy = acc(gt_img_list, pred_img_list)*100
        iou_score = iou(gt_img_list, pred_img_list)*100
        dice = dice_coef(gt_img_list, pred_img_list)*100
        pre, sens, spec, fnr, fpr = smp_metrics(gt_img_list, pred_img_list)
        metrics.append([fold, accuracy, iou_score, dice, pre.item(), sens.item(), spec.item(), fnr.item(), fpr.item()])
    
    metric_df = pd.DataFrame(metrics, columns=['Fold', 'Accuracy', 'IoU', 'Dice_Score', 'Precision', 'Sensitivity', 'Specificity', 'False Negative Rate', 'False Positive Rate'])
    mean = metric_df.mean(axis = 0).tolist()
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
    
    