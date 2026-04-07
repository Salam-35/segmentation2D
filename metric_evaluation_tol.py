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

def dice_coef(tp, fp, fn, tn):
    #calculating according to formula
    dice = (2*tp)/(2*tp + fp + fn)
    return dice

def iou(tp, fp, fn, tn):
    #converting to 1-D np array
    jac = (tp)/(tp + fp + fn)
    return jac

def acc(tp, fp, fn, tn):
    #converting to 1-D np array
    acc = (tp + tn)/(tp + fp + tn + fn)
    return acc

def tol_get_stats(y_true, y_pred, t):
    tp = fp = fn = tn = 0
    for gt_mask, pred_mask in zip(y_true, y_pred):
        gt_mask = gt_mask.astype(np.uint16)
        pred_mask = pred_mask.astype(np.uint16)
        kernel = np.ones((t, t), np.uint16)
        d_gt_mask = cv2.dilate(gt_mask, kernel, iterations=1)
        d_pred_mask = cv2.dilate(pred_mask, kernel, iterations=1)
        tp_mat = cv2.bitwise_and(d_gt_mask, pred_mask)
        tp += tp_mat.sum()
        fp_mat = cv2.bitwise_and(cv2.bitwise_not(d_gt_mask), pred_mask)
        fp += fp_mat.sum()
        fn_mat = cv2.bitwise_and(cv2.bitwise_not(d_pred_mask), gt_mask)
        fn += fn_mat.sum()
        tn_mat = cv2.bitwise_and(cv2.bitwise_not(gt_mask), cv2.bitwise_not(pred_mask))
        tn += tn_mat.sum()
    return tp, fp, fn, tn

def smp_metrics(y_true, y_pred, t):
    # y_true = torch.from_numpy(y_true)
    # y_pred = torch.from_numpy(y_pred)
    tp, fp, fn, tn = tol_get_stats(y_true, y_pred, t)
    pre = (tp)/(tp+fp)
    sensitivity = (tp)/(tp+fn)
    specificity = (tn)/(tn+fp)
    fnr= (fn)/(tp+fn)
    fpr= (fp)/(tn+fp)
    ac = acc(tp, fp, fn, tn)
    dice = dice_coef(tp, fp, fn, tn)
    io = iou(tp, fp, fn, tn)

    return ac*100, io*100, dice*100, pre*100, sensitivity*100, specificity*100, fnr*100, fpr*100

def evaluation(gnd, predict, save = False, t=3):
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

        # accuracy = acc(gt_img_list, pred_img_list)*100
        # iou_score = iou(gt_img_list, pred_img_list)*100
        # dice = dice_coef(gt_img_list, pred_img_list)*100
        accuracy, iou_score, dice, pre, sens, spec, fnr, fpr = smp_metrics(gt_img_list, pred_img_list, t)
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
    
    