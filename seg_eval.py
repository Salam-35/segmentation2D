import cv2
import os
import numpy as np
from sklearn.metrics import jaccard_score, accuracy_score
import argparse
import re
import segmentation_models_pytorch as smp
import torch
from importlib import import_module 

#creating command line arguments with argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, help='Test or Train mode')
parser.add_argument('--gt', type=str, required=False, help='Ground truth dir (only for manual mode)')
parser.add_argument('--pred', type=str, required=False, help='Prediction dir (only for manual mode)')
parser.add_argument('--avg', default='binary', required=False, type=str, help='Jaccard Score average method')

#stores argument input
args = parser.parse_args()

#ground truth and prediction directory
mode = args.mode

if mode == 'train':
  fname = "config.py"
  configuration = import_module(fname.split(".")[0])
  config = configuration.config

  gt = config['parentdir'] + 'Data/Test/' + 'fold_'+ str(config['fold_to_run'][0]) + '/masks'
  pred = config['generated_masks'] + '/' + config['model_name'] + '/' + 'fold_'+ str(config['fold_to_run'][0])
  print('Predicting from:', pred)
  
elif mode == 'test':
  fname = "config_test.py"
  configuration = import_module(fname.split(".")[0])
  config = configuration.config
  

  gt = config['parentdir'] + 'Data/Test/' +  'fold_'+ str(config['fold_to_run'][0]) + '/masks'
  pred = config['generated_masks'] + '/' + config['new_name'] + '/' +  'fold_'+  str(config['fold_to_run'][0])
  print('Predicting from:', pred)


elif mode == 'manual':
  gt = args.gt
  pred = args.pred
  print('Predicting from:', pred)

print("\nEvaluating....\n")

#natural Sort
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

#loading mask paths
gt_images = sorted_alphanumeric([os.path.join(gt, x) for x in os.listdir(gt)])
pred_images = sorted_alphanumeric([os.path.join(pred, x) for x in os.listdir(pred)])

gt_img_list = []
pred_img_list = []

#storing ground truth and prediction masks in separate lists
for gt_img in gt_images:
    img = cv2.imread(gt_img, cv2.IMREAD_GRAYSCALE)
    img = img/255.0
    gt_img_list.append(img)

for pred_img in pred_images:
    img = cv2.imread(pred_img, cv2.IMREAD_GRAYSCALE)
    img = img/255.0
    pred_img_list.append(img)


def dice_coef(y_true, y_pred):
    #calculating according to formula
    intersection = np.logical_and(y_true, y_pred)
    im_sum = y_true.sum() + y_pred.sum()
    return 2. * intersection.sum() / im_sum


def iou(y_true, y_pred):
    #converting to 1-D np array
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    jac = jaccard_score(y_true, y_pred, average=args.avg)
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



if args.avg == 'binary':
    print(f"Accuracy:            {accuracy}, \nIoU/Jaccard Score:   {iou_score}, \nDice Score:          {dice}, \nPrecision:           {pre}, \nSensitivity:         {sens}, \nSpecificity:         {spec}, \nFalse_negative_rate: {fnr}, \nFalse_positive_rate: {fpr}")

else:
    print(f"Accuracy:                   {accuracy}, \n{args.avg.capitalize():8s} Avg Jaccard Score: {iou_score}, \nDice Score:                 {dice}")   