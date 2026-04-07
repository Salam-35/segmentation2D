import cv2
import os
import numpy as np
import argparse
import re
from matplotlib import pyplot as plt
from tqdm import tqdm

#natural sort
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

#creating command line arguments with argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, required=True, help='ground_truth dir')
parser.add_argument('--pred', type=str, required=True, help='prediction dir')
parser.add_argument('--des', type=str, required=True, help='output dir')


#stores argument input
args = parser.parse_args()

gt = args.gt
pred = args.pred
des = args.des

if not os.path.exists(des):
    os.makedirs(des, exist_ok = True)

#loading mask paths
gt_images = sorted_alphanumeric([os.path.join(gt, x) for x in os.listdir(gt)])
pred_images = sorted_alphanumeric([os.path.join(pred, x) for x in os.listdir(pred)])

gt_img_list = []
pred_img_list = []

img = cv2.imread(pred_images[0], cv2.IMREAD_GRAYSCALE)
image_shape = img.shape

#storing ground truth and prediction masks in separate lists
for gt_img in gt_images:
    img = cv2.imread(gt_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (image_shape[1], image_shape[0]))
    img = img/255.0
    gt_img_list.append(img)

for pred_img in pred_images:
    img = cv2.imread(pred_img, cv2.IMREAD_GRAYSCALE)
    img = img/255.0
    pred_img_list.append(img)

#converting to np array
gt_img_list = np.asarray(gt_img_list)
pred_img_list = np.asarray(pred_img_list)
#converting to int
gt_img_list = gt_img_list.astype(int)
pred_img_list = pred_img_list.astype(int)

pbar = tqdm(zip(gt_img_list, pred_img_list, gt_images), desc = 'Progress', ncols = 10)
for _, (gt_img, pred_img, path) in enumerate(pbar):
    comp_img = np.zeros((256, 256, 3),  dtype= np.uint8)
    for i in range(gt_img.shape[0]):
        for j in range(gt_img.shape[1]):
            #set true positive pixels to green
            if gt_img[i][j] == 1 and pred_img[i][j] == 1:
                comp_img[i][j][1] = 150
              

            #set false positive pixels to yellow 
            elif gt_img[i][j] == 0 and pred_img[i][j] == 1:
                comp_img[i][j][2] = 255
                comp_img[i][j][1] = 255

            #set false negative pixels to red   
            elif gt_img[i][j] == 1 and pred_img[i][j] == 0:
                comp_img[i][j][0] = 255
                

    name = path.split('/')[-1]
    cv2.imwrite( os.path.join(des, name), cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))