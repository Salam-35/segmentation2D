import cv2
import os
import numpy as np
import argparse
import re
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

#creating command line arguments with argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, required=True, help='ground_truth dir')
parser.add_argument('--pred', type=str, required=True, help='prediction dir')
parser.add_argument('--des', type=str, required=True, help='output dir')
parser.add_argument('--type', type=str, required=False, default='binary', help='ground_truth dir')


#stores argument input
args = parser.parse_args()

gt = args.gt
pred = args.pred
des = args.des


if not os.path.exists(des):
    os.makedirs(des, exist_ok = True)

#get size of one pred image (used for resizing the gt if needed)
img = cv2.imread(os.path.join(pred, os.listdir(pred)[0]), cv2.IMREAD_GRAYSCALE)
image_shape = img.shape

#mask preprocessing function similar to the one used in the dataloader of the pipeline
def mask_load(path):
    ONN = False
    mask_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_shape[1], image_shape[0]),  interpolation = transforms.InterpolationMode.NEAREST), #nearest neighbour interpolation is important for palette images
                    transforms.ToTensor(),  
                    ]) 
    mask  = Image.open(path)
    mask = np.array(mask)
    mask = mask_transforms(mask)*255
    mask = mask.long() 
    return mask


#separate arguments passed for binary and multiclass segmentation comparison
if args.type == 'binary':
    for mask_path in os.listdir(pred):
        #loading masks
        gt_img = cv2.imread(os.path.join(gt, mask_path), cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.resize(gt_img, (image_shape[1], image_shape[0]))
        #ensuring gt is binary
        _, gt_img = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
        #normalizing
        gt_img = gt_img/255
        pred_img = cv2.imread(os.path.join(pred, mask_path), cv2.IMREAD_GRAYSCALE)
        pred_img = pred_img/255
        gt_img= np.asarray(gt_img).astype(int)
        pred_img= np.asarray(pred_img).astype(int)

        #TP FP FN TN area calculation
        match = np.equal(gt_img, pred_img).astype(int)
        TN = np.logical_and(gt_img == 0, pred_img == 0).astype(int)
        FP = np.logical_and(gt_img == 0, pred_img != 0).astype(int)
        FN = np.logical_and(gt_img != 0, pred_img == 0).astype(int)
        TP = match-TN

        #empty RGB image
        compare_image = np.zeros((256, 256, 3), dtype=np.uint8)

        #coloring comparison output
        compare_image[TP == 1] = [0, 255, 0]  # Green for TP
        compare_image[FP == 1] = [0, 255, 255]  # Cyan for FP
        compare_image[FN == 1] = [255, 0, 0]  # Red for FN

        plt.imsave(os.path.join(des, mask_path), compare_image)

#multiclass starts here
else:
    for mask_name in os.listdir(pred):
        #loading masks
        gt_mask = mask_load(os.path.join(gt, mask_name))
        pred_mask = mask_load(os.path.join(pred, mask_name))

        #reducing extra dimension
        gt_mask = gt_mask.squeeze(0).numpy().astype(np.uint8)
        pred_mask = pred_mask.squeeze(0).numpy().astype(np.uint8)

        #TP FP FN TN & confusion area calculation
        match = np.equal(gt_mask, pred_mask).astype(int)
        non_match = np.not_equal(gt_mask, pred_mask).astype(int)
        TN = np.logical_and(gt_mask == 0, pred_mask == 0).astype(int)
        FP = np.logical_and(gt_mask == 0, pred_mask != 0).astype(int)
        FN = np.logical_and(gt_mask != 0, pred_mask == 0).astype(int)
        TP = match-TN
        conf = non_match - np.logical_or(FP, FN).astype(int)

        #empty RGB image    
        compare_image = np.zeros((256, 256, 3), dtype=np.uint8)

        #coloring comparison output
        compare_image[TP == 1] = [0, 255, 0]  # Green for TP
        compare_image[FP == 1] = [0, 255, 255]  # Cyan for FP
        compare_image[FN == 1] = [255, 0, 0]  # Red for FN
        compare_image[conf == 1] = [255, 255, 0]  # Yellow for conf

        plt.imsave(os.path.join(des, mask_name), compare_image)