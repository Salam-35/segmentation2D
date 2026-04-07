# PyTorch
from torchvision import transforms
from torchvision.utils import save_image 
import torch
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
# Data science tools
import numpy as np
import pandas as pd 
import os
from skimage import io  
from sklearn.metrics import confusion_matrix 
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Timing utility
from timeit import default_timer as timer
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
#
from os import path 
from utils import * 
from unet import UNet
from modified_unet import M_UNet 
from keras_unet import K_UNet 
from importlib import import_module  


# Parse command line arguments
fname = "config_Detection.py"
configuration = import_module(fname.split(".")[0])
config = configuration.config

if __name__ ==  '__main__':  
    # torch.set_num_threads(1)
    ################## Network hyper-parameters 
    parentdir = config['parentdir']                     # main directory
    classes = config['classes'] 
    batch_size = config['batch_size']                   # batch size, Change to fit hardware
    in_channels = config['in_channels']                 # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays  
    out_channels = config['out_channels']               # Binary output is added so far: use '2' for NLLLoss, or '1' for 'BCELoss' or 'DiceLoss' 
    input_mean = config['input_mean']                   # Dataset mean, provide 3 numbers for RGB images or 1 number for gray scale images
    input_std = config['input_std']                     # Dataset std, provide 3 numbers for RGB images or 1 number for gray scale images
    lossType = config['lossType']                       # loss function: 'NLLLoss' or 'DiceLoss' 
    num_folds = config['num_folds']                     # number of cross validation folds
    Resize_h = config['Resize_h']                       # network input size
    Resize_w = config['Resize_w']  
    load_model = config['load_model']                   # specify full path of pretrained model pt file or set to False to load trained model
    model_name = config['model_name']                   # name of trained model .pt file
    new_name = config['new_name']                       # specify a new folder name to save test results, 
    #                                                   # else set to False to overwrite test results genertaed by train code
    Results_path = config['Results_path']               # main results file
    save_path = config['save_path']                     # save path 
    ##################
    # test Directory 
    testdir = parentdir + 'Data/Test/' 
    # Create  Directory
    if path.exists(Results_path):  
        pass
    else: 
        os.mkdir(Results_path)
    # Create  Directory
    if path.exists(save_path):
        pass
    else:
        os.mkdir(save_path) 

    
    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}') 
    # Number of gpus
    if train_on_gpu: 
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True 
        else:
            multi_gpu = False 

    test_history = []
    index = [] 
    # loop through folds
    for fold_idx in range(1, num_folds+1): 
        print('#############################################################')
        print(f'started fold {fold_idx}')
        # save_file_name = save_path + '/' + model_name  + f'_fold_{fold_idx}.pt'
        testdir_fold = testdir + f'fold_{fold_idx}/' 
        
        # load model 
        if load_model:
            checkpoint = torch.load(load_model)
            model = checkpoint['model']  
            del checkpoint 
            # Set to evaluation mode
            model.eval()
        else: 
            pt_file = Results_path+ '/' + model_name + '/' + model_name + f'_fold_{fold_idx}.pt'
            checkpoint = torch.load(pt_file)
            model = checkpoint['model'] 
            del   pt_file, checkpoint
            # Set to evaluation mode
            model.eval() 
        model = model.to('cuda')  
        # check if model on cuda
        if next(model.parameters()).is_cuda:
            print('model device: cuda')

        if lossType == 'NLLLoss':   
            softmax = nn.Softmax(dim=1)
        
        # test dataloader 
        categories, n_Class_test, img_names_test = Createlabels(testdir_fold, Seg_state=True)  
        test_ds = SegData(root_dir=testdir_fold, images_path='images' , masks_path='masks', img_names=img_names_test, h=Resize_h, w=Resize_w, 
            mean=input_mean, std=input_std, in_channels=in_channels,  return_path=True)
        test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)
        # release memeory (delete variables)
        del  n_Class_test, img_names_test 
        torch.cuda.empty_cache() 


        i=0
        for data, targets, _ in test_dl:
            # Tensors to gpu
            if train_on_gpu:
                data = data.to('cuda', non_blocking=True)
                targets = targets.to('cuda', non_blocking=True)
            # compute network output
            out = model(data) 
            # target:B x 1 x H x W,     output: B x 1 x H x W,     where all values [0,1] 
            if lossType == 'BCELoss': 
                out = torch.sigmoid(out) 
                # loss = criterion(out.squeeze(1), targets.float().squeeze(1))
            elif lossType == 'DiceLoss': 
                out = torch.sigmoid(out) 
                # loss = criterion(out, targets)
            elif lossType == 'NLLLoss': 
                out = softmax(out)
                out = out[:,1,:,:] 
                out_index = 1*(out>=0.5)
                out = out*out_index 
                out = out.unsqueeze(1) 
            # if at least one pixel is foreground then consider out/target as positive class
            out = out.sum(dim=3).sum(dim=2)
            out = 1*(out>0.0)
            targets = targets.sum(dim=3).sum(dim=2)
            targets = 1*(targets>0.0)  

            if i==0:
                fold_preds = out.detach().cpu().numpy()
                fold_targets = targets.detach().cpu().numpy()
            else:
                fold_preds = np.concatenate((fold_preds, out.detach().cpu().numpy()))
                fold_targets = np.concatenate((fold_targets, targets.detach().cpu().numpy()))
            i +=1
        if fold_idx==1:
            all_preds = fold_preds
            all_targets = fold_targets
        else:
            all_preds = np.concatenate((all_preds, fold_preds))
            all_targets = np.concatenate((all_targets, fold_targets))

        # release memeory (delete variables)
        del model, test_ds, test_dl
        del data, out
        del fold_preds, fold_targets
        torch.cuda.empty_cache()  

        print(f'completed fold {fold_idx}')

    print('#############################################################') 
    
    # from sklearn.metrics import confusion_matrix
    # main confusion matrix 
    cm = confusion_matrix(all_targets, all_preds)
    # create confusion matrix table (pd.DataFrame)
    cm_table = pd.DataFrame(cm, index=classes , columns=classes)

    # compute evaluation metircs
    Eval_Mat = []
    TN = cm[0][0] 
    FP = cm[0][1]   
    FN = cm[1][0]  
    TP = cm[1][1]  
    Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
    Precision = round(100*(TP)/(TP+FP), 2)  
    Sensitivity = round(100*(TP)/(TP+FN), 2) 
    F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)  
    Specificity = round(100*(TN)/(TN+FP), 2)  
    # create evaluation metrics table
    Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
    headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
    Eval_Mat_table = pd.DataFrame(Eval_Mat, index=['Overall'] ,columns=headers)

    # save to excel file
    if new_name:
        save_file = save_path +'/'+ new_name +'_Detection.xlsx'
    else:
        save_file = save_path +'/'+ model_name +'_Detection.xlsx' 
    writer = pd.ExcelWriter(save_file, engine='openpyxl') 
    # confusion matrix table
    col =2; row =2 
    cm_table.to_excel(writer, "Results", startcol=col,startrow=row)
    # evaluation metrics table
    col =6; row =2  
    Eval_Mat_table.to_excel(writer, "Results", startcol=col,startrow=row)
    # save
    writer.save() 

    print('\n') 
    print(cm_table) 
    print('\n') 
    print(Eval_Mat_table) 
    print('\n') 

    print('#############################################################') 
