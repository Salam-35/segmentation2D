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
from SelfONN_decoders import SelfONN_ResUnet
from tqdm import tqdm
import argparse

#creating command line arguments with argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='Initial mask dir')
parser.add_argument('--thres', type=float, required=True, help='Segmentation Threshold')
parser.add_argument('--name', type=str, required=True, help='name of the new folder to be generated')
parser.add_argument('--resdir', type=str, required=True, help='Result_dir')

#stores argument input
args = parser.parse_args()


if __name__ ==  '__main__':  
    # torch.set_num_threads(1)
    ################## Network hyper-parameters 
    parentdir = args.dir                    # main directory
    ONN = True                                 # set to 'True' if you are using ONN
    batch_size = 8                   # batch size, Change to fit hardware
    in_channels = 1                 # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays  
    out_channels = 1               # # '1' for binary class losses: 'BCELoss' or 'DiceLoss', 'number of classes' for multi-class losses 
    input_mean = [0.058]                   # Dataset mean, provide 3 numbers for RGB images or 1 number for gray scale images
    input_std = [0.2268]                     # Dataset std, provide 3 numbers for RGB images or 1 number for gray scale images
    class_weights = None             # class weights for multi class masks, default: none
    lossType = 'DiceLoss'                       # loss function: 'CrossEntropy' for multi-class. 'BCELoss' or 'DiceLoss' for binary class
    num_folds = 5                     # number of cross validation folds
    Resize_h = 256                       # network input size
    Resize_w = 256  
    load_model = False                   # specify full path of pretrained model pt file or set to False to load trained model
    Test_Mask =  True                    # set to true if you have the test masks
    model_name = 'refinementfull_resnet18*Unet'                   # name of trained model .pt file
    new_name = args.name                       # specify a new folder name to save test results, 
    fold_to_run = [1,5]                 # define as [] to loop through all folds, or specify start and end folds i.e. [3 5]
    seg_threshold = args.thres               # else set to False to overwrite test results genertaed by train code
    #                                                   # else set to False to overwrite test results genertaed by train code
    Results_path = args.resdir  +'/'+  'Results'             # main results file
    save_path = args.resdir  +'/'+  new_name                     # save path 
    generated_masks = args.resdir  +'/'+  'Generated_mask'         # path to save generated_masks for test set 
    ##################
    # test Directory 
    testdir = parentdir
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
    # Create  Directory
    if path.exists(generated_masks):
        pass
    else:
        os.mkdir(generated_masks) 
    # Create  Directory
    if new_name:
        generated_masks = generated_masks + '/' + new_name
    else:
        generated_masks = generated_masks + '/' + model_name
    if path.exists(generated_masks):
        pass
    else:
        os.mkdir(generated_masks)

    # conversion factor to convert class values from np.linspace(0,255,out_channels) to np.linspace(0,out_channels-1,out_channels)
    if out_channels==1:
        conv_fact = 255.0/1.0  
    else:
        conv_fact = 255.0/(out_channels-1)
 
    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}') 
    # Number of gpus
    if train_on_gpu: 
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            # multi_gpu = True
            # temp
            multi_gpu = False 
            # temo  
        else:
            multi_gpu = False 

    test_history = []
    index = [] 
    # loop through folds
    if not fold_to_run:
        loop_start = 1
        loop_end = num_folds+1
    else:
        loop_start = fold_to_run[0]
        loop_end = fold_to_run[1]+1
    for fold_idx in range(loop_start, loop_end): 
        print('#############################################################')
        print(f'started fold {fold_idx}')
        # save_file_name = save_path + '/' + model_name  + f'_fold_{fold_idx}.pt'
        testdir_fold = testdir + f'fold_{fold_idx}/' 

        gen_fold_mask = generated_masks + f'/fold_{fold_idx}'
        # Create  Directory
        if path.exists(gen_fold_mask): 
            pass
        else:
            os.mkdir(gen_fold_mask) 

        
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
            # # temp
            # import segmentation_models_pytorch as smp
            # model = smp.FPN('densenet121', in_channels=1, classes=2, activation=None) 
            # dc_file = Results_path+ '/' + model_name + '/' + model_name + f'_dict_fold_{fold_idx}.pt'
            # dict_checkpoint = torch.load(dc_file)
            # model.load_state_dict(dict_checkpoint['model_state_dict']) 
            # # temp
            del   pt_file, checkpoint
            # Set to evaluation mode
            model.eval() 
        
        model = model.to('cuda')  
 
        # check if model on cuda
        if next(model.parameters()).is_cuda:
            print('model device: cuda')
    
        # choose model loss function 
        if lossType == 'CrossEntropy': 
            if class_weights != None:
                class_weights = torch.tensor(class_weights).cuda() 
            criterion = nn.CrossEntropyLoss(class_weights) 
        elif lossType == 'BCELoss': 
            criterion = nn.BCELoss()   
        elif lossType == 'DiceLoss':
            from unet_loss import DiceLoss 
            criterion = DiceLoss() 
        elif lossType == 'CompoundLoss':
            from unet_loss import CompoundLoss 
            criterion = CompoundLoss()   
        # test dataloader
        categories, n_Class_test, img_names_test = Createlabels(testdir_fold, Seg_state=True)  
        if Test_Mask:
            test_ds = SegData(root_dir=testdir_fold, images_path='images' , masks_path='masks', img_names=img_names_test, h=Resize_h, w=Resize_w, 
                mean=input_mean, std=input_std, in_channels=in_channels,  out_channels=out_channels, return_path=True, ONN=ONN)
        else:
            test_ds = TestData(root_dir=testdir_fold, images_path='images', img_names=img_names_test, h=Resize_h, w=Resize_w, 
                mean=input_mean, std=input_std, in_channels=in_channels,  return_path=True, ONN=ONN) 
        test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)

        # release memeory (delete variables)
        del  n_Class_test, img_names_test 
        torch.cuda.empty_cache()


        # Set to evaluation mode
        model.eval() 
        if Test_Mask: 
            test_acc = 0.0 
            test_loss = 0.0
            test_IoU = 0.0 
            test_DSC = 0.0
            pbar = tqdm(test_dl, ncols=100, desc=f"Testing", leave=False)
            for _, (data, target, im_path) in enumerate(pbar):
                # Tensors to gpu
                if train_on_gpu:
                    data = data.to('cuda', non_blocking=True)
                    target = target.to('cuda', non_blocking=True)
                # compute network output
                output = model(data)
                # compute loss
                if lossType == 'CrossEntropy':  
                    loss = criterion(output, target.squeeze(1)) 
                    output_temp = F.softmax(output, dim=1)
                    target_temp = F.one_hot(target.squeeze(1), out_channels)
                    target_temp = target_temp.permute(0,1,3,2).permute(0,2,1,3)
                    _, output = torch.max(output, dim=1)
                elif lossType == 'BCELoss':  
                    output = torch.sigmoid(output) 
                    loss = criterion(output.squeeze(1), target.float().squeeze(1)) 
                    output = 1.0*(output>seg_threshold)
                elif lossType == 'DiceLoss':
                    output = torch.sigmoid(output)
                    loss = criterion(output, target) 
                    output = 1.0*(output>seg_threshold)
                elif lossType == 'CompoundLoss':
                    output = torch.sigmoid(output)
                    loss = criterion(output, target) 
                    output = 1.0*(output>seg_threshold)  
                test_loss += loss.item() * data.size(0)
                # compute accuracy 
                accuracy = calc_acc(target,output,seg_threshold)
                test_acc += accuracy.item() * data.size(0) 
                if out_channels==1:
                    # compute IoU
                    IoU  = compute_IoU(target,output,seg_threshold)
                    test_IoU += IoU * data.size(0) 
                    # compute DSC
                    DSC = compute_DSC(target,output,seg_threshold)
                    test_DSC += DSC * data.size(0)
                else:
                    # compute IoU
                    IoU = torch.zeros((out_channels-1,1))
                    for clas_i in range(1,out_channels):
                        IoU[clas_i-1,0]  = compute_IoU((target_temp[:,clas_i,:,:]),(output_temp[:,clas_i,:,:]),seg_threshold)
                    test_IoU += IoU * data.size(0)  
                    # compute DSC
                    DSC = torch.zeros((out_channels-1,1))
                    for clas_i in range(1,out_channels):
                        DSC[clas_i-1,0]  = compute_DSC((target_temp[:,clas_i,:,:]),(output_temp[:,clas_i,:,:]),seg_threshold)
                    test_DSC += DSC * data.size(0)  
                # write generated masks to file
                for i in range(data.shape[0]):
                    tensor_image = output[i].squeeze(0).cpu()
                    # tensor_image = 1*(tensor_image>=0.5) 
                    # plt.imshow(tensor_image)
                    # plt.show() 
                    image_name = im_path[i] 
                    filename = gen_fold_mask+ '/' + image_name 
                    # tensor_image = tensor_image.type(torch.float) 
                    # save_image(tensor_image, filename)
                    tensor_image = tensor_image.numpy()  
                    tensor_image = conv_fact*tensor_image
                    # tensor_image = 125.0*(tensor_image==1.) + 255.0*(tensor_image==2.) 
                    tensor_image = tensor_image.astype('uint8') 
                    io.imsave(filename, tensor_image, check_contrast=False) 
        else:
            pbar = tqdm(test_dl, ncols=100, desc=f"Testing", leave=False)
            for data, im_path in enumerate(pbar): 
                # Tensors to gpu
                if train_on_gpu:
                    data = data.to('cuda', non_blocking=True)
                # compute network output
                out = model(data)
                # compute loss
                if lossType == 'CrossEntropy':  
                    output = F.softmax(output, dim=1)
                    _, output = torch.max(output, dim=1)
                elif lossType == 'BCELoss':  
                    output = torch.sigmoid(output) 
                    output = 1.0*(output>seg_threshold)
                elif lossType == 'DiceLoss':
                    output = torch.sigmoid(output)
                    output = 1.0*(output>seg_threshold) 
                elif lossType == 'CompoundLoss':
                    output = torch.sigmoid(output)
                    output = 1.0*(output>seg_threshold) 
                # write generated masks to file
                for i in range(data.shape[0]):
                    tensor_image = output[i].squeeze(0).cpu()
                    # tensor_image = 1*(tensor_image>=0.5) 
                    # plt.imshow(tensor_image)
                    # plt.show() 
                    image_name = im_path[i] 
                    filename = gen_fold_mask+ '/' + image_name 
                    # tensor_image = tensor_image.type(torch.float) 
                    # save_image(tensor_image, filename)
                    tensor_image = tensor_image.numpy()  
                    tensor_image = conv_fact*tensor_image
                    # tensor_image = 125.0*(tensor_image==1.) + 255.0*(tensor_image==2.) 
                    tensor_image = tensor_image.astype('uint8') 
                    io.imsave(filename, tensor_image, check_contrast=False) 
                
        if Test_Mask:    
            test_loss = test_loss / len(test_dl.dataset)
            test_loss = round(test_loss,4)
            test_acc = test_acc / len(test_dl.dataset) 
            test_acc = round(100*test_acc,2)  
            test_IoU = test_IoU / len(test_dl.dataset)   
            if out_channels != 1:
                test_IoU_per_class = test_IoU
                test_IoU = test_IoU.mean().item()
            test_IoU = round(100*test_IoU,2)
            test_DSC = test_DSC / len(test_dl.dataset)  
            if out_channels != 1:
                test_DSC_per_class = test_DSC
                test_DSC = test_DSC.mean().item()                     
            test_DSC = round(100*test_DSC,2) 
            # # test_history.append([test_loss, test_acc, test_IoU, test_DSC])  
            fold_test_history = [test_loss, test_acc, test_IoU, test_DSC]
            for clas_i in range(0,out_channels-1):
                fold_test_history.append(round(100*test_IoU_per_class[clas_i].item(),2))
                fold_test_history.append(round(100*test_DSC_per_class[clas_i].item(),2))
            test_history.append(fold_test_history)
            index.extend([f'fold_{fold_idx}'])                  
            print(f'Test Loss: {test_loss},  Test Accuracy: {test_acc:.2f}%')
            print(f'Test IoU: {test_IoU},  Test DSC: {test_DSC}')
            del target, loss, test_loss, accuracy, test_acc
            del IoU, test_IoU, DSC, test_DSC
        
        # release memeory (delete variables)
        del model, criterion, test_ds, test_dl
        del data, output
        del tensor_image, image_name, filename
        torch.cuda.empty_cache()  

        print(f'completed fold {fold_idx}')

    print('#############################################################') 

    if Test_Mask:  
        # # Saving Test Results
        TestChPoint = {}
        TestChPoint['test_history'] = test_history
        columns_names = ['loss', 'Accuracy', 'IoU','DSC']
        for clas_i in range(1,out_channels):
            columns_names.append(f'IoU_{clas_i}')
            columns_names.append(f'DSC_{clas_i}')
        # temp = pd.DataFrame(test_history,columns=['loss', 'Accuracy', 'IoU','DSC']) 
        temp = pd.DataFrame(test_history,columns=columns_names)  
        # compute average values
        test_loss = np.mean(temp['loss'])
        test_acc = np.mean(temp['Accuracy'])
        test_IoU = np.mean(temp['IoU'])
        test_DSC = np.mean(temp['DSC']) 
        # test_history.append([test_loss, test_acc, test_IoU, test_DSC])
        avg_test_history = [test_loss, test_acc, test_IoU, test_DSC]
        N = 2*(out_channels-1)
        for i in range(4,4+N):
            current_value = np.mean(temp[columns_names[i]])
            avg_test_history.append(current_value)
        test_history.append(avg_test_history)
        index.extend(['Average'])   
        # test_history = pd.DataFrame(test_history,columns=['loss', 'Accuracy', 'IoU','DSC'], index=index) 
        test_history = pd.DataFrame(test_history,columns=columns_names, index=index)   
        save_file = save_path +'/'+ model_name +'_test_results.pt'
        torch.save(TestChPoint, save_file) 
        # save to excel file
        save_file = save_path +'/'+ model_name  +'.xlsx'
        writer = pd.ExcelWriter(save_file, engine='openpyxl') 
        col =2; row =2 
        test_history.to_excel(writer, "Results", startcol=col,startrow=row)
        # save 
        writer.save()  

        print('\n') 
        print(test_history) 
        print('\n') 

        print('#############################################################') 
