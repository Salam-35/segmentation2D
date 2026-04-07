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

from segmentation_models_pytorch.losses import DiceLoss as smp_DiceLoss
from segmentation_models_pytorch.losses import JaccardLoss, FocalLoss, LovaszLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss, MCCLoss, TverskyLoss

class SMP_CombinedLoss(torch.nn.Module):
    def __init__(self, loss_types, weights, smp_loss_mode):
        super(SMP_CombinedLoss, self).__init__()
        print("Combined Loss initatied with:")
        print(f"loss_types: {loss_types}, weights: {weights}")
        if disregard_background and calculate_loss_on_bg and out_channels != 1:
            class_list = list(range(1, out_channels))
        else:
            class_list = None
        self.losses = []
        for i, lossType in enumerate(loss_types):
            if lossType == 'SMP_DiceLoss':
                self.losses.append((smp_DiceLoss(mode=smp_loss_mode, classes=class_list), weights[i]))
            elif lossType == 'SMP_JaccardLoss':
                self.losses.append((JaccardLoss(mode=smp_loss_mode, classes=class_list), weights[i]))
            elif lossType == 'SMP_FocalLoss':
                self.losses.append((FocalLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_LovaszLoss':
                self.losses.append((LovaszLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_SoftBCEWithLogitsLoss':
                self.losses.append((SoftBCEWithLogitsLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_TverskyLoss':
                self.losses.append((TverskyLoss(mode=smp_loss_mode, classes=class_list, alpha=0.3, beta=0.7), weights[i]))
            elif lossType == 'SMP_SoftCrossEntropyLoss':
                self.losses.append((SoftCrossEntropyLoss(mode=smp_loss_mode), weights[i]))
            elif lossType == 'SMP_MCCLoss':
                self.losses.append((MCCLoss(mode=smp_loss_mode), weights[i]))
    
    def forward(self, inputs, targets):
        combined_loss = 0
        
        for loss, weight in self.losses:
            combined_loss += weight * loss(inputs, targets)
        return combined_loss

def Multi_Class_DSC(target,output,C,threshold, disregard_background = True):
    """
    Computes a Dice  from 2D input of class scores and a target of integer labels.

    Parameters
    ----------
    input : 
        size B x C x H x W representing class scores.
    target : 
        integer label representation of the ground truth, same size as the input.
        size B x 1 x H x W representing class scores, where each value, 0 < label_i < C-1

    Returns
    -------
    dice_total : float.
    total dice 
    """
    # output = F.softmax(output, dim=1)
    # target = F.one_hot(target, C)
    # target = target.permute(0,1,3,2).permute(0,2,1,3)

    # DSC = torch.zeros((C-1,1))
    # for clas_i in range(1,C):
    #     DSC[clas_i-1,0]  = DSC_training((target[:,clas_i,:,:]),(output[:,clas_i,:,:]),threshold)

    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes = C)
    DSC = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
    if disregard_background:
        DSC = DSC[:, 1:]
    #return DSC.mean().item()
    return DSC.mean(dim=0)

def Multi_Class_IoU(target,output,C,threshold, disregard_background = True):
    """
    Computes a IoU  from 2D input of class scores and a target of integer labels.

    Parameters
    ----------
    input : 
        size B x C x H x W representing class scores.
    target : 
        integer label representation of the ground truth, same size as the input.
        size B x 1 x H x W representing class scores, where each value, 0 < label_i < C-1

    Returns
    -------
    Iou_total : float.
    total iou
    """
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes = C)
    IOU = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
    if disregard_background:
        IOU = IOU[:, 1:]
    #return IOU.mean().item()
    return IOU.mean(dim=0)


# Parse command line arguments
fname = "config_test.py"
configuration = import_module(fname.split(".")[0])
config = configuration.config

if __name__ ==  '__main__':  
    # torch.set_num_threads(1)
    ################## Network hyper-parameters 
    parentdir = config['parentdir']                     # main directory
    ONN = config['ONN']                                 # set to 'True' if you are using ONN
    batch_size = config['batch_size']                   # batch size, Change to fit hardware
    in_channels = config['in_channels']                 # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays  
    out_channels = config['out_channels']               # # '1' for binary class losses: 'BCELoss' or 'DiceLoss', 'number of classes' for multi-class losses
    palette = config['palette']
    disregard_background = config['disregard_background'] 
    if 'calculate_loss_on_bg' in config:
        calculate_loss_on_bg = config['calculate_loss_on_bg']                 # set to True if you are using multi-label masks
    else:
        calculate_loss_on_bg = True
    input_mean = config['input_mean']                   # Dataset mean, provide 3 numbers for RGB images or 1 number for gray scale images
    input_std = config['input_std']                     # Dataset std, provide 3 numbers for RGB images or 1 number for gray scale images
    class_weights = config['class_weights']             # class weights for multi class masks, default: none
    lossType = config['lossType']                       # loss function: 'CrossEntropy' for multi-class. 'BCELoss' or 'DiceLoss' for binary class
    num_folds = config['num_folds']                     # number of cross validation folds
    Resize_h = config['Resize_h']                       # network input size
    Resize_w = config['Resize_w']  
    load_model = config['load_model']                   # specify full path of pretrained model pt file or set to False to load trained model
    Test_Mask =  config['Test_Mask']                    # set to true if you have the test masks
    model_name = config['model_name']                   # name of trained model .pt file
    new_name = config['new_name']                       # specify a new folder name to save test results, 
    fold_to_run = config['fold_to_run']                 # define as [] to loop through all folds, or specify start and end folds i.e. [3 5]
    seg_threshold = config['seg_threshold']              # else set to False to overwrite test results genertaed by train code
    #                                                   # else set to False to overwrite test results genertaed by train code
    Results_path = config['Results_path']               # main results file
    save_path = config['save_path']                     # save path 
    generated_masks = config['generated_masks']         # path to save generated_masks for test set 
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
    
    if out_channels == 1:
        smp_loss_mode = 'binary'
    else:
        smp_loss_mode = 'multiclass'
 
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
            checkpoint = torch.load(load_model, weights_only=False)
            model = checkpoint['model']  
            del checkpoint 
            # Set to evaluation mode
            model.eval()
        else: 
            pt_file = Results_path+ '/' + model_name + '/' + model_name + f'_fold_{fold_idx}.pt'
            checkpoint = torch.load(pt_file, weights_only=False)
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

        combined_loss_flag = False
        if '+' in lossType:
            lossType = lossType.replace(' ', '')
            lossType_list = lossType.split('+')
            weights_list = []
            losses_list = []
            for smp_loss in lossType_list:
                weights_list.append(float(smp_loss.split('*')[0]))
                losses_list.append(smp_loss.split('*')[1])

            criterion = SMP_CombinedLoss(losses_list, weights_list, smp_loss_mode)
        else:
            if disregard_background and calculate_loss_on_bg and out_channels != 1:
                class_list = list(range(1, out_channels))
            else:
                class_list = None
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
            elif lossType == 'SMP_DiceLoss':
                criterion = smp_DiceLoss(mode = smp_loss_mode, classes=class_list)
            elif lossType == 'SMP_JaccardLoss':
                criterion = JaccardLoss(mode = smp_loss_mode, classes=class_list)
            elif lossType == 'SMP_FocalLoss':
                criterion = FocalLoss(mode = smp_loss_mode)
            elif lossType == 'SMP_LovaszLoss':
                criterion = LovaszLoss(mode = smp_loss_mode)
            elif lossType == 'SMP_SoftBCEWithLogitsLoss':
                criterion = SoftBCEWithLogitsLoss(mode = smp_loss_mode)
            elif lossType == 'SMP_TverskyLoss':
                criterion = TverskyLoss(mode = smp_loss_mode, classes=class_list,alpha=0.3, beta=0.7)
            elif lossType == 'SMP_SoftCrossEntropyLoss':
                criterion = SoftCrossEntropyLoss(mode = smp_loss_mode)
            elif lossType == 'SMP_MCCLoss':
                criterion = MCCLoss(mode = smp_loss_mode)   
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

        if combined_loss_flag:

            # this is done to keep if SMP condition in utils.py for losses (lazy coding)
            lossType_temp = lossType
            lossType = lossType_list[0]

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
                    _, output = torch.max(output_temp, dim=1)
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
                elif 'SMP' in lossType:
                    loss = criterion(output, target.squeeze(1))
                    if out_channels == 1:
                        output = torch.sigmoid(output)
                        output = 1.0*(output>seg_threshold)
                    else:
                        #loss = criterion(output, target.squeeze(1)) 
                        #loss = criterion(output, target) 
                        #output_temp = F.softmax(output, dim=1)
                        #target_temp = F.one_hot(target.squeeze(1), out_channels)
                        #target_temp = target_temp.permute(0,1,3,2).permute(0,2,1,3)
                        output = F.softmax(output, dim=1)
                        _, output = torch.max(output, dim=1)  
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
                    IoU = Multi_Class_IoU(target.squeeze(1), output, out_channels, seg_threshold, disregard_background)
                    test_IoU += IoU * data.size(0)  
                    # compute DSC
                    DSC = Multi_Class_DSC(target.squeeze(1), output, out_channels, seg_threshold, disregard_background)
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
                    if out_channels == 1:
                        tensor_image = conv_fact*tensor_image
                        # tensor_image = 125.0*(tensor_image==1.) + 255.0*(tensor_image==2.) 
                        tensor_image = tensor_image.astype('uint8') 
                        io.imsave(filename, tensor_image, check_contrast=False) 
                    else:
                        tensor_image = tensor_image.astype('uint8') 
                        pil_image = Image.fromarray(tensor_image)
                        pil_image = pil_image.convert('P')
                        pil_image.putpalette(palette)
                        pil_image.save(filename) 
        else:
            pbar = tqdm(test_dl, ncols=100, desc=f"Testing", leave=False)
            for _, (data, im_path) in enumerate(pbar): 
                # Tensors to gpu
                if train_on_gpu:
                    data = data.to('cuda', non_blocking=True)
                # compute network output
                output = model(data)
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
                elif 'SMP' in lossType:
                    if out_channels == 1:
                        output = torch.sigmoid(output)
                        output = 1.0*(output>seg_threshold)
                    else:
                        #loss = criterion(output, target.squeeze(1)) 
                        #loss = criterion(output, target) 
                        
                        # target_temp = F.one_hot(target.squeeze(1), out_channels)
                        # target_temp = target_temp.permute(0,1,3,2).permute(0,2,1,3)
                        output= F.softmax(output, dim=1)
                        _, output = torch.max(output, dim=1) 
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
                    if out_channels == 1:
                        tensor_image = conv_fact*tensor_image
                        # tensor_image = 125.0*(tensor_image==1.) + 255.0*(tensor_image==2.) 
                        tensor_image = tensor_image.astype('uint8') 
                        io.imsave(filename, tensor_image, check_contrast=False) 
                    else:
                        tensor_image = tensor_image.astype('uint8') 
                        pil_image = Image.fromarray(tensor_image)
                        pil_image = pil_image.convert('P')
                        pil_image.putpalette(palette)
                        pil_image.save(filename)  
                
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
            if out_channels != 1:
                if disregard_background:
                    class_num = out_channels-1
                else:
                    class_num = out_channels
                for clas_i in range(0,class_num):
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
        
        # store lossType back for the new fold
        if combined_loss_flag:
            lossType = lossType_temp

    print('#############################################################') 

    if Test_Mask:  
        # # Saving Test Results
        TestChPoint = {}
        TestChPoint['test_history'] = test_history
        columns_names = ['loss', 'Accuracy', 'IoU','DSC']
        if out_channels != 1:
            if disregard_background and out_channels != 1:
                first_class = 1
            else:
                first_class = 0
            for clas_i in range(first_class, out_channels):
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
        writer._save()  

        print('\n') 
        print(test_history) 
        print('\n') 

        print('#############################################################') 