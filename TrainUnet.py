# PyTorch
from torchvision import transforms
from torchvision.utils import save_image 
import torch
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
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

from PIL import Image

# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from segmentation_models_pytorch.losses import DiceLoss as smp_DiceLoss
from segmentation_models_pytorch.losses import JaccardLoss, FocalLoss, LovaszLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss, MCCLoss, TverskyLoss

#
from os import path 
from utils import * 
from models import get_pretrained_model 
from importlib import import_module 
import shutil
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


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
fname = "config.py"
configuration = import_module(fname.split(".")[0])
config = configuration.config
 
if __name__ ==  '__main__':  
    # torch.set_num_threads(1)
    ################## Network hyper-parameters 
    parentdir = config['parentdir']                     # main directory
    ONN = config['ONN']                                 # set to 'True' if you are using ONN
    batch_size = config['batch_size']                   # batch size, Change to fit hardware
    in_channels = config['in_channels']                 # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays  
    out_channels = config['out_channels']               # '1' for binary class losses: 'BCELoss' or 'DiceLoss', 'number of classes' for multi-class losses 
    palette = config['palette']
    disregard_background = config['disregard_background']
    input_mean = config['input_mean']                   # Dataset mean, provide 3 numbers for RGB images or 1 number for gray scale images
    input_std = config['input_std']                     # Dataset std, provide 3 numbers for RGB images or 1 number for gray scale images
    optim_fc = config['optim_fc']                       # 'Adam' or 'SGD'
    lr =  config['lr']                                  # learning rate
    if 'calculate_loss_on_bg' in config:
        calculate_loss_on_bg = config['calculate_loss_on_bg']                 # set to True if you are using multi-label masks
    else:
        calculate_loss_on_bg = True
    class_weights = config['class_weights']             # class weights for multi class masks, default: none
    lossType = config['lossType']                       # loss function: 'CrossEntropy' for multi-class. 'BCELoss' or 'DiceLoss' for binary class
    n_epochs= config['n_epochs']                        # number of training epochs
    epochs_patience= config['epochs_patience']          # if val loss did not decrease for a number of epochs then decrease learning rate by a factor of lr_factor
    lr_factor= config['lr_factor']  
    max_epochs_stop = config['max_epochs_stop']         # maximum number of epochs with no improvement in validation loss for early stopping
    num_folds = config['num_folds']                     # number of cross validation folds
    Resize_h = config['Resize_h']                       # network input size
    Resize_w = config['Resize_w']  
    load_model = config['load_model']                   # specify path of pretrained model wieghts or set to False to train from scratch   
    Test_Mask =  config['Test_Mask']                    # set to true if you have the test masks
    model_type = config['model_type']                   # SMP libary models : SMP, Custom models : Custom
    model_to_load = config['model_to_load']             # chosse one of the following models: 'UNet' or 'M_UNet'
    model_name = config['model_name']                   # name of result folder 
    decoder_attention = config['decoder_attention']     # turns on attention layer for UNet/UNet++
    encoder_depth = config['encoder_depth']             # number of encoder layers
    encoder_weights = config['encoder_weights']         # pretrained weights or train from scratch
    activation = config['activation']                   # last layer activation function
    q_order = config['q_order']                         # ONN q-order
    max_shift = config['max_shift']                     # ONN max-shift
    seg_threshold = config['seg_threshold']              # Segmentation Threshold (Default 0.5)
    U_init_features = config['U_init_features']         # number of kernals in the first UNet conv layer
    if 'unfolding_decay' in config:
        unfolding_decay = config['unfolding_decay']         # unfolding or decay in the CSC_UNet or CSCA_UNet
    else:
        unfolding_decay = 1
    fold_to_run = config['fold_to_run']                 # define as [] to loop through all folds, or specify start and end folds i.e. [3 5]
    Results_path = config['Results_path']               # main results file
    save_path = config['save_path']                     # save path 
    generated_masks = config['generated_masks']         # path to save generated_masks for test set 
    ##################
    traindir = parentdir + 'Data/Train/'  
    testdir =  parentdir + 'Data/Test/' 
    valdir =  parentdir + 'Data/Val/'
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
    generated_masks = generated_masks + '/' + model_name  
    if path.exists(generated_masks):
        pass
    else:
        os.mkdir(generated_masks)

    shutil.copy('config.py', save_path)

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
    if not fold_to_run:
        loop_start = 1
        loop_end = num_folds+1
    else:
        loop_start = fold_to_run[0]
        loop_end = fold_to_run[1]+1
    for fold_idx in range(loop_start, loop_end): 
        print('#############################################################')
        if fold_idx==loop_start:
            print('training using '+model_to_load+' network') 
        print(f'started fold {fold_idx}')
        save_file_name = save_path + '/' + model_name + f'_fold_{fold_idx}.pt'
        # checkpoint_name = save_path + f'/checkpoint_fold_{fold_idx}.pt'
        checkpoint_name = save_path + f'/checkpoint.pt'  
        traindir_fold = traindir + f'fold_{fold_idx}/'
        testdir_fold = testdir + f'fold_{fold_idx}/' 
        valdir_fold = valdir + f'fold_{fold_idx}/' 

        gen_fold_mask = generated_masks + f'/fold_{fold_idx}'
        # Create  Directory
        if path.exists(gen_fold_mask):
            pass
        else:
            os.mkdir(gen_fold_mask) 


        # Create train labels
        categories, n_Class_train, img_names_train = Createlabels(traindir_fold, Seg_state=True)  
        class_num = len(categories)
        # Create val labels
        _, n_Class_val, img_names_val = Createlabels(valdir_fold, Seg_state=True)  

        # random shuffle before training 
        np.random.shuffle(img_names_train)  
        train_ds = SegData(root_dir=traindir_fold, images_path='images' , masks_path='masks', img_names=img_names_train, h=Resize_h, w=Resize_w,
            mean=input_mean, std=input_std, in_channels=in_channels,  out_channels=out_channels, return_path=False, ONN=ONN)
        if (len(train_ds)/batch_size)==0:
            train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=1)
        else:
            train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=1,drop_last=True) 
        # validation dataloader 
        val_ds = SegData(root_dir=valdir_fold, images_path='images' , masks_path='masks', img_names=img_names_val, h=Resize_h, w=Resize_w, 
            mean=input_mean, std=input_std, in_channels=in_channels,  out_channels=out_channels,  return_path=False, ONN=ONN) 
        val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)
        # test dataloader
        _, n_Class_test, img_names_test = Createlabels(testdir_fold, Seg_state=True)  
        if Test_Mask:
            test_ds = SegData(root_dir=testdir_fold, images_path='images' , masks_path='masks', img_names=img_names_test, h=Resize_h, w=Resize_w, 
            mean=input_mean, std=input_std, in_channels=in_channels,  out_channels=out_channels,  return_path=True, ONN=ONN) 
        else:
            test_ds = TestData(root_dir=testdir_fold, images_path='images', img_names=img_names_test, h=Resize_h, w=Resize_w, 
                mean=input_mean, std=input_std, in_channels=in_channels,  return_path=True, ONN=ONN)
        test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)
        
        # release memeory (delete variables) 
        del  n_Class_train, img_names_train
        del  n_Class_val, img_names_val
        del  n_Class_test, img_names_test 
        torch.cuda.empty_cache()

        if out_channels == 1:
           smp_loss_mode = 'binary'
        else:
            smp_loss_mode = 'multiclass'
        
        # load model 
        if load_model:
            checkpoint = torch.load(load_model)
            model = checkpoint['model']
            history = checkpoint['history']
            
            # Convert DataFrame back to list for continued training
            if isinstance(history, pd.DataFrame):
                history = history.values.tolist()
            
            start_epoch = len(history)
            print('Resuming training from checkpoint\n')
            del checkpoint
        else:
            history = []
            start_epoch = 0
            model = get_pretrained_model(parentdir, model_type, model_to_load, encoder_depth, encoder_weights, q_order, max_shift, U_init_features,in_channels,out_channels,train_on_gpu,multi_gpu, decoder_attention, activation, unfolding_decay) 
            model = model.to('cuda') 

        # model = model.to('cuda') 
        # check if model on cuda
        if next(model.parameters()).is_cuda:
            print('model device: cuda')
    
        # choose model loss function and optimizer
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
            combined_loss_flag = True

        else:
            if disregard_background and calculate_loss_on_bg and out_channels != 1:
                class_list = list(range(1, out_channels))
            else:
                class_list = None
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
                criterion = TverskyLoss(mode = smp_loss_mode, classes=class_list, alpha=0.3, beta=0.7)
            elif lossType == 'SMP_SoftCrossEntropyLoss':
                criterion = SoftCrossEntropyLoss(mode = smp_loss_mode)
            elif lossType == 'SMP_MCCLoss':
                criterion = MCCLoss(mode = smp_loss_mode)

        # optimizer
        if optim_fc == 'Adam':  
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False) 
        elif optim_fc == 'SGD': 
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False)
        elif optim_fc == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        elif optim_fc == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0.0001, momentum=0, centered=False)
        elif optim_fc == 'Adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=0.0001, initial_accumulator_value=0, eps=1e-10)
        elif optim_fc == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=0.0001)
        elif optim_fc == 'NAdam':
            optimizer = optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, momentum_decay=0.004)
        elif optim_fc == 'RAdam':
            optimizer = optim.RAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
        elif optim_fc == 'Lion':
            try:
                # Lion optimizer is available in newer PyTorch versions
                optimizer = optim.Lion(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0001)
            except AttributeError:
                print("Lion optimizer not available in your PyTorch version, falling back to Adam")
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
                
        # scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=epochs_patience, verbose=False, 
            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08) 

        trainable_params = sum(p.numel() for p in model.parameters(recurse=True) if p.requires_grad) /1000000
        print(f'Trainable parameters: {trainable_params}')

        #check if lossType_list is defined
        


        if combined_loss_flag:

            # this is done to keep if SMP condition in utils.py for losses (lazy coding)
            lossType_temp = lossType
            lossType = lossType_list[0]
    

        # Training 
        model, history = train(  
            out_channels,
            model,
            criterion,
            optimizer,
            lossType, 
            scheduler,
            train_dl, 
            val_dl,
            test_dl,
            Test_Mask,
            seg_threshold,
            checkpoint_name,
            train_on_gpu,
            history=history,
            max_epochs_stop=max_epochs_stop,
            start_epoch = start_epoch,
            n_epochs=n_epochs,
            print_every=1,
            disregard_background=disregard_background)

        # # Saving TrainModel
        TrainChPoint = {}
        TrainChPoint['model']=model
        TrainChPoint['history']=history
        torch.save(TrainChPoint, save_file_name)



        # # Training Results
        if Test_Mask:
            # We can inspect the training progress by looking at the `history`. 
            # plot loss
            plt.figure(figsize=(8, 6))
            for c in ['train_loss', 'val_loss', 'test_loss']:
                plt.plot( 
                    history[c], label=c)
            plt.legend()
            plt.xlabel('Epoch') 
            plt.ylabel('Loss')
            plt.savefig(save_path+f'/LossPerEpoch_fold_{fold_idx}.png')
            # plt.show()
            # plot accuracy
            plt.figure(figsize=(8, 6)) 
            for c in ['train_DSC', 'val_DSC', 'test_DSC']:
                plt.plot(
                    100 * history[c], label=c)
            plt.legend()
            plt.xlabel('Epoch') 
            plt.ylabel('DSC')
            plt.savefig(save_path+f'/DSCPerEpoch_fold_{fold_idx}.png')
            # plt.show()
        else:
            # We can inspect the training progress by looking at the `history`. 
            # plot loss
            plt.figure(figsize=(8, 6))
            for c in ['train_loss', 'val_loss']:
                plt.plot( 
                    history[c], label=c)
            plt.legend()
            plt.xlabel('Epoch') 
            plt.ylabel('Loss')
            plt.savefig(save_path+f'/LossPerEpoch_fold_{fold_idx}.png')
            # plt.show()
            # plot accuracy
            plt.figure(figsize=(8, 6)) 
            for c in ['train_DSC', 'val_DSC']:
                plt.plot(
                    100 * history[c], label=c)
            plt.legend()
            plt.xlabel('Epoch') 
            plt.ylabel('DSC')
            plt.savefig(save_path+f'/DSCPerEpoch_fold_{fold_idx}.png')
            # plt.show()
 
        # release memeory (delete variables)
        del  optimizer, TrainChPoint, scheduler 
        del  train_ds, train_dl, val_ds, val_dl
        torch.cuda.empty_cache()

        # conversion factor to convert class values from np.linspace(0,255,out_channels) to np.linspace(0,out_channels-1,out_channels)
        if out_channels==1:
            conv_fact = 255.0/1.0  
        else:
            conv_fact = 255.0/(out_channels-1)

        # Set to evaluation mode
        model.eval() 
        if Test_Mask: 
            test_acc = 0.0 
            test_loss = 0.0
            test_IoU = 0.0 
            test_DSC = 0.0
            for data, target, im_path in test_dl:
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
                    # # compute IoU
                    # IoU = torch.zeros((out_channels-1,1))
                    # for clas_i in range(1,out_channels):
                    #     IoU[clas_i-1,0]  = compute_IoU((target_temp[:,clas_i,:,:]),(output_temp[:,clas_i,:,:]),seg_threshold)
                    # test_IoU += IoU * data.size(0)  
                    # # compute DSC
                    # DSC = torch.zeros((out_channels-1,1))
                    # for clas_i in range(1,out_channels):
                    #     DSC[clas_i-1,0]  = compute_DSC((target_temp[:,clas_i,:,:]),(output_temp[:,clas_i,:,:]),seg_threshold)
                    # test_DSC += DSC * data.size(0)  
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
            for data, im_path in test_dl: 
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
                elif 'SMP' in lossType:
                    loss = criterion(output, target.squeeze(1))
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
        del model, criterion, history, test_ds, test_dl
        del data, output
        del tensor_image, image_name, filename
        torch.cuda.empty_cache()  

        print(f'completed fold {fold_idx}')

        # store lossType back for the new fold
        if combined_loss_flag:
            lossType = lossType_temp

    print('#############################################################') 

    os.remove(checkpoint_name)
    print("Checkpoint File Removed!")

    if Test_Mask:  
        # # Saving Test Results
        TestChPoint = {}
        TestChPoint['test_history'] = test_history
        columns_names = ['loss', 'Accuracy', 'IoU','DSC']
        if disregard_background:
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