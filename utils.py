# PyTorch
from torchvision import transforms
import cv2
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
from PIL import Image
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
import segmentation_models_pytorch as smp
from os import path 
from unet import UNet
from modified_unet import M_UNet 
from keras_unet import K_UNet 
from importlib import import_module 
from tqdm import tqdm
import tifffile


class SegData(Dataset):
    
    def __init__(self, root_dir, images_path , masks_path, img_names, h, w, mean, std, in_channels, out_channels, return_path=False, ONN=True):
        self.root_dir = root_dir
        self.images_path = images_path
        self.masks_path = masks_path 
        self.img_names = img_names
        self.h = h # image height 
        self.w = w # image width
        self.mean = mean
        self.std = std 
        self.ONN = ONN
        if self.ONN:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std) 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_path = return_path
        self.conv_fact = 1 
        # conversion factor to convert class values from np.linspace(0,255,out_channels) to np.linspace(0,out_channels-1,out_channels)
        if self.out_channels > 1:
            self.conv_fact = 255

        if self.ONN:
            if self.in_channels==1:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),   
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    ]) 
            elif self.in_channels==3:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    ])
            else:
                # New transform for n > 3 channels
                self.img_transforms = transforms.Compose([
        
                    transforms.ToTensor(),
                    transforms.Resize((self.h, self.w))
            ])
        else:
            if self.in_channels==1:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),   
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.mean, std=self.std) 
                    ]) 
            elif self.in_channels==3:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.mean, std=self.std)  
                    ]) 
            else:
                # New transform for n > 3 channels
                self.img_transforms = transforms.Compose([

                    transforms.ToTensor(),
                    transforms.Resize((self.h, self.w)),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ])
        if self.out_channels==1:
            self.mask_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1), 
                transforms.Resize((self.h,self.w), interpolation = transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),  
                ]) 
        else:
            self.mask_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.h,self.w), interpolation = transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),  
                ]) 
    
    def __len__(self): 
        return len(self.img_names)
    
    def __getitem__(self, index):

        # Get the base filename (without extension)
        img_base_name = os.path.splitext(self.img_names[index])[0]
        
        # Create paths with proper extensions
        image_path = os.path.join(self.root_dir, self.images_path, self.img_names[index])
        mask_path = os.path.join(self.root_dir, self.masks_path, img_base_name + '.png') #makes sure mask is always loaded in png
        
        # Handle image loading based on channel count
        if self.in_channels <= 3:
            # Original approach for 1-3 channels
            image = np.array(Image.open(image_path))
            if image.dtype != np.uint8:
                # High bit-depth PNG (uint16 or int32 depending on Pillow version)
                # stored = HU + 1024, range 0-4095 → normalize to uint8 [0,255]
                image = np.clip(image.astype(np.float32) / 4095.0 * 255.0, 0, 255).astype(np.uint8)
        else:
            # For n > 3 channels, use tifffile
            import tifffile
            print(image_path)
            image = tifffile.imread(image_path)
            image = image.astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))

        mask  = Image.open(mask_path)
        mask = np.array(mask)
        
        # apply transformation  
        image = self.img_transforms(image)
        if self.ONN:
            image = 2.0*image -1  
        # image = 2.0*image -1 
        # image = image-self.mean / self.std
        mask = self.conv_fact*self.mask_transforms(mask)
        mask = mask.long() 
        # mask= 1*(mask>=0.5)  
        if self.return_path: 
            return image, mask, self.img_names[index]
        else:
            return image, mask


class TestData(Dataset): 
    
    def __init__(self, root_dir, images_path, img_names, h, w, mean, std, in_channels, return_path=False, ONN=True):
        self.root_dir = root_dir
        self.images_path = images_path
        self.img_names = img_names
        self.h = h # image height 
        self.w = w # image width
        self.mean = mean
        self.std = std 
        self.ONN = ONN
        if self.ONN:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std) 
        self.in_channels = in_channels
        self.return_path = return_path 
        if self.ONN:
            if self.in_channels==1:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),   
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    ]) 
            elif self.in_channels==3:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    ])
            else:
                # New transform for n > 3 channels
                self.img_transforms = transforms.Compose([
        
                    transforms.ToTensor(),
                    transforms.Resize((self.h, self.w))
            ])
        else:
            if self.in_channels==1:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),   
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.mean, std=self.std) 
                    ]) 
            elif self.in_channels==3:
                self.img_transforms = transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((self.h,self.w)),  
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.mean, std=self.std)  
                    ]) 
            else:
                # New transform for n > 3 channels
                self.img_transforms = transforms.Compose([

                    transforms.ToTensor(),
                    transforms.Resize((self.h, self.w)),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ])
        if self.out_channels==1:
            self.mask_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1), 
                transforms.Resize((self.h,self.w)),
                transforms.ToTensor(),  
                ]) 
        else:
            self.mask_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.h,self.w), interpolation = transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),  
                ]) 

    def __len__(self):
        return len(self.img_names)
      
    def __getitem__(self, index):
        # read image
        #image = io.imread(os.path.join(self.root_dir, self.images_path+'/'+self.img_names[index]))
        image_path = os.path.join(self.root_dir, self.images_path+'/'+self.img_names[index])

        if self.in_channels <= 3:
            # Original approach for 1-3 channels
            image = np.array(Image.open(image_path))
            if image.dtype != np.uint8:
                # High bit-depth PNG (uint16 or int32 depending on Pillow version)
                # stored = HU + 1024, range 0-4095 → normalize to uint8 [0,255]
                image = np.clip(image.astype(np.float32) / 4095.0 * 255.0, 0, 255).astype(np.uint8)
        else:
            # For n > 3 channels, use tifffile
            import tifffile
            image = tifffile.imread(image_path)
            image = image.astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
        # apply transformation
        image = self.img_transforms(image)
        if self.ONN:
            image = 2.0*image -1  
        if self.return_path:
            return image, self.img_names[index] 
        else: 
            return image


# import torch.nn.functional as F
# ohe = F.one_hot(labels, num_classes)

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    if labels.is_cuda:
        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    else:
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    assert labels.is_cuda == target.is_cuda, 'target & labels disagree on CUDA status'
    return target

def IoU_training(target,output,threshold):
    smooth = 1.0
    output = 1*(output>=threshold)  # if output[i] >=0.5 then set to 1 else 0
    target = 1*(target>=threshold)  # if arget[i] >=0.5 then set to 1 else 0
    TP = torch.sum((output==1) & (target==1)).double()
    FP = torch.sum((output==1) & (target==0)).double() 
    FN = torch.sum((output==0) & (target==1)).double() 
    IoU = (TP+smooth)/(TP+FP+FN+smooth) 
    return IoU.item()

def DSC_training(target,output,threshold):
    smooth = 1.0
    output = 1*(output>=threshold)  # if output[i] >=0.5 then set to 1 else 0
    target = 1*(target>=threshold)  # if arget[i] >=0.5 then set to 1 else 0
    output = output.reshape((output.size(0),-1))
    target = target.reshape((target.size(0),-1))
    intersection = (output * target).sum(1)
    DSC = (2. * intersection + smooth) / (output.sum(1) + target.sum(1) + smooth)
    return DSC.mean().item()


def compute_IoU(target,output,threshold):
    IoU = 0.0
    smooth = 1
    for i in range(target.shape[0]):
        temp_o = 1*(output[i]>=threshold)  # if output[i] >=0.5 then set to 1 else 0
        temp_g = 1*(target[i]>=threshold)  # if arget[i] >=0.5 then set to 1 else 0
        TP = torch.sum((temp_o==1) & (temp_g==1)).double()
        FP = torch.sum((temp_o==1) & (temp_g==0)).double() 
        FN = torch.sum((temp_o==0) & (temp_g==1)).double() 
        temp_IoU = (TP+smooth)/(TP+FP+FN+smooth) 
        # temp_IoU = (TP)/(TP+FP+FN+1e-9)
        IoU += temp_IoU 
    return IoU.item()/target.shape[0] 

def compute_DSC(target,output,threshold):
    DSC = 0.0   
    smooth = 1
    for i in range(target.shape[0]):
        temp_o = 1*(output[i]>=threshold)  # if output[i] >=0.5 then set to 1 else 0
        temp_g = 1*(target[i]>=threshold)  # if arget[i] >=0.5 then set to 1 else 0
        TP = torch.sum((temp_o==1) & (temp_g==1)).double()
        FP = torch.sum((temp_o==1) & (temp_g==0)).double() 
        FN = torch.sum((temp_o==0) & (temp_g==1)).double() 
        temp_DSC = (2*TP+smooth)/(2*TP+FP+FN+smooth)  
        # temp_DSC = (2*TP)/(2*TP+FP+FN+1e-9) 
        DSC += temp_DSC 
    return DSC.item()/target.shape[0]


# def Multi_Class_DSC(target,output,C):
#         """
#         Computes a Dice  from 2D input of class scores and a target of integer labels.

#         Parameters
#         ----------
#         input : 
#             size B x C x H x W representing class scores.
#         target : 
#             integer label representation of the ground truth, same size as the input.
#             size B x 1 x H x W representing class scores, where each value, 0 < label_i < C-1

#         Returns
#         -------
#         dice_total : float.
#             total dice 
#         """
#         smooth  = 1

#         target = F.one_hot(target, C)
#         target = target.permute(0,1,3,2).permute(0,2,1,3)
        
#         assert output.size() == target.size(), "Input sizes must be equal."
#         assert output.dim() == 4, "Input must be a 4D Tensor."

#         probs=F.softmax(output, dim=1)
#         num=probs*target#b,c,h,w--p*g
#         num=torch.sum(num,dim=3)#b,c,h
#         num=torch.sum(num,dim=2)

#         # den1=probs*probs#--p^2
#         den1=probs 
#         den1=torch.sum(den1,dim=3)#b,c,h
#         den1=torch.sum(den1,dim=2)


#         # den2=target*target#--g^2
#         den2=target 
#         den2=torch.sum(den2,dim=3)#b,c,h
#         den2=torch.sum(den2,dim=2)#b,c

#         # dice=2*(num/(den1+den2)) 
#         # dice=2*((num+self.smooth)/(den1+den2+self.smooth))
#         dice=(2.0*num+smooth)/(den1+den2+smooth) 
#         dice_eso=dice[:,1:]#we ignore backgrounf dice val, and take the foreground 
#         # dice_eso=dice  
#         temp = 1*torch.isnan(dice_eso) 
#         temp = (temp == 1).nonzero() 
#         if temp.numel(): 
#             # print('hello') 
#             dice_eso[temp]=1.0  

#         dice_total = 1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
#         # dice_total = torch.sum(0.25*dice[:,0:])/dice.size(0)#divide by batch_sz
#         # dice_total += torch.sum(0.75*dice[:,1:])/dice.size(0)#divide by batch_sz 
#         # dice_total = dice_total/2.0

#         return dice_total.item()




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

    output = F.softmax(output, dim=1)
    _, output = torch.max(output, dim=1)
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes = C)
    DSC = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
    if disregard_background:
        DSC = DSC[:, 1:]
    #return DSC.mean().item()
    return DSC.mean().item()

# def Multi_Class_DSC_test(target,output,C,threshold):
#     """
#     Computes a Dice  from 2D input of class scores and a target of integer labels.

#     Parameters
#     ----------
#     input : 
#         size B x C x H x W representing class scores.
#     target : 
#         integer label representation of the ground truth, same size as the input.
#         size B x 1 x H x W representing class scores, where each value, 0 < label_i < C-1

#     Returns
#     -------
#     dice_total : float.
#     total dice 
#     """
#     # output = F.softmax(output, dim=1)
#     # target = F.one_hot(target, C)
#     # target = target.permute(0,1,3,2).permute(0,2,1,3)

#     # DSC = torch.zeros((C-1,1))
#     # for clas_i in range(1,C):
#     #     DSC[clas_i-1,0]  = DSC_training((target[:,clas_i,:,:]),(output[:,clas_i,:,:]),threshold)

#     output = F.softmax(output, dim=1)

#     _, output = torch.max(output, dim=1)

    
#     # print('output:', output.sum().item(), 'target:', target.sum().item())
#     # print(output.dtype, target.dtype)
#     # print('Classes:', C)
#     tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes = C)
#     #print('tp, fp, fn, tn:', tp, fp, fn, tn)
#     DSC = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
#     #print('pre:', DSC, end='')
#     DSC = DSC[:, 1:]
#     #print('post:',DSC)
#     #return DSC.mean().item()
#     return DSC.mean().item()


def calc_acc(target,output,threshold):
    target = 1*(target>=threshold)  # if target >=0.5 then set to 1 else 0
    output = 1*(output>=threshold)  # if output >=0.5 then set to 1 else 0
    return torch.mean((target==output).float()) 


def Createlabels(datadir, Seg_state=False):
    categories = []
    n_Class = [] 
    img_names = []
    labels = []
    i = 0
    for d in os.listdir(datadir):
        categories.append(d)
        if Seg_state:
            if i==1:
                break
        temp = os.listdir(datadir + d)
        img_names.extend(temp) 
        n_temp = len(temp)
        if i==0:
            labels = np.zeros((n_temp,1)) 
        else:
            labels = np.concatenate( (labels, i*np.ones((n_temp,1))) )
        i = i+1
        n_Class.append(n_temp)
    if Seg_state:
        return categories, n_Class, img_names 
    else:
        return categories, n_Class, img_names, labels 
    


def train(out_channels,
          model,
          criterion,
          optimizer,
          lossType, 
          scheduler, 
          train_loader,
          valid_loader,
          test_loader,
          Test_Mask,
          threshold,
          save_file_name,
          train_on_gpu,
          history=[],
          max_epochs_stop=5,
          start_epoch=0,
          n_epochs=30,
          print_every=2,
          disregard_background=True):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and DSC
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.inf 
    # train_loss_min = np.inf

    valid_max_acc = 0

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(start_epoch, n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0
        train_DSC = 0
        valid_DSC = 0
        test_DSC = 0
        # Set to training
        model.train()
        start = timer()
        pbar = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch}", leave=False)
        # Training loop
        for ii, (data, target) in enumerate(pbar):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs 
            output = model(data)   
            # print(output.shape)
            # print(output)
            # Loss and backpropagation of gradients

            if lossType == 'CrossEntropy':  
                loss = criterion(output, target.squeeze(1)) 
            elif lossType == 'BCELoss':  
                output = torch.sigmoid(output) 
                loss = criterion(output.squeeze(1), target.float().squeeze(1)) 
            elif lossType == 'DiceLoss':
                output = torch.sigmoid(output)
                loss = criterion(output, target)
            elif lossType == 'CompoundLoss':
                output = torch.sigmoid(output)
                loss = criterion(output, target) 
            elif 'SMP' in lossType:
                loss = criterion(output, target.squeeze(1))

            
            loss.backward()
            # Update the parameters
            optimizer.step() 
            # accumalate loss
            train_loss += loss.item() * data.size(0) 
            # Calculate DSC 
            if out_channels==1:
                DSC = DSC_training(target,output,threshold)   
            else:
                DSC = Multi_Class_DSC(target.squeeze(1),output,out_channels,threshold, disregard_background) 
                #DSC = Multi_Class_DSC(target,output,out_channels,threshold) 
            train_DSC += DSC * data.size(0)
            pbar.set_postfix(loss = loss.item(), DSC = DSC) 
            # release memeory (delete variables)
            del output, data, target 
            del loss, DSC

        # After training loops ends, start validation
        else:
            model.epochs += 1
            # Don't need to keep track of gradients 
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()
                # Validation loop
                ii = 0
                pbar_val = tqdm(valid_loader, ncols=100, desc = 'Validating', leave=False)
                for _, (data, target) in enumerate(pbar_val):
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
                    # Forward pass
                    output = model(data)
                    # print(output.shape)
                    # print(output)
                    # Loss 
                    if lossType == 'CrossEntropy':  
                        loss = criterion(output, target.squeeze(1)) 
                    elif lossType == 'BCELoss':  
                        output = torch.sigmoid(output) 
                        loss = criterion(output.squeeze(1), target.float().squeeze(1)) 
                    elif lossType == 'DiceLoss':
                        output = torch.sigmoid(output)
                        loss = criterion(output, target)
                    elif lossType == 'CompoundLoss':
                        output = torch.sigmoid(output)
                        loss = criterion(output, target)
                    elif 'SMP' in lossType:
                        loss = criterion(output, target.squeeze(1))
                     
                    # Multiply average loss times the number of examples in batch 
                    valid_loss += loss.item() * data.size(0)    
                    # Calculate validation DSC
                    if out_channels==1:
                        DSC = DSC_training(target,output,threshold)   
                    else:
                        DSC = Multi_Class_DSC(target.squeeze(1),output,out_channels,threshold, disregard_background) 
                        #DSC = Multi_Class_DSC(target,output,out_channels,threshold) 
                    # Multiply average DSC times the number of examples
                    valid_DSC += DSC * data.size(0)
                    pbar_val.set_postfix(loss = loss.item(), DSC = DSC)  
                # test loop
                if Test_Mask:
                    pbar_test = tqdm(test_loader, ncols=100, desc = 'Testing', leave=False)
                    for _, (data, target, _) in enumerate(pbar_test):
                        # Tensors to gpu
                        if train_on_gpu:
                            data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
                        # Forward pass
                        output = model(data)
                        
                        # Loss 
                        if lossType == 'CrossEntropy':  
                            loss = criterion(output, target.squeeze(1)) 
                        elif lossType == 'BCELoss':  
                            output = torch.sigmoid(output) 
                            loss = criterion(output.squeeze(1), target.float().squeeze(1)) 
                        elif lossType == 'DiceLoss':
                            output = torch.sigmoid(output)
                            loss = criterion(output, target)
                        elif lossType == 'CompoundLoss':
                            output = torch.sigmoid(output)
                            loss = criterion(output, target)  
                        elif 'SMP' in lossType:
                            loss = criterion(output, target.squeeze(1))
                        # Multiply average loss times the number of examples in batch 
                        test_loss += loss.item() * data.size(0)
                        # Calculate validation DSC
                        if out_channels==1:
                            DSC = DSC_training(target,output,threshold)   
                        else:
                            DSC = Multi_Class_DSC(target.squeeze(1),output,out_channels,threshold, disregard_background)
                            
                            #print(target.size(), output.size())
                            # t_output = F.softmax(output, dim=1)
                            # _, t_output = torch.max(output, dim=1)
                            # t_output = t_output.cpu().numpy()
                            # print(t_output[t_output!=0].sum()) 
                            #print(DSC)
                            #DSC = Multi_Class_DSC(target,output,out_channels,threshold) 
                        # Multiply average DSC times the number of examples
                        test_DSC += DSC * data.size(0)
                        pbar_test.set_postfix(loss = loss.item(), DSC = DSC)   
                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset) 
                valid_loss = valid_loss / len(valid_loader.dataset)
                if Test_Mask:
                    test_loss = test_loss / len(test_loader.dataset)
                scheduler.step(valid_loss) 
                # Calculate average DSC
                train_DSC = train_DSC / len(train_loader.dataset)
                valid_DSC = valid_DSC / len(valid_loader.dataset)
                if Test_Mask:
                    # print("TEST DSC TEST", test_DSC, len(test_loader.dataset))
                    test_DSC = test_DSC / len(test_loader.dataset) 
                    history.append([train_loss, valid_loss, test_loss, train_DSC, valid_DSC, test_DSC]) 
                else:
                    history.append([train_loss, valid_loss, train_DSC, valid_DSC])
                # Print training and validation results
                if Test_Mask:
                    if (epoch + 1) % print_every == 0:
                        print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f} \t Test Loss: {test_loss:.4f}')
                        print(f'\t\tTraining DSC: {100 * train_DSC:.2f}% \t Validation DSC: {100 * valid_DSC:.2f}% \t Test DSC: {100 * test_DSC:.2f}%')
                    # release memeory (delete variables)
                    del output, data, target
                    del loss, DSC
                else:
                    if (epoch + 1) % print_every == 0:
                        print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                        print(f'\t\tTraining DSC: {100 * train_DSC:.2f}%\t Validation DSC: {100 * valid_DSC:.2f}%')
                    # release memeory (delete variables)
                    del output, data, target
                    del loss, DSC

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # # Saving TrainModel
                    TrainChPoint = {}
                    TrainChPoint['model']=model
                    TrainChPoint['history']=history
                    torch.save(TrainChPoint, save_file_name)
                    # Save model
                    # torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_DSC = valid_DSC
                    best_epoch = epoch
                # # Save the model if validation loss decreases
                # if train_loss < train_loss_min: 
                # # if 1:
                #     # Save model
                #     torch.save(model.state_dict(), save_file_name)
                #     # Track improvement
                #     epochs_no_improve = 0
                #     valid_loss_min = train_loss
                #     valid_best_DSC = train_DSC
                #     best_epoch = epoch
                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.4f} and DSC: {100 * valid_best_DSC:.2f}%')
                        total_time = timer() - overall_start
                        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')
                        # Load the best state dict
                        checkpoint = torch.load(save_file_name)
                        model = checkpoint['model']
                        # Attach the optimizer
                        model.optimizer = optimizer
                        if Test_Mask:
                            # Format history
                            history = pd.DataFrame(history,columns=['train_loss', 'val_loss', 'test_loss', 'train_DSC','val_DSC', 'test_DSC'])
                            return model, history
                        else:
                            # Format history
                            history = pd.DataFrame(history,columns=['train_loss', 'val_loss', 'train_DSC','val_DSC'])
                            return model, history

    # Load the best state dict
    checkpoint = torch.load(save_file_name)
    model = checkpoint['model']
  

    # Attach the optimizer 
    model.optimizer = optimizer
    # Record overall time and print out stats 
    total_time = timer() - overall_start 
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.4f} and DSC: {100 * valid_DSC:.2f}%')
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.')
    if Test_Mask:
        # Format history
        history = pd.DataFrame(history,columns=['train_loss', 'val_loss', 'test_loss', 'train_DSC','val_DSC', 'test_DSC'])
    else:
        # Format history
        history = pd.DataFrame(history,columns=['train_loss', 'val_loss', 'train_DSC', 'val_DSC'])
    return model, history






