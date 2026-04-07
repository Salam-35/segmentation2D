import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F 
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
 

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size() 
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

class CompoundLoss(nn.Module):

    def __init__(self):
        super(CompoundLoss, self).__init__()
        self.smooth = 1.0
 

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size() 
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )

        bce = nn.BCELoss()
        bce_loss = bce.forward(y_pred.float(), y_true.float())

        return (1. - dsc) + bce_loss



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




class dice_loss_integer(nn.Module):
    
    def __init__(self):
        super(dice_loss_integer, self).__init__()
        self.smooth = 1e-6

    def forward(self, input_, target, C=2): 
        """
        Computes a Dice loss from 2D input of class scores and a target of integer labels.

        Parameters
        ----------
        input : torch.autograd.Variable
            size B x C x H x W representing class scores.
        target : torch.autograd.Variable
            integer label representation of the ground truth, same size as the input.

        Returns
        -------
        dice_total : float.
            total dice loss.
        """
        target = make_one_hot(target, C=C)
        # subindex target without the ignore label
        # target = target[:,:ignore_label,...]

        assert input_.size() == target.size(), "Input sizes must be equal."
        assert input_.dim() == 4, "Input must be a 4D Tensor."

        probs=F.softmax(input_, dim=1)
        num=probs*target#b,c,h,w--p*g
        num=torch.sum(num,dim=3)#b,c,h
        num=torch.sum(num,dim=2)


        # den1=probs*probs#--p^2
        den1=probs 
        den1=torch.sum(den1,dim=3)#b,c,h
        den1=torch.sum(den1,dim=2)


        # den2=target*target#--g^2
        den2=target 
        den2=torch.sum(den2,dim=3)#b,c,h
        den2=torch.sum(den2,dim=2)#b,c


        # dice=2*(num/(den1+den2)) 
        # dice=2*((num+self.smooth)/(den1+den2+self.smooth))
        dice=(2.0*num+self.smooth)/(den1+den2+self.smooth) 
        dice_eso=dice[:,1:]#we ignore backgrounf dice val, and take the foreground 
        # dice_eso=dice  
        temp = 1*torch.isnan(dice_eso) 
        temp = (temp == 1).nonzero() 
        if temp.numel(): 
            # print('hello') 
            dice_eso[temp]=1.0  

        dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
        # dice_total = torch.sum(0.25*dice[:,0:])/dice.size(0)#divide by batch_sz
        # dice_total += torch.sum(0.75*dice[:,1:])/dice.size(0)#divide by batch_sz 
        # dice_total = dice_total/2.0

        # return -1*dice_total 
        return 1.0 - dice_total



# # def dice_loss_integer(input_, target, ignore_label=3, C=2):
# def dice_loss_integer(input_, target, C=2):
#     """
#     Computes a Dice loss from 2D input of class scores and a target of integer labels.

#     Parameters
#     ----------
#     input : torch.autograd.Variable
#         size B x C x H x W representing class scores.
#     target : torch.autograd.Variable
#         integer label representation of the ground truth, same size as the input.

#     Returns
#     -------
#     dice_total : float.
#         total dice loss.
#     """
#     target = make_one_hot(target, C=C)
#     # subindex target without the ignore label
#     # target = target[:,:ignore_label,...]

#     assert input_.size() == target.size(), "Input sizes must be equal."
#     assert input_.dim() == 4, "Input must be a 4D Tensor."

#     probs=F.softmax(input_, dim=1)
#     num=probs*target#b,c,h,w--p*g
#     num=torch.sum(num,dim=3)#b,c,h
#     num=torch.sum(num,dim=2)


#     den1=probs*probs#--p^2
#     den1=torch.sum(den1,dim=3)#b,c,h
#     den1=torch.sum(den1,dim=2)


#     den2=target*target#--g^2
#     den2=torch.sum(den2,dim=3)#b,c,h
#     den2=torch.sum(den2,dim=2)#b,c


#     dice=2*(num/(den1+den2))
#     dice_eso=dice[:,1:]#we ignore backgrounf dice val, and take the foreground

#     # dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
#     dice_total=torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

#     # return dice_total 
#     return 1.0 - dice_total




# import os 
# import shutil
# import numpy as np

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# import sys
# import model as mdls
# import torchvision.utils as vutils
# import utils
# from skimage.io import imsave

# # Inspired by W. Kentaro (@wkentaro)
# def crossentropy2d(pred, target, weight=None, ignore_index=2, size_average=True):
#     '''
#     Parameters
#     -----------
#     pred : autograd.Variable. (N, C, H, W)
#         where C is number of classes.
#     target : (N, H, W), where all values 0 <= target[i] <= C-1.

#     Returns
#     -------
#     loss : Tensor.
#     '''
#     # pred dims (N, C, H, W)
#     n, c, h, w = pred.size()
#     # log_p : log_probabilities (N, C, H, W)
#     log_p = F.log_softmax(pred)
#     # Linearize log_p
#     # log_p : (N, C, H, W) --> (N*H*W, C)
#     # move dim C over twice (N, C, H, W) > (N, H, C, W) > (N, H, W, C)
#     log_p = log_p.transpose(1,2).transpose(2,3).contiguous()
#     # (N, H, W, C) --> (N*H*W, C)
#     log_p = log_p.view(-1, c)

#     # Reshape target to a (N*H*W,), where each values 0 <= i <= C-1
#     target = target.view(-1)
#     loss = F.nll_loss(log_p, target, weight=weight,
#                         size_average=True, ignore_index=ignore_index)

#     return loss

# # Dice loss from Roger Trullo
# #
# class DiceLoss(nn.Module):

#     def __init__(self, ignore_label: int=3, C: int=3) -> None:
#         '''
#         Computes a Dice loss from 2D input of class scores and a target of integer labels.

#         Parameters
#         ----------
#         ignore_label : integer.
#             Must be final label in the sequence (TODO, generalize).
#         C : integer.
#             number of classes (including an ignored label if present!)

#         Notes
#         -----
#         Credit to Roger Trullo
#         https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
#         '''
#         super(DiceLoss, self).__init__()
#         self.ignore_label = ignore_label
#         self.C = C
#         return

#     def forward(self, input_, target) -> float:
#         target = utils.make_one_hot(target, C=self.C)
#         # subindex target without the ignore label
#         target = target[:,:self.ignore_label,...]

#         assert input_.size() == target.size(), "Input sizes must be equal."
#         assert input_.dim() == 4, "Input must be a 4D Tensor."

#         probs=F.softmax(input_, dim=1)
#         num=probs*target#b,c,h,w--p*g
#         num=torch.sum(num,dim=3)#b,c,h
#         num=torch.sum(num,dim=2)

#         den1=probs*probs#--p^2
#         den1=torch.sum(den1,dim=3)#b,c,h
#         den1=torch.sum(den1,dim=2)

#         den2=target*target#--g^2
#         den2=torch.sum(den2,dim=3)#b,c,h
#         den2=torch.sum(den2,dim=2)#b,c

#         dice=2*(num/(den1+den2))
#         dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

#         dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

#         return dice_total

# def dice_loss_integer(input_, target, ignore_label=3, C=3):
#     """
#     Computes a Dice loss from 2D input of class scores and a target of integer labels.

#     Parameters
#     ----------
#     input : torch.autograd.Variable
#         size B x C x H x W representing class scores.
#     target : torch.autograd.Variable
#         integer label representation of the ground truth, same size as the input.

#     Returns
#     -------
#     dice_total : float.
#         total dice loss.
#     """
#     target = utils.make_one_hot(target, C=C)
#     # subindex target without the ignore label
#     target = target[:,:ignore_label,...]

#     assert input_.size() == target.size(), "Input sizes must be equal."
#     assert input_.dim() == 4, "Input must be a 4D Tensor."

#     probs=F.softmax(input_, dim=1)
#     num=probs*target#b,c,h,w--p*g
#     num=torch.sum(num,dim=3)#b,c,h
#     num=torch.sum(num,dim=2)


#     den1=probs*probs#--p^2
#     den1=torch.sum(den1,dim=3)#b,c,h
#     den1=torch.sum(den1,dim=2)


#     den2=target*target#--g^2
#     den2=torch.sum(den2,dim=3)#b,c,h
#     den2=torch.sum(den2,dim=2)#b,c


#     dice=2*(num/(den1+den2))
#     dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

#     dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

#     return dice_total

# from torch.autograd import Variable

# class FocalLoss(nn.Module):
#     '''
#     Credit:
#     https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
#     '''
#     def __init__(self, gamma=2., alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,)): # long is removed in Python3
#             self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list):
#             self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2) # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2)) # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1) 
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()


# def tensor_norm(T):
#     return (T - T.min())/(T-T.min()).max()


"""
CrossentropyND and TopKLoss are from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/ND_Crossentropy.py
"""


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        return super(CrossentropyND, self).forward(inp, target)

class TopKLoss(CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)
        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        return wce_loss(inp, target)

class WeightedCrossEntropyLossV2(torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    Network has to have NO LINEARITY!
    copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L121
    """

    def forward(self, net_output, gt):
        # compute weight
        # shp_x = net_output.shape
        # shp_y = gt.shape
        # print(shp_x, shp_y)
        # with torch.no_grad():
        #     if len(shp_x) != len(shp_y):
        #         gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        #     if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
        #         # if this is the case then gt is probably already a one hot encoding
        #         y_onehot = gt
        #     else:
        #         gt = gt.long()
        #         y_onehot = torch.zeros(shp_x)
        #         if net_output.device.type == "cuda":
        #             y_onehot = y_onehot.cuda(net_output.device.index)
        #         y_onehot.scatter_(1, gt, 1)
        # y_onehot = y_onehot.transpose(0,1).contiguous()
        # class_weights = (torch.einsum("cbxyz->c", y_onehot).type(torch.float32) + 1e-10)/torch.numel(y_onehot)
        # print('class_weights', class_weights)
        # class_weights = class_weights.view(-1)
        class_weights = torch.cuda.FloatTensor([0.2,0.8])
        gt = gt.long()
        num_classes = net_output.size()[1]
        # class_weights = self._class_weights(inp)

        i0 = 1
        i1 = 2

        while i1 < len(net_output.shape): # this is ugly but torch only allows to transpose two axes at once
            net_output = net_output.transpose(i0, i1)
            i0 += 1
            i1 += 1

        net_output = net_output.contiguous()
        net_output = net_output.view(-1, num_classes) #shape=(vox_num, class_num)

        gt = gt.view(-1,)
        # print('*'*20)
        return F.cross_entropy(net_output, gt) # , weight=class_weights

    # @staticmethod
    # def _class_weights(input):
    #     # normalize the input first
    #     input = F.softmax(input, _stacklevel=5)
    #     flattened = flatten(input)
    #     nominator = (1. - flattened).sum(-1)
    #     denominator = flattened.sum(-1)
    #     class_weights = Variable(nominator / denominator, requires_grad=False)
    #     return class_weights

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    transposed = transposed.contiguous()
    return transposed.view(C, -1)

def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    GT = np.squeeze(GT)
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt)-pos_edt)*posmask 
        neg_edt =  distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt)-neg_edt)*negmask        
        res[i] = pos_edt/np.max(pos_edt) + neg_edt/np.max(neg_edt)
    return res

class DisPenalizedCE(torch.nn.Module):
    """
    Only for binary 3D segmentation

    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        # print(inp.shape, target.shape) # (batch, 2, xyz), (batch, 2, xyz)
        # compute distance map of ground truth
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(target.cpu().numpy()>0.5) + 1.0
        
        dist = torch.from_numpy(dist)
        if dist.device != inp.device:
            dist = dist.to(inp.device).type(torch.float32)
        dist = dist.view(-1,)

        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        log_sm = torch.nn.LogSoftmax(dim=1)
        inp_logs = log_sm(inp)

        target = target.view(-1,)
        # loss = nll_loss(inp_logs, target)
        loss = -inp_logs[range(target.shape[0]), target]
        # print(loss.type(), dist.type())
        weighted_loss = loss*dist

        return loss.mean()


def nll_loss(input, target):
    """
    customized nll loss
    source: https://medium.com/@zhang_yang/understanding-cross-entropy-
    implementation-in-pytorch-softmax-log-softmax-nll-cross-entropy-416a2b200e34
    """
    loss = -input[range(target.shape[0]), target]
    return loss.mean()
