import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class WeightedFocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, alpha=.25, gamma=2):
#         super(WeightedFocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = torch.tensor(gamma).cuda() 
#         self.criterion = nn.NLLLoss(reduction='none') 

#     def forward(self, inputs, targets):
#         # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         BCE_loss =self.criterion(inputs, targets) 
#         BCE_loss = BCE_loss.data.view(-1) 
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss) 
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         F_loss.requires_grad = True
#         return F_loss.mean()


class WeightedFocalLoss(nn.Module):
    '''
    Credit:
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''
    def __init__(self, gamma=2., alpha=None, size_average=True):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,)): # long is removed in Python3
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
            input = input.transpose(1,2) # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2)) # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        # logpt = F.log_softmax(input)
        logpt = input.gather(1,target)
        logpt = logpt.view(-1) 
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

pred = torch.rand(4,2,256,256).cuda()
log_softmax = nn.LogSoftmax(dim=1) 
pred = log_softmax(pred) 
target = torch.ones(4,256,256,dtype=torch.long).cuda()
criterion = WeightedFocalLoss(gamma=2., alpha=.25) 
loss = criterion(pred,target)
print(loss)
print('done')






