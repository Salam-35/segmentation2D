from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from SelfONN import SuperONN1d, SuperONN2d, SelfONN1d, SelfONN2d, TransposeSelfONN2d

class Self_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, q_order=1, up_q_order=1, out_q_order=1):  
        super(Self_UNet, self).__init__()

        features = init_features

        self.encoder1 = Self_UNet._block(in_channels, features, q_order=q_order, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Self_UNet._block(features, features * 2, q_order=q_order, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Self_UNet._block(features * 2, features * 4, q_order=q_order, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Self_UNet._block(features * 4, features * 8, q_order=q_order, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Self_UNet._block(features * 8, features * 16, q_order=q_order, name="bottleneck")

        self.upconv4 = TransposeSelfONN2d(features * 16, features * 8,kernel_size=2,stride=2,q=up_q_order)
        self.decoder4 = Self_UNet._block((features * 8) * 2 , features * 8, q_order=q_order, name="dec4")
        self.upconv3 = TransposeSelfONN2d(features * 8, features * 4,kernel_size=2,stride=2,q=up_q_order)
        self.decoder3 = Self_UNet._block((features * 4) * 2 , features * 4, q_order=q_order, name="dec3")
        self.upconv2 = TransposeSelfONN2d(features * 4, features * 2,kernel_size=2,stride=2,q=up_q_order)
        self.decoder2 = Self_UNet._block((features * 2) * 2 , features * 2, q_order=q_order, name="dec2")
        self.upconv1 = TransposeSelfONN2d(features * 2, features,kernel_size=2,stride=2,q=up_q_order)  
        self.decoder1 = Self_UNet._block(features * 2, features, q_order=q_order, name="dec1") 

        self.conv = SelfONN2d(in_channels=features, out_channels=out_channels, kernel_size=1, stride=1, padding=0, q=out_q_order, dropout=0) 


    def forward(self, x): 
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1) 

    @staticmethod
    def _block(in_channels, features, q_order, name):
        return nn.Sequential(
            OrderedDict(
                [ 
                    (name + "SelfConv1",SelfONN2d(in_channels=in_channels,out_channels=features,kernel_size=3, stride=1, padding=1, q=q_order, dropout=0),),
                    (name + "norm1", nn.BatchNorm2d(num_features=features, momentum=0.1)),
                    (name + "tanh", nn.Tanh()), 
                    (name + "SelfConv2",SelfONN2d(in_channels=features,out_channels=features,kernel_size=3, stride=1, padding=1, q=q_order, dropout=0),),
                    (name + "norm2", nn.BatchNorm2d(num_features=features, momentum=0.1)),
                    (name + "tanh", nn.Tanh()),
                ]
            )
        )



class SelfUNet_compact(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32, q_order=1, up_q_order=1, out_q_order=1):  
        super(SelfUNet_compact, self).__init__()

        features = init_features

        self.encoder1 = SelfUNet_compact._block(in_channels, features, q_order=q_order, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.encoder3 = SelfUNet_compact._block(features, features * 4, q_order=q_order, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4) 

        self.bottleneck = SelfUNet_compact._block(features * 4, features * 16, q_order=q_order, name="bottleneck")

        self.upconv3 = TransposeSelfONN2d(features * 16, features * 4,kernel_size=4,stride=4,q=up_q_order) 
        self.decoder3 = SelfUNet_compact._block((features * 4) * 2 , features * 4, q_order=q_order, name="dec3")
        self.upconv1 = TransposeSelfONN2d(features * 4, features,kernel_size=4,stride=4,q=up_q_order)   
        self.decoder1 = SelfUNet_compact._block(features * 2, features, q_order=q_order, name="dec1") 
        self.conv = SelfONN2d(in_channels=features, out_channels=out_channels, kernel_size=1, stride=1, padding=0, q=out_q_order, dropout=0)  

    def forward(self, x): 
        enc1 = self.encoder1(x)
        enc3 = self.encoder3(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec1 = self.upconv1(dec3)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1) 

    @staticmethod
    def _block(in_channels, features, q_order, name):
        return nn.Sequential(
            OrderedDict(
                [ 
                    (name + "SelfConv1",SelfONN2d(in_channels=in_channels,out_channels=features,kernel_size=3, stride=1, padding=1, q=q_order, dropout=0),),
                    (name + "norm1", nn.BatchNorm2d(num_features=features, momentum=0.1)),
                    (name + "tanh", nn.Tanh()), 
                    (name + "SelfConv2",SelfONN2d(in_channels=features,out_channels=features,kernel_size=3, stride=1, padding=1, q=q_order, dropout=0),),
                    (name + "norm2", nn.BatchNorm2d(num_features=features, momentum=0.1)),
                    (name + "tanh", nn.Tanh()),
                ]
            )
        )




class SelfUNet_super_compact(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32, q_order=1, up_q_order=1, out_q_order=1):  
        super(SelfUNet_super_compact, self).__init__()

        features = init_features
        self.encoder1 = SelfUNet_super_compact._block(in_channels, features, q_order=q_order, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.encoder3 = SelfUNet_super_compact._block(features, features * 4, q_order=q_order, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4) 

        self.bottleneck = SelfUNet_super_compact._block(features * 4, features * 16, q_order=q_order, name="bottleneck")

        self.upconv3 = TransposeSelfONN2d(features * 16, features * 4,kernel_size=4,stride=4,q=up_q_order) 
        self.decoder3 = SelfUNet_super_compact._block((features * 4) * 2 , features * 4, q_order=q_order, name="dec3")
        self.upconv1 = TransposeSelfONN2d(features * 4, features,kernel_size=4,stride=4,q=up_q_order)   
        self.decoder1 = SelfUNet_super_compact._block(features * 2, features, q_order=q_order, name="dec1") 
        self.conv = SelfONN2d(in_channels=features, out_channels=out_channels, kernel_size=1, stride=1, padding=0, q=out_q_order, dropout=0)  

    def forward(self, x): 
        enc1 = self.encoder1(x)
        enc3 = self.encoder3(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec1 = self.upconv1(dec3)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1) 

    @staticmethod
    def _block(in_channels, features, q_order, name):
        return nn.Sequential(
            OrderedDict(
                [ 
                    (name + "SelfConv1",SelfONN2d(in_channels=in_channels,out_channels=features,kernel_size=3, stride=1, padding=1, q=q_order, dropout=0),),
                    (name + "norm1", nn.BatchNorm2d(num_features=features, momentum=0.1)),
                    (name + "tanh", nn.Tanh()), 
                ]
            )
        )



class SuperUNet_super_compact(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32, q_order=1, up_q_order=1, out_q_order=1):  
        super(SuperUNet_super_compact, self).__init__()

        features = init_features
        self.encoder1 = SuperUNet_super_compact._block(in_channels, features, q_order=q_order, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.encoder3 = SuperUNet_super_compact._block(features, features * 4, q_order=q_order, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4) 

        self.bottleneck = SuperUNet_super_compact._block(features * 4, features * 16, q_order=q_order, name="bottleneck")

        self.upconv3 = TransposeSelfONN2d(features * 16, features * 4,kernel_size=4,stride=4,q=up_q_order) 
        self.decoder3 = SuperUNet_super_compact._block((features * 4) * 2 , features * 4, q_order=q_order, name="dec3")
        self.upconv1 = TransposeSelfONN2d(features * 4, features,kernel_size=4,stride=4,q=up_q_order)   
        self.decoder1 = SuperUNet_super_compact._block(features * 2, features, q_order=q_order, name="dec1") 
        self.conv = SelfONN2d(in_channels=features, out_channels=out_channels, kernel_size=1, stride=1, padding=0, q=out_q_order, dropout=0)  

    def forward(self, x): 
        enc1 = self.encoder1(x)
        enc3 = self.encoder3(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec1 = self.upconv1(dec3)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1) 

    @staticmethod
    def _block(in_channels, features, q_order, name):
        return nn.Sequential(
            OrderedDict(
                [ 
                    (name + "SelfConv1",SuperONN2d(in_channels=in_channels,out_channels=features,kernel_size=3,q=q_order,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=True,dropout=None)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features, momentum=0.1)),
                    (name + "tanh", nn.Tanh()), 
                ]
            )
        )







class Super_FPN(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32, q_order=1, up_q_order=1, out_q_order=1):  
        super(Super_FPN, self).__init__()

        features = init_features

        self.encoder1 = Super_FPN._block(in_channels, features, q_order=q_order, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Super_FPN._block(features, features * 2, q_order=q_order, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Super_FPN._block(features * 2, features * 4, q_order=q_order, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Super_FPN._block(features * 4, features * 8, q_order=q_order, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Super_FPN._block(features * 8, features * 16, q_order=q_order, name="bottleneck")

        self.upconv4 = TransposeSelfONN2d(features * 16, features * 8,kernel_size=2,stride=2,q=up_q_order)
        self.decoder4 = Super_FPN._block((features * 8) * 2 , features * 8, q_order=q_order, name="dec4")
        self.upconv3 = TransposeSelfONN2d(features * 8, features * 4,kernel_size=2,stride=2,q=up_q_order)
        self.decoder3 = Super_FPN._block((features * 4) * 2 , features * 4, q_order=q_order, name="dec3")
        self.upconv2 = TransposeSelfONN2d(features * 4, features * 2,kernel_size=2,stride=2,q=up_q_order)
        self.decoder2 = Super_FPN._block((features * 2) * 2 , features * 2, q_order=q_order, name="dec2")
        self.upconv1 = TransposeSelfONN2d(features * 2, features,kernel_size=2,stride=2,q=up_q_order)  
        self.decoder1 = Super_FPN._block(features * 2, features, q_order=q_order, name="dec1") 
        
        self.cat = FPN_cat()
        self.conv = nn.Sequential(
            SuperONN2d(in_channels=7*features,out_channels=features,kernel_size=3,q=q_order,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=True,dropout=None),
            SelfONN2d(in_channels=features,out_channels=out_channels,kernel_size=1, stride=1, padding=0, q=out_q_order, dropout=0)
        )


    def forward(self, x): 
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        dec1 = self.cat(dec1, dec2, dec3)
        return self.conv(dec1) 

    @staticmethod
    def _block(in_channels, features, q_order, name):
        return nn.Sequential(
            OrderedDict(
                [ 
                    (name + "SelfConv1",SuperONN2d(in_channels=in_channels,out_channels=features,kernel_size=3,q=q_order,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=True,dropout=None)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features, momentum=0.1)),
                    (name + "tanh", nn.Tanh()), 
                ]
            )
        )




class Attention_Super_FPN(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32, q_order=1, up_q_order=1, out_q_order=1):  
        super(Attention_Super_FPN, self).__init__()

        features = init_features

        self.encoder1 = Attention_Super_FPN._block(in_channels, features, q_order=q_order, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Attention_Super_FPN._block(features, features * 2, q_order=q_order, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Attention_Super_FPN._block(features * 2, features * 4, q_order=q_order, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Attention_Super_FPN._block(features * 4, features * 8, q_order=q_order, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Attention_Super_FPN._block(features * 8, features * 16, q_order=q_order, name="bottleneck")

        self.attention4 = AttentionBlock(in_channels_x=features*8, in_channels_g=features*16, int_channels=features*8)
        self.upconv4 = TransposeSelfONN2d(features * 16, features * 8,kernel_size=2,stride=2,q=up_q_order)
        self.decoder4 = Attention_Super_FPN._block((features * 8) * 2 , features * 8, q_order=q_order, name="dec4")

        self.attention3 = AttentionBlock(in_channels_x=features*4, in_channels_g=features*8, int_channels=features*4)
        self.upconv3 = TransposeSelfONN2d(features * 8, features * 4,kernel_size=2,stride=2,q=up_q_order)
        self.decoder3 = Attention_Super_FPN._block((features * 4) * 2 , features * 4, q_order=q_order, name="dec3")

        self.attention2 = AttentionBlock(in_channels_x=features*2, in_channels_g=features*4, int_channels=features*2)
        self.upconv2 = TransposeSelfONN2d(features * 4, features * 2,kernel_size=2,stride=2,q=up_q_order)
        self.decoder2 = Attention_Super_FPN._block((features * 2) * 2 , features * 2, q_order=q_order, name="dec2")

        self.attention1 = AttentionBlock(in_channels_x=features, in_channels_g=features*2, int_channels=features)
        self.upconv1 = TransposeSelfONN2d(features * 2, features,kernel_size=2,stride=2,q=up_q_order)  
        self.decoder1 = Attention_Super_FPN._block(features * 2, features, q_order=q_order, name="dec1") 
        
        self.cat = FPN_cat()
        # # Attention_Super_FPN 1
        # self.conv = nn.Sequential(
        #     SuperONN2d(in_channels=7*features,out_channels=features,kernel_size=3,q=q_order,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=True,dropout=None),
        #     SelfONN2d(in_channels=features,out_channels=out_channels,kernel_size=1, stride=1, padding=0, q=out_q_order, dropout=0)
        # )
        # # Attention_Super_FPN 2
        # self.conv = nn.Sequential(
        #     SuperONN2d(in_channels=7*features,out_channels=out_channels,kernel_size=3,q=q_order,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=True,dropout=None),
        #     # SelfONN2d(in_channels=features,out_channels=out_channels,kernel_size=1, stride=1, padding=0, q=out_q_order, dropout=0)
        # )
        # Attention_Super_FPN 3
        self.conv = nn.Sequential(
            SuperONN2d(in_channels=7*features,out_channels=features,kernel_size=3,q=q_order,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=True,dropout=None),
            SuperONN2d(in_channels=features,out_channels=out_channels,kernel_size=3,q=q_order,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=True,dropout=None)
        )


    def forward(self, x): 
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        aten4 = self.attention4(enc4, bottleneck)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, aten4), dim=1)
        dec4 = self.decoder4(dec4)

        aten3 = self.attention3(enc3, dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, aten3), dim=1)
        dec3 = self.decoder3(dec3)

        aten2 = self.attention2(enc2, dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, aten2), dim=1)
        dec2 = self.decoder2(dec2)

        aten1 = self.attention1(enc1, dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, aten1), dim=1)
        dec1 = self.decoder1(dec1)

        dec1 = self.cat(dec1, dec2, dec3)
        return self.conv(dec1) 

    @staticmethod
    def _block(in_channels, features, q_order, name):
        return nn.Sequential(
            OrderedDict(
                [ 
                    (name + "SelfConv1",SuperONN2d(in_channels=in_channels,out_channels=features,kernel_size=3,q=q_order,bias=True,padding=1,dilation=1,learnable=True,max_shift=2,rounded_shifts=True,dropout=None)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features, momentum=0.1)),
                    (name + "tanh", nn.Tanh()), 
                ]
            )
        )




class FPN_cat(nn.Module):

    def __init__(self, img_h=256):
        super(FPN_cat, self).__init__()
        self.img_h = img_h

    def forward(self, x1, x2, x3):
        x3 = F.interpolate(x3, x1.size()[2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, x1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2, x3), dim=1)
        return x




class AttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size = 1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
    
    def forward(self, x, g):
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode = 'bilinear', align_corners = False)
        out = self.psi(nn.ReLU()(x1 + g1))
        out = nn.Sigmoid()(out)
        return out*x