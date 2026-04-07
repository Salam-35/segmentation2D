from collections import OrderedDict

import torch
import torch.nn as nn


class M_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(M_UNet, self).__init__()

        features = init_features
        self.encoder1 = M_UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = M_UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = M_UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = M_UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = M_UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.residual4 = M_UNet._resblock(features * 8, features * 8, name="res4") 
        self.skip4 = M_UNet._skipblock(features * 8, features * 8, name="skip4")  
        self.decoder4 = M_UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.residual3 = M_UNet._resblock(features * 4, features * 4, name="res3") 
        self.skip3 = M_UNet._skipblock(features * 4, features * 4, name="skip3")  
        self.decoder3 = M_UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.residual2 = M_UNet._resblock(features * 2, features * 2, name="res2") 
        self.skip2 = M_UNet._skipblock(features * 2, features * 2, name="skip2")   
        self.decoder2 = M_UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2 
        )
        self.residual1 = M_UNet._resblock(features, features, name="res1") 
        self.skip1 = M_UNet._skipblock(features, features, name="skip1")    
        self.decoder1 = M_UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # nn.Upsample(mode='bilinear', scale_factor=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        res4 = self.residual4(self.pool4(enc4))   
        res4 = enc4 - res4 
        res4 = self.skip4(res4) 
        dec4 = torch.cat((dec4, res4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        res3 = self.residual3(self.pool3(enc3))  
        res3 = enc3 - res3 
        res3 = self.skip3(res3) 
        dec3 = torch.cat((dec3, res3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        res2 = self.residual2(self.pool2(enc2))  
        res2 = enc2 - res2 
        res2 = self.skip2(res2)  
        dec2 = torch.cat((dec2, res2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        res1 = self.residual1(self.pool1(enc1))    
        res1 = enc1 - res1 
        res1 = self.skip1(res1) 
        dec1 = torch.cat((dec1, res1), dim=1)
        dec1 = self.decoder1(dec1)
        # return torch.sigmoid(self.conv(dec1))
        return self.conv(dec1) 

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False, 
                        ), 
                    ), 
                    # (name + "drop1", nn.Dropout2d(p=0.8, inplace=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features,momentum=0.1)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ), 
                    ),
                    # (name + "drop2", nn.Dropout2d(p=0.5, inplace=False)), 
                    (name + "norm2", nn.BatchNorm2d(num_features=features,momentum=0.1)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


    @staticmethod
    def _resblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    ( 
                        name + "conv_Tranp",
                        nn.ConvTranspose2d(
                            features,
                            features,
                            kernel_size=2,
                            stride=2,
                            bias=False, 
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )


    @staticmethod
    def _skipblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    # (name + "drop1", nn.Dropout2d(p=0.8, inplace=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features,momentum=0.1)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )



