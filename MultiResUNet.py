import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class conv_bn(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3,stride=1,padding=1,act='relu',bias=False):
        super(conv_bn,self).__init__()
        if act == None:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
                nn.BatchNorm2d(ch_out)
            )
        elif act == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
        elif act == 'sigmoid':
            self.conv = nn.Sequential( 
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
                nn.BatchNorm2d(ch_out),
                nn.Sigmoid()
            )
        elif act == 'Tanh':
            self.conv = nn.Sequential( 
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
                nn.BatchNorm2d(ch_out),
                nn.Tanh()
            )

    def forward(self,x):
        x = self.conv(x)
        return x


class trans_conv_bn(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=2, stride=2, padding=0):
        super(trans_conv_bn,self).__init__() 
        self.up = nn.Sequential(
            # nn.ConvTranspose2d(ch_in , ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class MultiResBlock(nn.Module):
    def __init__(self, inp, U, alpha): 
        super(MultiResBlock,self).__init__() 
        self.W = alpha * U
        self.inp = inp
        self.outp_1 = int(self.W*0.167)
        self.outp_2 = int(self.W*0.333) 
        self.outp_3 = int(self.W*0.5) 
        self.out_res = self.outp_1+self.outp_2+self.outp_3 
        self.residual_layer = conv_bn(ch_in=self.inp, ch_out=self.out_res, kernel_size=1, stride=1, padding=0, act=None,bias=False)
        self.conv3x3 = conv_bn(ch_in=self.inp,    ch_out=self.outp_1,kernel_size=3,stride=1,padding=1,act='relu',bias=False)
        self.conv5x5 = conv_bn(ch_in=self.outp_1, ch_out=self.outp_2,kernel_size=3,stride=1,padding=1,act='relu',bias=False)
        self.conv7x7 = conv_bn(ch_in=self.outp_2, ch_out=self.outp_3,kernel_size=3,stride=1,padding=1,act='relu',bias=False)
        self.relu = nn.ReLU(inplace=True) 
        self.batchnorm_1 = nn.BatchNorm2d(self.out_res)
        self.batchnorm_2 = nn.BatchNorm2d(self.out_res)
        
    def forward(self, x):
        res = self.residual_layer(x)
        sbs = self.conv3x3(x)
        obo = self.conv5x5(sbs)
        cbc = self.conv7x7(obo)
        all_t = torch.cat((sbs, obo, cbc), 1)
        all_t_b = self.batchnorm_1(all_t) 
        out = all_t_b.add(res)
        out = self.relu(out)
        out = self.batchnorm_2(out)
        return out


class res_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(res_block,self).__init__()
        self.res =  conv_bn(ch_in=ch_in, ch_out=ch_out, kernel_size=1, stride=1, padding=0, act=None, bias=False)
        self.main = conv_bn(ch_in=ch_in, ch_out=ch_out, kernel_size=3, stride=1, padding=1, act='relu',bias=False)
        self.relu = nn.ReLU(inplace=True) 
        self.batchnorm = nn.BatchNorm2d(ch_out)

    def forward(self,x):
        res_x = self.res(x)
        main_x = self.main(x)
        out = res_x.add(main_x)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out


class ResPath(nn.Module):
    def __init__(self,ch_in,ch_out,stage):
        super(ResPath,self).__init__()
        self.stage = stage
        if self.stage ==1:
            self.block = res_block(ch_in, ch_out)
        elif self.stage ==2:
            self.block = nn.Sequential(res_block(ch_in, ch_out), res_block(ch_out, ch_out))
        elif self.stage ==3:
            self.block = nn.Sequential(res_block(ch_in, ch_out), res_block(ch_out, ch_out), res_block(ch_out, ch_out))
        elif self.stage ==4:
            self.block = nn.Sequential(res_block(ch_in, ch_out), res_block(ch_out, ch_out), res_block(ch_out, ch_out), res_block(ch_out, ch_out))

    def forward(self, x):
        out = self.block(x)
        return out


class MultiResUNet(nn.Module):
    def __init__(self, in_features=32, alpha=1.67, in_channels=1, out_channels=2): 
        super(MultiResUNet,self).__init__()
        U = in_features
        outp_1 = alpha*0.167
        outp_2 = alpha*0.333
        outp_3 = alpha*0.5


        # pooling to be used after each enocder
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        # encoder
        self.mresblock1 = MultiResBlock(inp=in_channels, U=in_features, alpha=alpha)  
        b1_out = int(U*outp_1) + int(U*outp_2)  + int(U*outp_3)  
        self.res_path1 = ResPath(ch_in=b1_out,ch_out=in_features,stage=4)
        
        self.mresblock2 = MultiResBlock(inp=b1_out, U=2*in_features, alpha=alpha)  
        b2_out = int(2*U*outp_1) + int(2*U*outp_2)  + int(2*U*outp_3)   
        self.res_path2 = ResPath(ch_in=b2_out,ch_out=2*in_features,stage=3)

        self.mresblock3 = MultiResBlock(inp=b2_out, U=4*in_features, alpha=alpha)  
        b3_out = int(4*U*outp_1) + int(4*U*outp_2)  + int(4*U*outp_3)   
        self.res_path3 = ResPath(ch_in=b3_out,ch_out=4*in_features,stage=2)

        self.mresblock4 = MultiResBlock(inp=b3_out, U=8*in_features, alpha=alpha)  
        b4_out = int(8*U*outp_1) + int(8*U*outp_2)  + int(8*U*outp_3)  
        self.res_path4 = ResPath(ch_in=b4_out,ch_out=8*in_features,stage=1)

        # bottle neck
        self.mresblock5 = MultiResBlock(inp=b4_out, U=16*in_features, alpha=alpha)  
        b5_out = int(16*U*outp_1) + int(16*U*outp_2)  + int(16*U*outp_3)  

        # decoder
        self.upconv4 = trans_conv_bn(ch_in=b5_out,ch_out=8*in_features,kernel_size=2, stride=2, padding=0)
        self.decoder4 = MultiResBlock(inp=2*8*in_features, U=8*in_features, alpha=alpha)

        self.upconv3 = trans_conv_bn(ch_in=b4_out,ch_out=4*in_features,kernel_size=2, stride=2, padding=0)
        self.decoder3 = MultiResBlock(inp=2*4*in_features, U=4*in_features, alpha=alpha)

        self.upconv2 = trans_conv_bn(ch_in=b3_out,ch_out=2*in_features,kernel_size=2, stride=2, padding=0)
        self.decoder2 = MultiResBlock(inp=2*2*in_features, U=2*in_features, alpha=alpha)

        self.upconv1 = trans_conv_bn(ch_in=b2_out,ch_out=in_features,kernel_size=2, stride=2, padding=0)
        self.decoder1 = MultiResBlock(inp=2*in_features, U=in_features, alpha=alpha)

        # final conv 
        self.final_conv = nn.Conv2d(in_channels=b1_out, out_channels=out_channels, kernel_size=1)

    def forward(self, x): 

        enc1 = self.mresblock1(x)
        enc2 = self.mresblock2(self.Maxpool(enc1))
        enc3 = self.mresblock3(self.Maxpool(enc2))
        enc4 = self.mresblock4(self.Maxpool(enc3))

        bottleneck = self.mresblock5(self.Maxpool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, self.res_path4(enc4)), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self.res_path3(enc3)), dim=1)
        dec3 = self.decoder3(dec3) 

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self.res_path2(enc2)), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, self.res_path1(enc1)), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1) 



