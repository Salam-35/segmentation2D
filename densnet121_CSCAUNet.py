import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchsummary
from torchsummary.torchsummary import summary

class Basic_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=1):
        super(Basic_blocks, self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x=self.conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2+x

class DSE(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(DSE, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.gapool=nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.layer1(gp)
        x=x*se
        gap=self.gapool(x)
        se2=self.layer2(gap)
        return x * se2

class Spaceatt(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(Spaceatt, self).__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.BatchNorm2d(in_channel // decay),
            nn.Conv2d(in_channel // decay, 1, 1),
            nn.Sigmoid()
        )
        self.K = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay, in_channel//decay, 3, padding=1),
            DSE(in_channel//decay)
        )
        self.V = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay, in_channel//decay, 3, padding=1),
            DSE(in_channel//decay)
        )
        self.sig = nn.Sequential(
            nn.Conv2d(in_channel // decay, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, high):
        Q = self.Q(low)
        K = self.K(low)
        V = self.V(high)
        att = Q * K
        att = att @ V
        return self.sig(att)


class CSCA_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=2):
        super(CSCA_blocks, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channel, in_channel, 2, stride=2)
        self.conv = Basic_blocks(in_channel*2, out_channel // 2)
        self.catt = DSE(out_channel // 2, decay)
        self.satt = Spaceatt(out_channel // 2, decay)
    def forward(self, high, low):
        up = self.upsample(high)
        # print("up:", up.size())
        concat = torch.cat([up, low], dim=1)
        # print("concat:", concat.size())
        point = self.conv(concat)
        # print("point:", point.size())
        catt = self.catt(point)
        # print("catt:", catt.size())
        satt = self.satt(point, catt)
        # print("satt:", satt.size())
        plusatt = catt * satt
        # print("plusatt:", plusatt.size())
        return torch.cat([plusatt, catt], dim=1)



class CSCAUNet_DenseNet(nn.Module):
    def __init__(self, in_channels, n_class=1, decay=2):
        super(CSCAUNet_DenseNet, self).__init__()
        
        # Load pre-trained DenseNet121 as encoder
        self.densenet = models.densenet121(pretrained=True)
        
        # Remove the last linear and pool layers
        self.densenet.classifier = nn.Identity()
        
        # Modify the first convolution layer to accept 'in_channels'
        self.densenet.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.center_dse = DSE(1024, decay)
        
        # Decoder
        self.up_conv5 = CSCA_blocks(1024, 512, decay)
        self.up_conv4 = CSCA_blocks(512, 256, decay)
        self.up_conv3 = CSCA_blocks(256, 128, decay)
        self.up_conv2 = CSCA_blocks(128, 64, decay)
        self.up_conv1 = CSCA_blocks(64, 32, decay)

        self.skip_covn5 = nn.Conv2d(1024, 512, 1)
        self.skip_covn4 = nn.Conv2d(512, 256, 1)
        self.skip_covn3 = nn.Conv2d(256, 128, 1)
        # self.skip_covn2 = nn.Conv2d(128, 64, 1)

        self.dp6 = nn.Conv2d(1024, n_class, 1)
        self.dp5 = nn.Conv2d(1024, n_class, 1)
        self.dp4 = nn.Conv2d(512, n_class, 1)
        self.dp3 = nn.Conv2d(256, n_class, 1)
        self.dp2 = nn.Conv2d(128, n_class, 1)
        self.out = nn.Conv2d(32, n_class, 3, padding=1)

        self.center5 = nn.Conv2d(1024, 512, 1)
        self.decodeup4 = nn.Conv2d(512, 256, 1)
        self.decodeup3 = nn.Conv2d(256, 128, 1)
        self.decodeup2 = nn.Conv2d(128, 64, 1)

        self.upsample = nn.ConvTranspose2d(32, 32, 2, stride=2)


    def forward(self, inputs):
        b, c, h, w = inputs.size()
        
        # Encoder (DenseNet121)
        x = self.densenet.features.conv0(inputs)
        x = self.densenet.features.norm0(x)
        down1 = self.densenet.features.relu0(x) # torch.Size([16, 64, 128, 128])
        pool1 = self.densenet.features.pool0(down1)
        
        x = self.densenet.features.denseblock1(pool1)
        down2 = x  # torch.Size([16, 256, 64, 64])
        pool2 = self.densenet.features.transition1(down2)
        
        x = self.densenet.features.denseblock2(pool2)
        down3 = x  # torch.Size([16, 512, 32, 32])
        pool3 = self.densenet.features.transition2(down3)
        
        x = self.densenet.features.denseblock3(pool3)
        down4 = x  # torch.Size([16, 1024, 16, 16])
        pool4 = self.densenet.features.transition3(down4)
        
        x = self.densenet.features.denseblock4(pool4)
        down5 = x  # torch.Size([16, 1024, 8, 8])

        down5 = self.center_dse(down5)
        

        # # print sizes of skip connections
        # print("down1:", down1.size())
        # print("down2:", down2.size())
        # print("down3:", down3.size())
        # print("down4:", down4.size())
        # print("down5:", down5.size())

        # Decoder (keep the existing structure)
        # out6 = self.dp6(center)
        # out6 = F.interpolate(out6, (h, w), mode='bilinear', align_corners=False)
        # # print("out6:", out6.size())

        #deco5 = self.up_conv5(center, down5)
        # out5 = self.dp5(down5)
        # out5 = F.interpolate(out5, (h, w), mode='bilinear', align_corners=False)
        center5 = self.center5(down5) 
        #center5 = F.interpolate(center5, (h // 16, w // 16), mode='bilinear', align_corners=False)
        deco5 = center5
        # print("deco5:", deco5.size())
        # # print("out5:", out5.size())
        down4 = self.skip_covn5(down4)
        deco4 = self.up_conv4(deco5, down4)
        # out4 = self.dp4(deco4)
        # out4 = F.interpolate(out4, (h, w), mode='bilinear', align_corners=False)
        decoderup4 = self.decodeup4(deco5)
        decoderup4 = F.interpolate(decoderup4, (h // 16, w // 16), mode='bilinear', align_corners=False)
        deco4 = deco4 + decoderup4
        # print("deco4:", deco4.size())
        # # print("out4:", out4.size())

        down3 = self.skip_covn4(down3)
        deco3 = self.up_conv3(deco4, down3)
        # out3 = self.dp3(deco3)
        # out3 = F.interpolate(out3, (h, w), mode='bilinear', align_corners=False)
        decoderup3 = self.decodeup3(deco4)
        decoderup3 = F.interpolate(decoderup3, (h // 8, w //8), mode='bilinear', align_corners=False)
        deco3 = deco3 + decoderup3
        # print("deco3:", deco3.size())
        # # print("out3:", out3.size())

        down2 = self.skip_covn3(down2)
        deco2 = self.up_conv2(deco3, down2)
        # out2 = self.dp2(deco2)
        # out2 = F.interpolate(out2, (h, w), mode='bilinear', align_corners=False)
        decoderup2 = self.decodeup2(deco3)
        decoderup2 = F.interpolate(decoderup2, (h // 4, w // 4), mode='bilinear', align_corners=False)
        deco2 = deco2 + decoderup2
        # print("deco2:", deco2.size())
        # # print("out2:", out2.size())

        deco1 = self.up_conv1(deco2, down1)

        deco0 = self.upsample(deco1)
        out = self.out(deco0)

        # print("deco1:", deco1.size())
        # print("deco0:", deco0.size())
        # print("out:", out.size())
        return out

if __name__ == '__main__':
    model = CSCAUNet_DenseNet(1, 2)
    summary(model, (1, 256, 256), batch_size=1, device='cpu')
