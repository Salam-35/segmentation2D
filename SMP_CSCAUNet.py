from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)


class Basic_blocks(nn.Module):
    def __init__(self, in_channel, out_channel):
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
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.gapool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.layer1(gp)
        x = x * se
        gap = self.gapool(x)
        se2 = self.layer2(gap)
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

class CSCA_DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, skip_channel, decay=2):
        super(CSCA_DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        self.conv = Basic_blocks(out_channel + skip_channel, out_channel // 2)
        self.catt = DSE(out_channel // 2, decay)
        self.satt = Spaceatt(out_channel // 2, decay)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        catt = self.catt(x)
        satt = self.satt(x, catt)
        plusatt = catt * satt
        return torch.cat([plusatt, catt], dim=1)


class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, decay=2):
        super(CenterBlock, self).__init__()
        self.double_conv = Basic_blocks(in_channels,out_channels)
        self.dscatt = DSE(out_channels, decay)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.dscatt(x)
        return x

class CSCAUnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
        decay = 2
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = CenterBlock(head_channels, head_channels, decay)

        self.channel_adjusts = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel_size=1) for in_ch, out_ch in zip(decoder_channels[:-1], decoder_channels[1:])])

        # if center:
        #     self.center = md.CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        # else:
        #     self.center = nn.Identity()

        blocks = []
        for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels):
            blocks.append(CSCA_DecoderBlock(in_ch, out_ch, skip_ch, decay))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        prev_decoder_output = None

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            #last block
            #if i == len(self.blocks) - 1:
            if i > 0:
                # prev_decoder_output = decoder_outputs[-1]
                prev_decoder_output = F.interpolate(prev_decoder_output, size=x.size()[2:], mode='bilinear', align_corners=False)
                prev_decoder_output = self.channel_adjusts[i-1](prev_decoder_output)
                x = x + prev_decoder_output
            
            prev_decoder_output = x
                
        return x

class CSCAUnet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        decay = 2
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        encoder_channels = self.encoder.out_channels
        decoder_channels = [encoder_channels[-1] // 2 ** i for i in range(encoder_depth)]

        self.decoder = CSCAUnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            decay=decay
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "csca-unet-{}".format(encoder_name)
        self.initialize()