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

from collections.abc import Iterable
from itertools import repeat
import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, cat, no_grad
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from selfonnlayer import SelfONNLayer







def SelfONNBlock(in_channels, out_channels, kernel_size=3, padding=1, stride=1, q=3, bias=True):
    return nn.Sequential(
        SelfONNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding, q = q, bias=bias),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      
    )

#Residual block for ResUnet based Models
class SelfONN_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, q_order=3):
        super().__init__()
        self.res = nn.Sequential(
            SelfONNBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, q=q_order),
            SelfONNBlock(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1, q=q_order)
        )
        self.shortcut = nn.Sequential(
            SelfONNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1,stride=stride,padding=0, q = q_order),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        res = self.res(x)
        shortcut = self.shortcut(x)
        output = shortcut + res
        return torch.tanh(output)

#Decoder block for ResUnet based Models
class SelfONN_ResDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        q_order,
        max_shift,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = SelfONN_ResidualBlock(in_channels=in_channels+skip_channels, out_channels=out_channels, kernel_size=3,stride=1,padding=1, q_order = q_order)
        
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.attention2(x)
        return x




#Decoder block for all Unet based Models
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        q_order,
        max_shift,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            #nn.Conv2d(in_channels=in_channels+skip_channels, out_channels=out_channels, kernel_size=3,stride=1,padding=1),
            SelfONNLayer(in_channels=in_channels+skip_channels, out_channels=out_channels, kernel_size=3,stride=1,padding=1, q = q_order),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()
            )
        
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = nn.Sequential(
            #nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            SelfONNLayer(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1, q = q_order),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()

            )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            print("\nskip:", skip.shape)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        
        return x

#Center block for all Unet Based models
class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

#Decoder layer for all Unet based Models
class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        q_order,
        max_shift,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        print(in_channels, out_channels, skip_channels)

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, q_order, max_shift, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
    


#SelfONN_Unet Main model
class SelfONNUnet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        q_order = 3,
        max_shift = 18,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            q_order = q_order ,
            max_shift = max_shift,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


#Decoder for Self_UnetPlusPlus
class UnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        q_order,
        max_shift,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, q_order, max_shift, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], q_order, max_shift, **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
        return dense_x[f"x_{0}_{self.depth}"]

#SelfONN_UnetPlusPlus Main model
class SelfONNUnetPlusPlus(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        q_order = 3,
        max_shift = 18,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError("UnetPlusPlus is not support encoder_name={}".format(encoder_name))

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            q_order = q_order ,
            max_shift = max_shift,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()


#Decoder for SelfONN_ResUnet
class SelfONN_ResUnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        q_order,
        max_shift,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            SelfONN_ResDecoderBlock(in_ch, skip_ch, out_ch, q_order, max_shift, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
    
#SelfONN_ResUnet Main model
class SelfONN_ResUnet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        q_order = 3,
        max_shift = 18,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = SelfONN_ResUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            q_order = q_order ,
            max_shift = max_shift,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

#Blocks for SelfONN_FPN
class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, q_order=3):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            SelfONNBlock(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, q=q_order, bias=False),
            #nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels, q_order=3):
        super().__init__()

        self.skip_conv = SelfONNBlock(skip_channels, pyramid_channels, kernel_size=(1,1), padding=0, stride=1, q=q_order)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0, q_order=3):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples), q_order=q_order)]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True, q_order=q_order))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


class FPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
        q_order=3
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = SelfONNBlock(encoder_channels[0], pyramid_channels, kernel_size=(1,1), padding=0, stride=1, q=q_order)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1],q_order=q_order)
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2],q_order=q_order)
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3],q_order=q_order)

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples, q_order=q_order)
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x
    
class SelfONN_FPN(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        q_order = 3,
        max_shift = 0
    ):
        super().__init__()

        # validate input params
        if encoder_name.startswith("mit_b") and encoder_depth != 5:
            raise ValueError("Encoder {} support only encoder_depth=5".format(encoder_name))

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
            q_order=q_order
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
            
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

class Basic_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, q=3):
        super(Basic_blocks, self).__init__()
        self.conv=SelfONNLayer(in_channel,out_channel,1, q=q)
        self.conv1 = nn.Sequential(
            SelfONNLayer(out_channel, out_channel, 3, padding=1, q=q),
            nn.BatchNorm2d(out_channel),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            SelfONNLayer(out_channel, out_channel, 3, padding=1, q=q),
            nn.BatchNorm2d(out_channel),
            nn.Tanh()
        )

    def forward(self, x):
        x=self.conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2+x


class DSE(nn.Module):
    def __init__(self, in_channel, decay=2, q=3):
        super(DSE, self).__init__()
        self.layer1 = nn.Sequential(
            SelfONNLayer(in_channel, in_channel // decay, 1, q=q),
            nn.Tanh(),
            SelfONNLayer(in_channel // decay, in_channel, 1, q=q),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            SelfONNLayer(in_channel, in_channel // decay, 1, q=q),
            nn.Tanh(),
            SelfONNLayer(in_channel // decay, in_channel, 1, q=q),
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
    def __init__(self, in_channel, decay=2, q=3):
        super(Spaceatt, self).__init__()
        self.Q = nn.Sequential(
            SelfONNLayer(in_channel, in_channel // decay, 1, q=q),
            nn.BatchNorm2d(in_channel // decay),
            SelfONNLayer(in_channel // decay, 1, 1, q=q),
            nn.Sigmoid()
        )
        self.K = nn.Sequential(
            SelfONNLayer(in_channel, in_channel // decay, 3, padding=1, q=q),
            nn.BatchNorm2d(in_channel//decay),
            SelfONNLayer(in_channel//decay, in_channel//decay, 3, padding=1, q=q),
            DSE(in_channel//decay)
        )
        self.V = nn.Sequential(
            SelfONNLayer(in_channel, in_channel // decay, 3, padding=1, q=q),
            nn.BatchNorm2d(in_channel//decay),
            SelfONNLayer(in_channel//decay, in_channel//decay, 3, padding=1, q=q),
            DSE(in_channel//decay)
        )
        self.sig = nn.Sequential(
            SelfONNLayer(in_channel // decay, in_channel, 3, padding=1, q=q),
            nn.BatchNorm2d(in_channel),
            nn.Tanh()
        )

    def forward(self, low, high):
        Q = self.Q(low)
        K = self.K(low)
        V = self.V(high)
        att = Q * K
        att = att @ V
        return self.sig(att)

class CSCA_DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, skip_channel, decay=2, q=3):
        super(CSCA_DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        self.conv = Basic_blocks(out_channel + skip_channel, out_channel // 2, q=q)
        self.catt = DSE(out_channel // 2, decay, q=q)
        self.satt = Spaceatt(out_channel // 2, decay, q=q)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        catt = self.catt(x)
        satt = self.satt(x, catt)
        plusatt = catt * satt
        return torch.cat([plusatt, catt], dim=1)


class CSCA_CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, decay=2, q=3):
        super(CSCA_CenterBlock, self).__init__()
        self.double_conv = Basic_blocks(in_channels,out_channels,q=q)
        self.dscatt = DSE(out_channels, decay, q=q)

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
        decay = 2,
        q = 3
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

        self.center = CSCA_CenterBlock(head_channels, head_channels, decay, q=q)

        self.channel_adjusts = nn.ModuleList([SelfONNLayer(in_ch, out_ch, kernel_size=1, q=q) for in_ch, out_ch in zip(decoder_channels[:-1], decoder_channels[1:])])

        # if center:
        #     self.center = md.CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        # else:
        #     self.center = nn.Identity()

        blocks = []
        for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels):
            blocks.append(CSCA_DecoderBlock(in_ch, out_ch, skip_ch, decay, q=q))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        decoder_outputs = []

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            #last block
            #if i == len(self.blocks) - 1:
            if i > 0:
                prev_decoder_output = decoder_outputs[-1]
                prev_decoder_output = F.interpolate(prev_decoder_output, size=x.size()[2:], mode='bilinear', align_corners=False)
                prev_decoder_output = self.channel_adjusts[i-1](prev_decoder_output)
                x = x + prev_decoder_output
            
            decoder_outputs.append(x)
                
        return x

class SelfONN_CSCAUnet(SegmentationModel):
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
        decay = 2,
        q = 3
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
            decay=decay,
            q=q
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

        self.name = "self-csca-unet-{}".format(encoder_name)
        self.initialize()


class PAB(nn.Module):
    def __init__(self, in_channels, out_channels, q_order = 3, pab_channels=64):
        super(PAB, self).__init__()
        # Series of 1x1 conv to generate attention feature maps
        self.pab_channels = pab_channels
        self.in_channels = in_channels
        self.top_conv = SelfONNLayer(in_channels, pab_channels, kernel_size=1, q = q_order)
        self.center_conv = SelfONNLayer(in_channels, pab_channels, kernel_size=1, q = q_order)
        self.bottom_conv = SelfONNLayer(in_channels, in_channels, kernel_size=3, padding=1, q = q_order)
        self.map_softmax = nn.Softmax(dim=1)
        self.out_conv = SelfONNLayer(in_channels, in_channels, kernel_size=3, padding=1, q = q_order)

    def forward(self, x):
        bsize = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        x_top = self.top_conv(x)
        x_center = self.center_conv(x)
        x_bottom = self.bottom_conv(x)

        x_top = x_top.flatten(2)
        x_center = x_center.flatten(2).transpose(1, 2)
        x_bottom = x_bottom.flatten(2).transpose(1, 2)

        sp_map = torch.matmul(x_center, x_top)
        sp_map = self.map_softmax(sp_map.view(bsize, -1)).view(bsize, h * w, h * w)
        sp_map = torch.matmul(sp_map, x_bottom)
        sp_map = sp_map.reshape(bsize, self.in_channels, h, w)
        x = x + sp_map
        x = self.out_conv(x)
        return x


class MFAB(nn.Module):
    def __init__(
        self, in_channels, skip_channels, out_channels, q_order = 3, use_batchnorm=True, reduction=16
    ):
        # MFAB is just a modified version of SE-blocks, one for skip, one for input
        super(MFAB, self).__init__()
        self.hl_conv = nn.Sequential(
            nn.Sequential(
            SelfONNLayer(in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                q = q_order),
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()
            ),
            nn.Sequential(
            SelfONNLayer(in_channels, skip_channels, kernel_size=1),
            nn.BatchNorm2d(skip_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()
            )
        )
        reduced_channels = max(1, skip_channels // reduction)
        self.SE_ll = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            SelfONNLayer(skip_channels, reduced_channels, 1, q = q_order),
            nn.Tanh(),
            SelfONNLayer(reduced_channels, skip_channels, 1, q = q_order),
            nn.Sigmoid(),
        )
        self.SE_hl = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            SelfONNLayer(skip_channels, reduced_channels, 1, q = q_order),
            nn.Tanh(),
            SelfONNLayer(reduced_channels, skip_channels, 1, q = q_order),
            nn.Sigmoid(),
        )
        self.conv1 = nn.Sequential(
            nn.Sequential(
            SelfONNLayer(skip_channels
            + skip_channels,  # we transform C-prime form high level to C from skip connection
            out_channels,
            kernel_size=3,
            padding=1,
            q = q_order),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()
            ))
        self.conv2 = nn.Sequential(
            nn.Sequential(
            SelfONNLayer(out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            q = q_order),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()
            ))

    def forward(self, x, skip=None):
        x = self.hl_conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        attention_hl = self.SE_hl(x)
        if skip is not None:
            attention_ll = self.SE_ll(skip)
            attention_hl = attention_hl + attention_ll
            x = x * attention_hl
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MAnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, q_order = 3, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            SelfONNLayer(in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            q = q_order),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()
            )
        self.conv2 = nn.Sequential(
            SelfONNLayer(out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            q = q_order),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()
            )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MAnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        reduction=16,
        use_batchnorm=True,
        pab_channels=64,
        q_order = 3
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]

        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = PAB(head_channels, head_channels, pab_channels=pab_channels, q_order = q_order)

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm)  # no attention type here
        blocks = [
            MFAB(in_ch, skip_ch, out_ch, reduction=reduction, q_order=q_order, **kwargs)
            if skip_ch > 0
            else MAnetDecoderBlock(in_ch, skip_ch, out_ch, q_order, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        # for the last we dont have skip connection -> use simple decoder block
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class SelfONN_MAnet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_pab_channels: int = 64,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        q_order: int = 3,
        max_shift = 1
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = MAnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            pab_channels=decoder_pab_channels,
            q_order = q_order
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

        self.name = "selfonn-manet-{}".format(encoder_name)
        self.initialize()