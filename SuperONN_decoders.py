from typing import Optional, Union, List
from fastonn import SuperONN2d

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

# Decoder block for all Unet-based models
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
            SuperONN2d(
                in_channels=in_channels + skip_channels,
                out_channels=out_channels,
                kernel_size=3,
                q=q_order,
                padding=1,
                max_shift=max_shift
            ),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()
        )

        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)

        self.conv2 = nn.Sequential(
            SuperONN2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                q=q_order,
                padding=1,
                max_shift=max_shift
            ),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh()
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            #print("\nskip:", skip.shape)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)

        return x

# Center block for all Unet-based models
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

# Decoder layer for all Unet-based models
# Decoder layer for all Unet-based models
class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        q_order,
        max_factor,
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

        # Remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        #print("Encoder channels after removing first skip:", encoder_channels)
        
        # Reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]
        #print("Reversed encoder channels:", encoder_channels)

        # Computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        #print("In channels:", in_channels)
        #print("Out channels:", out_channels)
        #print("Skip channels:", skip_channels)

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # Combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = []

        for idx, (in_ch, skip_ch, out_ch) in enumerate(zip(in_channels, skip_channels, out_channels)):
            # Simulate dynamic feature map size for debugging
            feature_map_height, feature_map_width = 128 // (2 ** idx), 128 // (2 ** idx)
            calculated_max_shift = min(feature_map_height, feature_map_width) // max_factor

            #print(f"Block {idx} | in_channels: {in_ch}, skip_channels: {skip_ch}, out_channels: {out_ch}")
            #print(f"Feature map size: ({feature_map_height}, {feature_map_width}), max_shift: {calculated_max_shift}")

            # Pass calculated max_shift to the DecoderBlock
            block = DecoderBlock(in_ch, skip_ch, out_ch, q_order, calculated_max_shift, **kwargs)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        #print("\nForward pass of UnetDecoder")
        features = features[1:]  # Remove first skip with same spatial resolution
        #print("Features after removing first skip:", [f.shape for f in features])

        features = features[::-1]  # Reverse channels to start from head of encoder
        #print("Reversed features:", [f.shape for f in features])

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        #print("Output after center block:", x.shape)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            #print(f"Passing to decoder block {i} | x: {x.shape}, skip: {None if skip is None else skip.shape}")
            x = decoder_block(x, skip)
            #print(f"Output from decoder block {i}:", x.shape)

        return x




# SelfONN_Unet Main model
class SuperONNUnet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        q_order=3,
        max_factor=4,
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
            q_order=q_order,
            max_factor=max_factor,
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
