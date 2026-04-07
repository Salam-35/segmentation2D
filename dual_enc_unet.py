from typing import Any, Optional, Union, Callable, Sequence, Tuple

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Sequence, List
from segmentation_models_pytorch.base import modules as md


class UnetDecoderBlock(nn.Module):
    """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(
        self,
        feature_map: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_map = F.interpolate(
            feature_map,
            size=(target_height, target_width),
            mode=self.interpolation_mode,
        )
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
            feature_map = self.attention1(feature_map)
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.attention2(feature_map)
        return feature_map


class UnetCenterBlock(nn.Sequential):
    """Center block of the Unet decoder. Applied to the last feature map of the encoder."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
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


class UnetDecoder(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples them while
    combining with skip connections. This helps preserve fine-grained details in the final segmentation.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
        add_center_block: bool = False,
        interpolation_mode: str = "nearest",
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

        if add_center_block:
            self.center = UnetCenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels in zip(
            in_channels, skip_channels, out_channels
        ):
            block = UnetDecoderBlock(
                block_in_channels,
                block_skip_channels,
                block_out_channels,
                use_batchnorm=use_batchnorm,
                attention_type=attention_type,
                interpolation_mode=interpolation_mode,
            )
            self.blocks.append(block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skip_connections = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            height, width = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip_connection)

        return x

class DualEncoderUnet(SegmentationModel):
    """
    Dual Encoder U-Net architecture that uses two separate encoders and combines their features
    before passing to a single decoder for semantic image segmentation.

    The model combines features from both encoders at corresponding levels before passing
    them to the decoder. This allows the model to learn complementary features from
    different encoder architectures.

    Args:
        encoder_name_1: Name of the first encoder
        encoder_name_2: Name of the second encoder
        encoder_depth: Number of stages used in encoders (3-5)
        encoder_weights: Pretrained weights for both encoders (None, "imagenet", etc.)
        decoder_use_batchnorm: Use BatchNorm in decoder (True, False, "inplace")
        decoder_channels: Decoder channel sizes
        decoder_attention_type: Attention module in decoder (None or "scse")
        decoder_interpolation_mode: Interpolation mode ("nearest", "bilinear", etc.)
        in_channels: Input channels for both encoders (default: 3)
        classes: Number of output classes
        activation: Final activation function
        aux_params: Parameters for auxiliary classification head
        kwargs: Additional arguments passed to encoder initialization
    """

    requires_divisible_input_shape = False

    @supports_config_loading
    def __init__(
        self,
        encoder_name_1: str = "resnet34",
        encoder_name_2: str = "densenet121",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation_mode: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        # Initialize both encoders with same parameters except architecture
        self.encoder1 = get_encoder(
            encoder_name_1,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.encoder2 = get_encoder(
            encoder_name_2,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        # Get encoder channels for both encoders
        encoder_channels_1 = self.encoder1.out_channels
        encoder_channels_2 = self.encoder2.out_channels
        
        # Calculate combined encoder channels (each level will be concatenated)
        combined_channels = [
            ch1 + ch2 
            for ch1, ch2 in zip(encoder_channels_1, encoder_channels_2)
        ]

        # Initialize decoder with combined channels
        # Note: First channel is removed in decoder, so combination is correct
        add_center_block = any(name.startswith("vgg") for name in [encoder_name_1, encoder_name_2])
        self.decoder = UnetDecoder(
            encoder_channels=combined_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            add_center_block=add_center_block,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation_mode,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=combined_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"dual-u-{encoder_name_1}-{encoder_name_2}"
        self.initialize()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor that will be passed to both encoders

        Returns:
            Segmentation mask if aux_params is None, otherwise (mask, classification)
        """
        # Get features from both encoders
        features1 = self.encoder1(x)
        features2 = self.encoder2(x)

        # Combine features from both encoders at each level
        # The features list contains [same_res, 1/2, 1/4, 1/8, ...] for each encoder
        combined_features = [
            torch.cat([f1, f2], dim=1) 
            for f1, f2 in zip(features1, features2)
        ]

        # Decoder expects features in this format (from UnetDecoder implementation):
        # - First element is removed (same resolution)
        # - List is reversed to start from the deepest features
        decoder_output = self.decoder(combined_features)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(combined_features[-1])
            return masks, labels
        return masks

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: Input tensor that will be passed to both encoders

        Returns:
            prediction mask
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x