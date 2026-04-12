"""
MSPC (Multi-Scale Parallel Convolution) models.

Usage in config.py:
    config['model_type']   = 'SMP'
    config['model_to_load'] = 'mobilenet_v2*MSPCUNetPlusPlus'   # or MSPCUnet
    config['decoder_attention'] = None   # or 'scse'

The encoder can be any SMP-compatible encoder (mobilenet_v2, resnet50, efficientnet-b4, …).
The MSPC block is inserted at the bottleneck (last encoder feature map) before decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np


# ─────────────────────────────────────────────
# Squeeze-Excitation Block
# ─────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ─────────────────────────────────────────────
# MSPC Block (Multi-Scale Parallel Conv)
#   5 parallel branches covering receptive fields up to ~23 px:
#     3×3 standard | 5×5 dw-sep | 7×7 dw-sep | dilated-3 | dilated-5
# ─────────────────────────────────────────────
class MSPCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = max(1, out_channels // 5)

        # Branch 1: 3×3 standard conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        # Branch 2: 5×5 depthwise-separable
        # depthwise: groups=in_channels, out must equal in; then pointwise to mid
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        # Branch 3: 7×7 depthwise-separable
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        # Branch 4: 3×3 dilated rate=3  (effective 7×7)
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        # Branch 5: 3×3 dilated rate=5  (effective 11×11)
        self.b5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels)

        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x), self.b5(x)], dim=1)
        out = self.fuse(out)
        out = self.se(out)
        return out + self.residual(x)


# ─────────────────────────────────────────────
# Base class: shared encoder + MSPC bottleneck logic
# ─────────────────────────────────────────────
class _MSPCBase(nn.Module):
    """Shared setup used by both MSPCUnet and MSPCUNetPlusPlus."""

    def _build(self, base_model, deep_supervision, num_classes):
        self.deep_supervision = deep_supervision
        self.base = base_model

        bottleneck_ch = self.base.encoder.out_channels[-1]
        self.mspc = MSPCBlock(bottleneck_ch, bottleneck_ch)

    def _encode_decode(self, x):
        features = self.base.encoder(x)
        features[-1] = self.mspc(features[-1])
        decoder_output = self.base.decoder(*features)
        main_out = self.base.segmentation_head(decoder_output)
        return main_out

    def forward(self, x):
        main_out = self._encode_decode(x)

        if self.training and self.deep_supervision:
            aux1 = F.interpolate(main_out, scale_factor=0.5,   mode='bilinear', align_corners=False)
            aux2 = F.interpolate(main_out, scale_factor=0.25,  mode='bilinear', align_corners=False)
            aux3 = F.interpolate(main_out, scale_factor=0.125, mode='bilinear', align_corners=False)
            return main_out, aux1, aux2, aux3

        return main_out


# ─────────────────────────────────────────────
# MSPCUnet  —  any encoder + MSPC bottleneck + UNet decoder
# ─────────────────────────────────────────────
class MSPCUnet(_MSPCBase):
    """
    Args match the SMP interface used in models.py:
        encoder_name, encoder_depth, encoder_weights,
        decoder_attention_type, in_channels, classes, activation,
        deep_supervision (optional, default False)
    """
    def __init__(
        self,
        encoder_name='mobilenet_v2',
        encoder_depth=5,
        encoder_weights='imagenet',
        decoder_attention_type=None,
        in_channels=3,
        classes=16,
        activation=None,
        deep_supervision=False,
    ):
        super().__init__()
        base = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
        self._build(base, deep_supervision, classes)


# ─────────────────────────────────────────────
# MSPCUNetPlusPlus  —  any encoder + MSPC bottleneck + UNet++ decoder
# ─────────────────────────────────────────────
class MSPCUNetPlusPlus(_MSPCBase):
    """
    Args match the SMP interface used in models.py:
        encoder_name, encoder_depth, encoder_weights,
        decoder_attention_type, in_channels, classes, activation,
        deep_supervision (optional, default False)
    """
    def __init__(
        self,
        encoder_name='mobilenet_v2',
        encoder_depth=5,
        encoder_weights='imagenet',
        decoder_attention_type=None,
        in_channels=3,
        classes=16,
        activation=None,
        deep_supervision=False,
    ):
        super().__init__()
        base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
        self._build(base, deep_supervision, classes)


# ─────────────────────────────────────────────
# Deep-Supervision Loss (used when deep_supervision=True)
# ─────────────────────────────────────────────
class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss, weights=(1.0, 0.4, 0.2, 0.1)):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, outputs, target):
        if isinstance(outputs, (list, tuple)):
            total = 0
            for w, out in zip(self.weights, outputs):
                t = target
                if out.shape[-2:] != target.shape[-2:]:
                    t = F.interpolate(
                        target.float().unsqueeze(1),
                        size=out.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                total += w * self.base_loss(out, t)
            return total
        return self.base_loss(outputs, target)


# ─────────────────────────────────────────────
# 3-Channel Multi-Window Preprocessing  (HU → 3-ch ImageNet-norm)
# ─────────────────────────────────────────────
def hu_window(slice_hu, center, width):
    low  = center - width / 2
    high = center + width / 2
    return np.clip((slice_hu.astype(np.float32) - low) / (high - low), 0, 1)


def multi_window_preprocess(slice_hu):
    """
    Input:  2-D numpy array of HU values (H, W)
    Output: (3, H, W) float32, ImageNet-normalised
    """
    ch1 = hu_window(slice_hu, center=-60,  width=400)   # soft tissue
    ch2 = hu_window(slice_hu, center=40,   width=400)   # organ parenchyma
    ch3 = hu_window(slice_hu, center=200,  width=700)   # vessel / bone
    img = np.stack([ch1, ch2, ch3], axis=0).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return (img - mean) / std


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == '__main__':
    for cls_name, cls in [('MSPCUnet', MSPCUnet), ('MSPCUNetPlusPlus', MSPCUNetPlusPlus)]:
        print(f'\n=== {cls_name} ===')
        model = cls(encoder_name='mobilenet_v2', classes=16, deep_supervision=True)
        model.train()
        x = torch.randn(2, 3, 256, 256)
        outs = model(x)
        print('Train outputs:')
        for i, o in enumerate(outs):
            print(f'  {i}: {o.shape}')
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f'Eval output: {out.shape}')
        total = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'Params: {total:.2f}M')
