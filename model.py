"""
Drift-Sense — model architecture.

DriftSenseNet: a lightweight Siamese-style heatmap-regression CNN.
- FeatureBackbone extracts features from the reference and
  search image independently (shared weights).
- DepthwiseXCorr cross-correlates the template feature map against the
  search feature map (a small ZNCC-like operation done in learned feature
  space rather than raw pixels, which is what lets it disambiguate the
  periodic regions a classical pixel-space matcher gets stuck on).
- A heatmap head turns the correlation volume into a per-location match
  score; an offset head refines the coordinate to sub-cell precision.
- log_prior is a fixed, non-trained Gaussian bias added to the heatmap
  logits before the final activation, centered on the search image middle.
  This encodes the tie-break rule ("if more than one region
  matches, return the one closest to the center") directly into the
  network's output distribution, rather than as a separate post-processing
  hack — so the network's own argmax already respects the tie-break by
  construction, and a naive argmax at inference time gives the same
  contractually-correct answer.

STRIDE=8 means a 1000x1000 input search image produces a 125x125 heatmap;
each heatmap cell covers an 8x8 pixel region of the input, refined by the
offset head.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 8
HEATMAP_SIZE = 125  


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        return F.silu(self.conv(x) + self.shortcut(x))


class FeatureBackbone(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels), nn.SiLU(inplace=True),
        )
        self.layer1 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.layer2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.layer3 = ConvBlock(base_channels * 4, base_channels * 4, stride=1)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(self.stem(x))))


class DepthwiseXCorr(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True),
        )

    def forward(self, z_feat, x_feat):
        z_feat = self.conv_kernel(z_feat)
        x_feat = self.conv_search(x_feat)
        out = []
        pad = z_feat.shape[3] // 2
        for i in range(x_feat.shape[0]):
            k = z_feat[i:i + 1]
            x_pad = F.pad(x_feat[i:i + 1], (pad, pad, pad, pad))
            out.append(F.conv2d(x_pad, k.permute(1, 0, 2, 3), groups=x_feat.shape[1]))
        return torch.cat(out, dim=0)


class DriftSenseNet(nn.Module):
    """
    Applied Materials Drift-Sense localization network.
    Uses a fixed Log-Gaussian center prior (added to the heatmap logits
    before the sigmoid) so the network's own highest-confidence prediction
    natively satisfies "closest to center" tie-break rule, rather
    than needing a separate post-processing override.
    """

    def __init__(self, base_c=32):
        super().__init__()
        self.backbone = FeatureBackbone(in_channels=1, base_channels=base_c)
        feat_c = base_c * 4
        self.xcorr = DepthwiseXCorr(feat_c)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_c, feat_c, 3, 1, 1), nn.BatchNorm2d(feat_c), nn.SiLU(inplace=True),
            nn.Conv2d(feat_c, 1, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(feat_c, feat_c, 3, 1, 1), nn.BatchNorm2d(feat_c), nn.SiLU(inplace=True),
            nn.Conv2d(feat_c, 2, 1),
        )

        ys, xs = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE]
        center = (HEATMAP_SIZE - 1) / 2.0
        log_prior = -((xs - center) ** 2 + (ys - center) ** 2) / (2.0 * (40.0 ** 2))
        self.register_buffer('log_prior', torch.from_numpy(log_prior).float().unsqueeze(0).unsqueeze(0))

    def forward(self, template, search):
        z_feat = self.backbone(template)
        x_feat = self.backbone(search)
        corr_feat = self.xcorr(z_feat, x_feat)
        heatmap_logits = self.heatmap_head(corr_feat) + self.log_prior
        offset = self.offset_head(corr_feat)
        return heatmap_logits, offset
