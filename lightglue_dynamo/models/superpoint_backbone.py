#!/usr/bin/env python3
"""
SuperPoint backbone model for TensorRT compatibility
Accepts RGB images and converts to grayscale internally
"""

import torch
import torch.nn.functional as F
from torch import nn


class SuperPointBackbone(nn.Module):
    """SuperPoint backbone without keypoint selection post-processing"""

    weights_url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"

    def __init__(self, descriptor_dim: int = 256):
        super().__init__()
        self.descriptor_dim = descriptor_dim

        # Layers
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 64, kernel_size=1, stride=1, padding=0)  # Changed from 65 to 64

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.descriptor_dim, kernel_size=1, stride=1, padding=0)

        # Load pretrained weights
        self._load_weights()

    def _load_weights(self):
        """Load SuperPoint pretrained weights with optimizations"""
        try:
            state_dict = torch.hub.load_state_dict_from_url(self.weights_url, map_location=torch.device("cpu"))

            # Remove the 65th channel (dustbin) from convPb weights and bias
            if "convPb.weight" in state_dict:
                # Original shape: (65, 256, 1, 1) -> New shape: (64, 256, 1, 1)
                state_dict["convPb.weight"] = state_dict["convPb.weight"][:64]
                print(
                    f"✅ Trimmed convPb.weight from {state_dict['convPb.weight'].shape[0] + 1} to {state_dict['convPb.weight'].shape[0]} channels"
                )

            if "convPb.bias" in state_dict:
                # Original shape: (65,) -> New shape: (64,)
                state_dict["convPb.bias"] = state_dict["convPb.bias"][:64]
                print(
                    f"✅ Trimmed convPb.bias from {state_dict['convPb.bias'].shape[0] + 1} to {state_dict['convPb.bias'].shape[0]} channels"
                )

            self.load_state_dict(state_dict)
            print("✅ Successfully loaded SuperPoint pretrained weights with dustbin removal")
        except Exception as e:
            print(f"⚠️  Could not load pretrained weights: {e}")
            print("   Proceeding with random weights for architecture testing...")

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass accepting RGB images

        Args:
            image: RGB image tensor (B, 3, H, W)

        Returns:
            tuple of (heatmaps, descriptors)
            - heatmaps: Full resolution keypoint probability maps (B, H, W)
            - descriptors: Feature descriptors (B, D, H//8, W//8)
        """
        # Convert RGB to grayscale (SuperPoint expects grayscale)
        # RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        if image.size(1) == 3:
            image = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

        # Shared encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Keypoint detection head with baked-in softmax + reshaping
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)  # (B, 64, H//8, W//8)

        # Apply softmax and reshape to full resolution
        scores = F.softmax(scores, dim=1)  # Softmax over 64 channels
        b, _, h, w = scores.shape  # 64 channels
        s = 8  # SuperPoint scale factor (8x8 = 64)

        # Reshape from 64 channels of 8x8 patches to full resolution
        heatmaps = (
            scores.reshape(b, s, s, h, w)
            .permute(0, 3, 1, 4, 2)  # (B, H, S, W, S)
            .reshape(b, h * s, w * s)  # (B, H*8, W*8) - full resolution
        )

        # Descriptor head
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = F.normalize(descriptors, p=2, dim=1)

        return heatmaps, descriptors
