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
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.descriptor_dim, kernel_size=1, stride=1, padding=0)

        # Load pretrained weights
        self._load_weights()

    def _load_weights(self):
        """Load SuperPoint pretrained weights"""
        try:
            state_dict = torch.hub.load_state_dict_from_url(self.weights_url, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)
            print("✅ Successfully loaded SuperPoint pretrained weights")
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

        # Keypoint detection head
        cPa = self.relu(self.convPa(x))
        heatmaps = self.convPb(cPa)

        # Descriptor head
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = F.normalize(descriptors, p=2, dim=1)

        return heatmaps, descriptors
