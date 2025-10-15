#!/usr/bin/env python3
"""
DISK backbone model for TensorRT compatibility
Already accepts RGB images natively
"""

import torch
from torch import nn


class DISKBackbone(nn.Module):
    """DISK U-Net backbone without keypoint selection post-processing"""

    def __init__(self, descriptor_dim: int = 128):
        super().__init__()
        self.descriptor_dim = descriptor_dim

        # Import and create U-Net
        from .disk.unet import Unet

        self.unet = Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64], up=[64, 64, 64, descriptor_dim + 1])

        # Load pretrained weights
        self._load_weights()

    def _load_weights(self):
        """Load DISK pretrained weights with CPU mapping"""
        try:
            # Use the DISK URL from the original implementation
            url = "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location=torch.device("cpu"))["extractor"]

            self.load_state_dict(state_dict)
            print("✅ Successfully loaded DISK pretrained weights")

        except Exception as e:
            print(f"⚠️  Could not load pretrained weights: {e}")
            print("   Proceeding with random weights for architecture testing...")

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that only runs the U-Net backbone

        Args:
            image: (B, 3, H, W) - RGB images

        Returns:
            heatmaps: (B, 1, H, W) - Dense detection heatmaps
            descriptors: (B, 128, H, W) - Dense descriptor maps
        """
        unet_output = self.unet(image)
        descriptors = unet_output[:, : self.descriptor_dim]  # (B, 128, H, W)
        heatmaps = unet_output[:, self.descriptor_dim]  # (B, H, W)

        return heatmaps, descriptors
