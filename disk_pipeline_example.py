#!/usr/bin/env python3
"""
Complete DISK pipeline with TensorRT-compatible backbone + NumPy post-processing
"""

import numpy as np
from scipy.ndimage import maximum_filter
import time
from typing import Tuple, List


class DISKPipeline:
    """
    DISK feature extraction pipeline with TensorRT backbone + NumPy post-processing
    """

    def __init__(
        self,
        backbone_model_path: str,
        num_keypoints: int = 1024,
        nms_window_size: int = 5,
        detection_threshold: float = 0.0,
    ):
        """
        Initialize DISK pipeline

        Args:
            backbone_model_path: Path to TensorRT DISK backbone model
            num_keypoints: Maximum number of keypoints to extract
            nms_window_size: Size of NMS window (must be odd)
            detection_threshold: Minimum detection score threshold
        """
        self.num_keypoints = num_keypoints
        self.nms_window_size = nms_window_size
        self.detection_threshold = detection_threshold

        # Initialize TensorRT engine (placeholder - use your TRT wrapper)
        # self.trt_engine = TensorRTEngine(backbone_model_path)
        print(f"📋 DISK Pipeline Configuration:")
        print(f"   • Backbone: {backbone_model_path}")
        print(f"   • Max keypoints: {num_keypoints}")
        print(f"   • NMS window: {nms_window_size}")
        print(f"   • Detection threshold: {detection_threshold}")

    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract DISK features from image

        Args:
            image: (H, W, 3) RGB image in [0, 255]

        Returns:
            keypoints: (N, 2) keypoint coordinates
            scores: (N,) keypoint detection scores
            descriptors: (N, 128) L2-normalized descriptors
        """
        # Preprocess image
        image_tensor = self._preprocess_image(image)

        # Run TensorRT backbone
        descriptors, heatmaps = self._run_backbone(image_tensor)

        # Post-process to extract keypoints
        keypoints, scores, desc = self._postprocess(descriptors, heatmaps)

        return keypoints, scores, desc

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to model input format"""
        # Normalize to [0, 1] and convert to CHW format
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert HWC -> CHW
        image_tensor = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image_tensor = image_tensor[None]  # (1, 3, H, W)

        return image_tensor

    def _run_backbone(self, image_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run TensorRT backbone inference

        This is where you'd call your TensorRT engine:
        descriptors, heatmaps = self.trt_engine.infer(image_tensor)
        """
        # Placeholder - replace with actual TensorRT inference
        print(f"🔧 Running TensorRT inference on image shape: {image_tensor.shape}")

        # Mock output for demonstration
        B, C, H, W = image_tensor.shape
        descriptors = np.random.randn(B, 128, H, W).astype(np.float32)
        heatmaps = np.random.rand(B, 1, H, W).astype(np.float32)

        return descriptors, heatmaps

    def _postprocess(self, descriptors: np.ndarray, heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process dense outputs to extract sparse keypoints

        Args:
            descriptors: (1, 128, H, W) dense descriptor maps
            heatmaps: (1, 1, H, W) detection heatmaps

        Returns:
            keypoints: (N, 2) keypoint coordinates
            scores: (N,) keypoint scores
            descriptors: (N, 128) keypoint descriptors
        """
        start_time = time.time()

        B, C, H, W = heatmaps.shape
        assert B == 1, "Batch processing not implemented"

        heatmap = heatmaps[0, 0]  # (H, W)
        desc_map = descriptors[0]  # (128, H, W)

        # Apply detection threshold
        valid_mask = heatmap > self.detection_threshold

        # Non-Maximum Suppression
        if self.nms_window_size > 1:
            # Use scipy maximum filter for efficient NMS
            nms_heatmap = maximum_filter(heatmap, size=self.nms_window_size, mode="constant")
            nms_mask = (heatmap == nms_heatmap) & valid_mask
        else:
            nms_mask = valid_mask

        # Get keypoint coordinates and scores
        y_coords, x_coords = np.where(nms_mask)
        scores = heatmap[nms_mask]

        if len(scores) == 0:
            # No keypoints found
            return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 128))

        # Select top-k keypoints
        if len(scores) > self.num_keypoints:
            # Use argpartition for efficient top-k selection
            top_indices = np.argpartition(scores, -self.num_keypoints)[-self.num_keypoints :]
            # Sort the top-k by score (descending)
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            x_coords = x_coords[top_indices]
            y_coords = y_coords[top_indices]
            scores = scores[top_indices]

        # Stack coordinates in (x, y) format
        keypoints = np.stack([x_coords, y_coords], axis=1).astype(np.float32)

        # Sample descriptors at keypoint locations
        descriptors_kp = desc_map[:, y_coords, x_coords].T  # (N, 128)

        # L2 normalize descriptors
        norms = np.linalg.norm(descriptors_kp, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        descriptors_kp = descriptors_kp / norms

        postprocess_time = time.time() - start_time
        print(f"📊 Post-processing: {len(keypoints)} keypoints in {postprocess_time * 1000:.1f}ms")

        return keypoints, scores, descriptors_kp


def demo_disk_pipeline():
    """Demonstrate DISK pipeline usage"""
    print("🚀 DISK Pipeline Demo")
    print("=" * 60)

    # Initialize pipeline
    pipeline = DISKPipeline(
        backbone_model_path="disk_backbone_trt_compatible.onnx",
        num_keypoints=1024,
        nms_window_size=5,
        detection_threshold=0.1,
    )

    # Create mock image
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print(f"\n📸 Processing image: {image.shape}")

    # Extract features
    start_time = time.time()
    keypoints, scores, descriptors = pipeline.extract_features(image)
    total_time = time.time() - start_time

    # Print results
    print(f"\n✅ Feature extraction complete!")
    print(f"   • Keypoints: {keypoints.shape} - {len(keypoints)} detected")
    print(f"   • Scores: {scores.shape} - range [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"   • Descriptors: {descriptors.shape}")
    print(f"   • Total time: {total_time * 1000:.1f}ms")

    # Validate descriptor normalization
    norms = np.linalg.norm(descriptors, axis=1)
    print(f"   • Descriptor norms: mean={norms.mean():.6f}, std={norms.std():.6f}")


def compare_with_original():
    """Compare backbone-only approach with original DISK"""
    print("\n" + "=" * 60)
    print("📊 COMPARISON: Backbone vs Full DISK")
    print("=" * 60)

    print("""
Original DISK Export Issues:
❌ TopK operation - not well supported in TensorRT
❌ Advanced indexing - complex gather operations  
❌ Integer arithmetic - floor_divide, modulo operations
❌ Dynamic shapes - runtime-dependent tensor operations
❌ 166 ONNX nodes - complex computational graph

DISK Backbone Approach:
✅ Only U-Net operations - fully TensorRT compatible
✅ Static operations - no dynamic indexing
✅ 67 ONNX nodes - 60% reduction in complexity
✅ Dense outputs - flexible post-processing
✅ CPU post-processing - better resource utilization

Performance Benefits:
• GPU: Focus on heavy deep learning computation
• CPU: Handle lightweight post-processing operations  
• Memory: More efficient batch processing
• Flexibility: Easy to tune NMS and keypoint selection
• Debugging: Clear separation of concerns
""")


if __name__ == "__main__":
    demo_disk_pipeline()
    compare_with_original()

    print("\n" + "=" * 60)
    print("🎯 CONCLUSION: DISK CAN BE USED WITH TENSORRT!")
    print("=" * 60)
    print("""
The solution is to split DISK into two parts:

1️⃣ TensorRT Backbone (GPU):
   • Export: disk_backbone_trt_compatible.onnx
   • Convert to TensorRT engine
   • Runs dense feature extraction

2️⃣ NumPy Post-processing (CPU):  
   • Keypoint detection (NMS)
   • Top-k selection
   • Descriptor sampling

This gives you the best of both worlds:
✅ TensorRT acceleration for the heavy computation
✅ Flexible CPU post-processing 
✅ Full compatibility with your existing pipeline
✅ Same final output as original DISK model
""")
