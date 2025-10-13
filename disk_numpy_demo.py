#!/usr/bin/env python3
"""
Pure NumPy implementation of DISK post-processing (no scipy dependency)
"""

import numpy as np
import time


def numpy_maximum_filter(array: np.ndarray, size: int) -> np.ndarray:
    """
    Pure NumPy implementation of maximum filter for NMS

    Args:
        array: 2D array to filter
        size: filter window size (must be odd)

    Returns:
        Filtered array of same shape
    """
    if size == 1:
        return array.copy()

    H, W = array.shape
    pad = size // 2

    # Pad array
    padded = np.pad(array, pad, mode="constant", constant_values=-np.inf)

    # Initialize output
    output = np.zeros_like(array)

    # Apply maximum filter
    for i in range(H):
        for j in range(W):
            # Extract window
            window = padded[i : i + size, j : j + size]
            output[i, j] = np.max(window)

    return output


def disk_postprocess_numpy(
    descriptors: np.ndarray,
    heatmaps: np.ndarray,
    num_keypoints: int = 1024,
    nms_size: int = 5,
    detection_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-process DISK outputs using pure NumPy (batch size 1 only)

    Args:
        descriptors: (1, 128, H, W) dense descriptor maps (batch size must be 1)
        heatmaps: (1, 1, H, W) detection heatmaps (batch size must be 1)
        num_keypoints: maximum number of keypoints to extract
        nms_size: NMS window size
        detection_threshold: minimum detection score

    Returns:
        keypoints: (N, 2) keypoint coordinates
        scores: (N,) keypoint scores
        descriptors: (N, 128) keypoint descriptors
    """
    B, C, H, W = heatmaps.shape
    assert B == 1, "Only batch size 1 supported"

    heatmap = heatmaps[0, 0]  # (H, W)
    desc_map = descriptors[0]  # (128, H, W)

    # Apply detection threshold
    valid_mask = heatmap > detection_threshold

    # Non-Maximum Suppression
    if nms_size > 1:
        nms_heatmap = numpy_maximum_filter(heatmap, nms_size)
        nms_mask = (heatmap == nms_heatmap) & valid_mask
    else:
        nms_mask = valid_mask

    # Get keypoint coordinates and scores
    y_coords, x_coords = np.where(nms_mask)
    scores = heatmap[nms_mask]

    if len(scores) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 128))

    # Select top-k keypoints
    if len(scores) > num_keypoints:
        top_indices = np.argpartition(scores, -num_keypoints)[-num_keypoints:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        x_coords = x_coords[top_indices]
        y_coords = y_coords[top_indices]
        scores = scores[top_indices]

    # Stack coordinates (x, y format)
    keypoints = np.stack([x_coords, y_coords], axis=1).astype(np.float32)

    # Sample descriptors at keypoint locations
    descriptors_kp = desc_map[:, y_coords, x_coords].T  # (N, 128)

    # L2 normalize descriptors
    norms = np.linalg.norm(descriptors_kp, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    descriptors_kp = descriptors_kp / norms

    return keypoints, scores, descriptors_kp


def demo_numpy_postprocessing():
    """Demo the pure NumPy post-processing"""
    print("🚀 Pure NumPy DISK Post-Processing Demo")
    print("=" * 50)

    # Create mock dense outputs (similar to TensorRT backbone outputs)
    # Fixed batch size of 1
    B, H, W = 1, 256, 256

    # Generate mock descriptor maps and heatmaps
    rng = np.random.default_rng(42)
    descriptors = rng.normal(0, 1, (B, 128, H, W)).astype(np.float32)

    # Create realistic heatmap with some peaks
    heatmaps = rng.random((B, 1, H, W)).astype(np.float32)

    # Add some artificial peaks for more realistic results
    for _ in range(100):
        y, x = rng.integers(20, H - 20), rng.integers(20, W - 20)
        heatmaps[0, 0, y - 2 : y + 3, x - 2 : x + 3] = rng.random() * 0.5 + 0.5

    print(f"📊 Input shapes:")
    print(f"   • Descriptors: {descriptors.shape}")
    print(f"   • Heatmaps: {heatmaps.shape}")
    print(f"   • Heatmap range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")

    # Test different configurations
    configs = [
        {"num_keypoints": 512, "nms_size": 3, "detection_threshold": 0.3},
        {"num_keypoints": 1024, "nms_size": 5, "detection_threshold": 0.2},
        {"num_keypoints": 2048, "nms_size": 7, "detection_threshold": 0.1},
    ]

    for i, config in enumerate(configs):
        print(f"\n🔧 Configuration {i + 1}: {config}")

        start_time = time.time()
        keypoints, scores, desc = disk_postprocess_numpy(descriptors, heatmaps, **config)
        processing_time = time.time() - start_time

        print(f"   ✅ Results:")
        print(f"      • Keypoints: {len(keypoints)}")
        print(f"      • Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"      • Descriptor norm: {np.linalg.norm(desc, axis=1).mean():.6f}")
        print(f"      • Processing time: {processing_time * 1000:.1f}ms")


def benchmark_postprocessing():
    """Benchmark the post-processing performance"""
    print("\n" + "=" * 50)
    print("⏱️  PERFORMANCE BENCHMARK")
    print("=" * 50)

    # Test different image sizes
    sizes = [(256, 256), (512, 512), (1024, 1024)]

    for H, W in sizes:
        print(f"\n📐 Image size: {H}x{W}")

        # Generate test data (fixed batch size of 1)
        rng = np.random.default_rng(42)
        descriptors = rng.normal(0, 1, (1, 128, H, W)).astype(np.float32)
        heatmaps = rng.random((1, 1, H, W)).astype(np.float32)

        # Benchmark
        times = []
        for _ in range(5):  # Run multiple times for stable measurement
            start = time.time()
            keypoints, scores, desc = disk_postprocess_numpy(
                descriptors, heatmaps, num_keypoints=1024, nms_size=5, detection_threshold=0.2
            )
            times.append(time.time() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"   • Keypoints extracted: {len(keypoints)}")
        print(f"   • Processing time: {avg_time * 1000:.1f} ± {std_time * 1000:.1f}ms")
        print(f"   • Throughput: {1 / avg_time:.1f} FPS")


def integration_example():
    """Show how to integrate with TensorRT pipeline"""
    print("\n" + "=" * 50)
    print("🔗 TENSORRT INTEGRATION EXAMPLE")
    print("=" * 50)

    example_code = """
# Example TensorRT + NumPy pipeline
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class DISKTensorRTPipeline:
    def __init__(self, engine_path: str):
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime().deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # Allocate GPU memory
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()
    
    def extract_features(self, image: np.ndarray):
        # 1. Preprocess image
        input_tensor = self._preprocess(image)
        
        # 2. Copy to GPU and run TensorRT inference
        cuda.memcpy_htod(self.inputs[0].device, input_tensor)
        self.context.execute_v2(bindings=self.bindings)
        
        # 3. Copy results back to CPU
        descriptors = np.empty(self.outputs[0].shape, dtype=np.float32)
        heatmaps = np.empty(self.outputs[1].shape, dtype=np.float32)
        cuda.memcpy_dtoh(descriptors, self.outputs[0].device)
        cuda.memcpy_dtoh(heatmaps, self.outputs[1].device)
        
        # 4. Post-process with NumPy (CPU)
        keypoints, scores, desc = disk_postprocess_numpy(
            descriptors, heatmaps,
            num_keypoints=1024,
            nms_size=5,
            detection_threshold=0.2
        )
        
        return keypoints, scores, desc

# Usage
pipeline = DISKTensorRTPipeline("disk_backbone_trt_compatible.trt")
keypoints, scores, descriptors = pipeline.extract_features(image)
"""

    print(example_code)

    print("🎯 Key Benefits:")
    print("✅ GPU accelerated backbone inference")
    print("✅ CPU post-processing (no GPU memory transfer overhead)")
    print("✅ Flexible keypoint selection parameters")
    print("✅ Easy to debug and modify")
    print("✅ Compatible with existing LightGlue pipeline")


if __name__ == "__main__":
    demo_numpy_postprocessing()
    benchmark_postprocessing()
    integration_example()

    print("\n" + "=" * 50)
    print("🎉 CONCLUSION")
    print("=" * 50)
    print("DISK is now TensorRT-ready with this hybrid approach:")
    print("• Export backbone: disk_backbone_trt_compatible.onnx → TensorRT")
    print("• Post-process: Pure NumPy (fast and flexible)")
    print("• Result: Same functionality as original DISK")
    print("• Performance: GPU acceleration + efficient CPU processing")
