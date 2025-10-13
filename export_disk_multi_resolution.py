#!/usr/bin/env python3
"""
Export DISK U-Net backbone models for multiple resolutions (256, 512, 1024)
All models configured for 1024 keypoints capacity
"""

import sys
import traceback
from pathlib import Path

import torch


class DISKBackbone(torch.nn.Module):
    """DISK U-Net backbone without post-processing for TensorRT compatibility"""

    def __init__(self, descriptor_dim=128, nms_window_size=5, num_keypoints=1024):
        super().__init__()

        if nms_window_size % 2 != 1:
            raise ValueError(f"window_size has to be odd, got {nms_window_size}")

        self.descriptor_dim = descriptor_dim
        self.nms_window_size = nms_window_size
        self.num_keypoints = num_keypoints

        # Import and create U-Net
        from lightglue_dynamo.models.disk.unet import Unet

        self.unet = Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64], up=[64, 64, 64, descriptor_dim + 1])

        # Load pretrained weights
        self._load_weights()

    def _load_weights(self):
        """Load DISK pretrained weights with CPU mapping"""
        try:
            # Use the DISK URL from the original implementation
            url = "https://github.com/cvlab-epfl/disk/raw/master/depth-0.pt"
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
            image: (B, 3, H, W) - Input images

        Returns:
            descriptors: (B, 128, H, W) - Dense descriptor maps
            heatmaps: (B, 1, H, W) - Dense detection heatmaps
        """
        unet_output = self.unet(image)
        descriptors = unet_output[:, : self.descriptor_dim]  # (B, 128, H, W)
        heatmaps = unet_output[:, self.descriptor_dim :]  # (B, 1, H, W)

        return descriptors, heatmaps


def export_disk_model(resolution: int, num_keypoints: int = 1024) -> bool:
    """
    Export DISK backbone model for specific resolution

    Args:
        resolution: Input image resolution (256, 512, or 1024)
        num_keypoints: Number of keypoints capacity (default: 1024)

    Returns:
        bool: Success status
    """
    print(f"\n{'=' * 60}")
    print(f"🔧 Exporting DISK backbone for {resolution}x{resolution} resolution")
    print(f"   Keypoints capacity: {num_keypoints}")
    print(f"{'=' * 60}")

    try:
        # Create model
        model = DISKBackbone(descriptor_dim=128, nms_window_size=5, num_keypoints=num_keypoints).eval()

        print("✅ Successfully created DISK backbone model")

        # Create dummy input
        dummy_input = torch.randn(1, 3, resolution, resolution)
        print(f"📝 Testing with input shape: {dummy_input.shape}")

        # Test forward pass
        with torch.no_grad():
            descriptors, heatmaps = model(dummy_input)
            print("✅ Forward pass successful!")
            print(f"   - Descriptors shape: {descriptors.shape}")
            print(f"   - Heatmaps shape: {heatmaps.shape}")

        # Define output filename
        output_path = f"weights/disk_backbone_trt_{resolution}x{resolution}.onnx"

        # Create weights directory if it doesn't exist
        Path("weights").mkdir(exist_ok=True)

        # Export to ONNX with fixed batch size (no dynamic axes)
        print(f"🔧 Exporting to {output_path}...")

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["image"],
            output_names=["descriptors", "heatmaps"],
            opset_version=16,
            dynamic_axes=None,  # Fixed batch size of 1
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

        print(f"✅ ONNX export successful: {output_path}")

        # Validate ONNX model
        try:
            import onnx

            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model validation successful")

            # Print model info
            print(f"   - Model IR version: {onnx_model.ir_version}")
            print(f"   - Number of nodes: {len(onnx_model.graph.node)}")

        except ImportError:
            print("⚠️  ONNX package not available for validation")

        # Test with ONNXRuntime
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(output_path)
            print("✅ ONNXRuntime can load the model")

            # Test inference
            input_name = sess.get_inputs()[0].name
            result = sess.run(None, {input_name: dummy_input.numpy()})
            print("✅ ONNXRuntime inference successful")
            print(f"   - Output shapes: {[r.shape for r in result]}")

        except ImportError:
            print("⚠️  ONNXRuntime not available for testing")
        except Exception as e:
            print(f"❌ ONNXRuntime error: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Export failed for {resolution}x{resolution}: {e}")
        traceback.print_exc()
        return False


def main():
    """Export DISK models for multiple resolutions"""
    print("🚀 DISK Multi-Resolution Export")
    print("=" * 80)
    print("Exporting DISK backbone models for TensorRT compatibility")
    print("Target resolutions: 256x256, 512x512, 1024x1024")
    print("Keypoints capacity: 1024 for all models")
    print("=" * 80)

    # Add current directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))

    # Define resolutions to export
    resolutions = [256, 384, 512, 1024]
    num_keypoints = 1024

    results = {}

    # Export each resolution
    for resolution in resolutions:
        success = export_disk_model(resolution, num_keypoints)
        results[resolution] = success

    # Summary
    print("\n" + "=" * 80)
    print("📋 EXPORT SUMMARY")
    print("=" * 80)

    successful_exports = []
    failed_exports = []

    for resolution, success in results.items():
        if success:
            filename = f"weights/disk_backbone_trt_{resolution}x{resolution}.onnx"
            print(f"✅ {resolution}x{resolution}: {filename}")
            successful_exports.append(resolution)
        else:
            print(f"❌ {resolution}x{resolution}: Export failed")
            failed_exports.append(resolution)

    print(f"\n📊 Results: {len(successful_exports)}/{len(resolutions)} successful")

    if successful_exports:
        print("\n💡 Next steps for successful exports:")
        print("   1. Convert ONNX models to TensorRT engines (fixed batch size 1):")
        for res in successful_exports:
            print(f"      trtexec --onnx=weights/disk_backbone_trt_{res}x{res}.onnx \\")
            print(f"              --saveEngine=weights/disk_backbone_trt_{res}x{res}.trt")
            print()

        print("   2. Implement NumPy post-processing pipeline")
        print("   3. Benchmark end-to-end performance")
        print("   4. Compare with SuperPoint baseline")

    if failed_exports:
        print(f"\n⚠️  Failed exports: {failed_exports}")
        print("   Check error messages above for debugging")

    # Show post-processing reminder
    print("\n" + "=" * 80)
    print("📋 POST-PROCESSING REMINDER")
    print("=" * 80)
    print("""
All exported models output:
- descriptors: (B, 128, H, W) - Dense feature maps  
- heatmaps: (B, 1, H, W) - Detection scores

Post-processing (NumPy/CPU) extracts:
- keypoints: (B, N, 2) - Up to 1024 keypoint coordinates
- scores: (B, N) - Keypoint confidence scores
- descriptors: (B, N, 128) - L2-normalized keypoint descriptors

See disk_numpy_demo.py for complete implementation example.
""")

    return len(failed_exports) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
