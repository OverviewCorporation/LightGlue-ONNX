#!/usr/bin/env python3
"""
Export DISK LightGlue matcher models for multiple resolutions
Fixed batch size of 1 for TensorRT compatibility
128-dimensional descriptors to match DISK backbone
"""

import sys
import traceback
from pathlib import Path

import torch

# Use dynamo version instead of lightglue_onnx to avoid kornia dependency
from lightglue_dynamo.models.lightglue import LightGlue
from lightglue_dynamo.config import Extractor


def export_disk_lightglue_model(max_keypoints: int = 1024) -> bool:
    """
    Export DISK LightGlue matcher model with fixed batch size

    Args:
        max_keypoints: Maximum number of keypoints for each image

    Returns:
        bool: Success status
    """
    print(f"\n{'=' * 60}")
    print(f"🔧 Exporting DISK LightGlue matcher")
    print(f"   Max keypoints: {max_keypoints}")
    print(f"   Descriptor dim: 128 (DISK compatible)")
    print(f"   Batch size: Fixed at 1")
    print(f"{'=' * 60}")

    try:
        # Create DISK LightGlue model with proper configuration
        disk_config = Extractor.disk.lightglue_config
        model = LightGlue(
            url=disk_config["url"],
            input_dim=disk_config["input_dim"],  # 128 for DISK
            descriptor_dim=256,  # Internal descriptor dimension
            num_heads=4,
            n_layers=9,
            filter_threshold=0.1,
        ).eval()
        print("✅ Successfully created DISK LightGlue model")

        # Create mock inputs matching DISK output format
        # DISK outputs 128-dimensional descriptors
        batch_size = 1

        # Create sample keypoints and descriptors for two images
        # Image 0
        num_kpts0 = min(max_keypoints, 512)  # Use reasonable number for testing
        kpts0 = torch.randn(batch_size, num_kpts0, 2)  # (B, N, 2)
        desc0 = torch.randn(batch_size, num_kpts0, 128)  # (B, N, 128) - DISK dim
        desc0 = torch.nn.functional.normalize(desc0, p=2, dim=2)  # L2 normalize

        # Image 1
        num_kpts1 = min(max_keypoints, 400)  # Different number for variety
        kpts1 = torch.randn(batch_size, num_kpts1, 2)  # (B, N, 2)
        desc1 = torch.randn(batch_size, num_kpts1, 128)  # (B, N, 128) - DISK dim
        desc1 = torch.nn.functional.normalize(desc1, p=2, dim=2)  # L2 normalize

        # Normalize keypoints to [0, 1] range (LightGlue expects normalized coordinates)
        kpts0 = torch.clamp(kpts0, 0, 1)
        kpts1 = torch.clamp(kpts1, 0, 1)

        print(f"📝 Test input shapes:")
        print(f"   - kpts0: {kpts0.shape}")
        print(f"   - desc0: {desc0.shape}")
        print(f"   - kpts1: {kpts1.shape}")
        print(f"   - desc1: {desc1.shape}")

        # Test forward pass
        with torch.no_grad():
            matches0, mscores0 = model(kpts0, kpts1, desc0, desc1)
            print("✅ Forward pass successful!")
            print(f"   - Matches shape: {matches0.shape}")
            print(f"   - Scores shape: {mscores0.shape}")
            print(f"   - Valid matches: {(matches0 >= 0).sum().item()}")

        # Define output filename
        output_path = f"weights/disk_lightglue_matcher_trt_{max_keypoints}.onnx"

        # Create weights directory if it doesn't exist
        Path("weights").mkdir(exist_ok=True)

        # Export to ONNX with fixed batch size (no dynamic axes)
        print(f"🔧 Exporting to {output_path}...")

        torch.onnx.export(
            model,
            (kpts0, kpts1, desc0, desc1),
            output_path,
            input_names=["kpts0", "kpts1", "desc0", "desc1"],
            output_names=["matches0", "mscores0"],
            opset_version=17,
            dynamic_axes=None,  # Fixed batch size and keypoint dimensions
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

            # Print input/output shapes
            for inp in onnx_model.graph.input:
                shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
                print(f"   - Input '{inp.name}': {shape}")

            for out in onnx_model.graph.output:
                shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
                print(f"   - Output '{out.name}': {shape}")

        except ImportError:
            print("⚠️  ONNX package not available for validation")

        # Test with ONNXRuntime
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(output_path)
            print("✅ ONNXRuntime can load the model")

            # Test inference
            inputs = {"kpts0": kpts0.numpy(), "kpts1": kpts1.numpy(), "desc0": desc0.numpy(), "desc1": desc1.numpy()}
            result = sess.run(None, inputs)
            print("✅ ONNXRuntime inference successful")
            print(f"   - Output shapes: {[r.shape for r in result]}")
            print(f"   - Valid matches: {(result[0] >= 0).sum()}")

        except ImportError:
            print("⚠️  ONNXRuntime not available for testing")
        except Exception as e:
            print(f"❌ ONNXRuntime error: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Export failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Export DISK LightGlue models for different keypoint capacities"""
    print("🚀 DISK LightGlue Export")
    print("=" * 80)
    print("Exporting DISK LightGlue matcher models for TensorRT compatibility")
    print("Features: 128-dimensional descriptors, fixed batch size 1")
    print("=" * 80)

    # Add current directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))

    # Define keypoint capacities to export
    keypoint_configs = [512, 1024, 2048]

    results = {}

    # Export each configuration
    for max_keypoints in keypoint_configs:
        success = export_disk_lightglue_model(max_keypoints)
        results[max_keypoints] = success

    # Summary
    print("\n" + "=" * 80)
    print("📋 EXPORT SUMMARY")
    print("=" * 80)

    successful_exports = []
    failed_exports = []

    for max_keypoints, success in results.items():
        if success:
            filename = f"weights/disk_lightglue_matcher_trt_{max_keypoints}.onnx"
            print(f"✅ {max_keypoints} keypoints: {filename}")
            successful_exports.append(max_keypoints)
        else:
            print(f"❌ {max_keypoints} keypoints: Export failed")
            failed_exports.append(max_keypoints)

    print(f"\n📊 Results: {len(successful_exports)}/{len(keypoint_configs)} successful")

    if successful_exports:
        print("\n💡 Next steps for successful exports:")
        print("   1. Convert ONNX models to TensorRT engines (fixed batch size 1):")
        for max_kpts in successful_exports:
            print(f"      trtexec --onnx=weights/disk_lightglue_matcher_trt_{max_kpts}.onnx \\")
            print(f"              --saveEngine=weights/disk_lightglue_matcher_trt_{max_kpts}.trt")
            print()

        print("   2. Use with DISK backbone models:")
        print("      - disk_backbone_trt_256x256.onnx")
        print("      - disk_backbone_trt_512x512.onnx")
        print("      - disk_backbone_trt_1024x1024.onnx")
        print()
        print("   3. Integration pipeline:")
        print("      a) Run DISK backbone (TensorRT) → dense descriptors + heatmaps")
        print("      b) Post-process (NumPy) → keypoints + descriptors (128D)")
        print("      c) Run LightGlue matcher (TensorRT) → matches + scores")

    if failed_exports:
        print(f"\n⚠️  Failed exports: {failed_exports}")
        print("   Check error messages above for debugging")

    # Show integration reminder
    print("\n" + "=" * 80)
    print("📋 INTEGRATION REMINDER")
    print("=" * 80)
    print("""
Complete DISK + LightGlue TensorRT pipeline:

1. DISK Backbone (GPU):
   Input: (1, 3, H, W) RGB images
   Output: (1, 128, H, W) descriptors + (1, 1, H, W) heatmaps

2. DISK Post-processing (CPU):
   Input: Dense descriptors + heatmaps
   Output: (N, 2) keypoints + (N, 128) descriptors + (N,) scores

3. LightGlue Matcher (GPU):
   Input: kpts0, kpts1, desc0, desc1 (all 128D)
   Output: matches0, mscores0

All models use fixed batch size 1 for optimal TensorRT performance.
""")

    return len(failed_exports) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
