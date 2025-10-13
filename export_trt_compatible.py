#!/usr/bin/env python3
"""
Simple script to export TensorRT-compatible LightGlue models
Supports both SuperPoint and DISK extractors
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import from lightglue_dynamo
sys.path.insert(0, str(Path(__file__).parent))

import torch
import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from lightglue_dynamo.models import SuperPoint, DISK
from lightglue_dynamo.models.lightglue_trt import LightGlueTRT
from lightglue_dynamo.config import Extractor


def export_trt_compatible_models(extractor_name="superpoint"):
    """Export TensorRT-compatible extractor and LightGlue matcher"""

    print(f"🚀 Exporting TensorRT-compatible {extractor_name.upper()} models...")

    # Configuration
    if extractor_name.lower() == "disk":
        extractor_type = Extractor.disk
        extractor = DISK(num_keypoints=1024).eval()
        input_channels = 3  # DISK uses RGB
        descriptor_dim = 128  # DISK uses 128D descriptors
    else:
        extractor_type = Extractor.superpoint
        extractor = SuperPoint(num_keypoints=1024).eval()
        input_channels = 1  # SuperPoint uses grayscale
        descriptor_dim = 256  # SuperPoint uses 256D descriptors

    num_keypoints = 1024
    batch_size = 1
    height = 512
    width = 512
    opset = 16  # Use opset 16 for grid_sampler support

    # Create matcher with proper configuration
    matcher = LightGlueTRT(**extractor_type.lightglue_config).eval()

    # Set output paths
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    extractor_output = weights_dir / f"{extractor_name}_extractor_trt.onnx"
    matcher_output = weights_dir / f"{extractor_name}_lightglue_matcher_trt.onnx"

    print(f"📁 Output directory: {weights_dir}")

    # Export extractor
    print(f"🔧 Exporting {extractor_name.upper()} extractor...")

    # Create dummy input with correct channels
    if extractor_name.lower() == "disk":
        dummy_input = {"image": torch.zeros(batch_size, 3, height, width)}  # RGB for DISK
    else:
        dummy_input = {"image": torch.zeros(batch_size, 1, height, width)}  # Grayscale for SuperPoint

    extractor_dynamic_axes = {
        "image": {0: "batch_size", 2: "height", 3: "width"},
        "keypoints": {0: "batch_size", 1: "num_keypoints"},
        "scores": {0: "batch_size", 1: "num_keypoints"},
        "descriptors": {0: "batch_size", 1: "descriptor_dim", 2: "num_keypoints"},
    }

    torch.onnx.export(
        extractor,
        (dummy_input,),
        str(extractor_output),
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
        opset_version=opset,
        dynamic_axes=extractor_dynamic_axes,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        export_params=True,
    )

    # Validate and optimize extractor
    onnx.checker.check_model(str(extractor_output))
    onnx.save_model(
        SymbolicShapeInference.infer_shapes(onnx.load_model(str(extractor_output)), auto_merge=True),
        str(extractor_output),
    )
    print(f"✅ Extractor exported: {extractor_output}")

    # Export matcher
    print("🔧 Exporting LightGlue matcher...")

    # Create dummy features for matcher
    with torch.no_grad():
        dummy_kpts, dummy_scores, dummy_desc = extractor(dummy_input["image"])

    # Normalize keypoints: 2 * keypoints / size - 1
    size = torch.tensor([width, height], device=dummy_kpts.device, dtype=torch.float32)
    dummy_kpts_normalized = 2 * dummy_kpts.float() / size - 1

    # Create interleaved batch for LightGlue
    dummy_kpts_interleaved = torch.cat([dummy_kpts_normalized, dummy_kpts_normalized], dim=0)
    dummy_desc_interleaved = torch.cat([dummy_desc, dummy_desc], dim=0)

    matcher_dynamic_axes = {
        "keypoints": {0: "batch_x2", 1: "num_keypoints"},
        "descriptors": {0: "batch_x2", 1: "num_keypoints"},
        "matches": {0: "num_matches"},
        "mscores": {0: "num_matches"},
    }

    torch.onnx.export(
        matcher,
        (dummy_kpts_interleaved, dummy_desc_interleaved),
        str(matcher_output),
        input_names=["keypoints", "descriptors"],
        output_names=["matches", "mscores"],
        opset_version=opset,
        dynamic_axes=matcher_dynamic_axes,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        export_params=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )

    # Validate and optimize matcher
    onnx.checker.check_model(str(matcher_output))
    onnx.save_model(
        SymbolicShapeInference.infer_shapes(onnx.load_model(str(matcher_output)), auto_merge=True), str(matcher_output)
    )
    print(f"✅ Matcher exported: {matcher_output}")

    print("\n🎉 Export complete!")
    print("📋 Summary:")
    print(f"  • Extractor: {extractor_output}")
    print(f"  • Matcher: {matcher_output}")
    print(f"  • ONNX Opset: {opset}")
    print("\n💡 These models should be compatible with TensorRT conversion.")
    print("💡 Key improvements:")
    print("  • Custom LayerNorm implementation for TensorRT compatibility")
    print("  • Lower ONNX opset version (14) for better support")
    print("  • Explicit float32 operations in normalization")
    print("  • Avoided INT64 operations that cause TensorRT issues")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export TensorRT-compatible models")
    parser.add_argument(
        "--extractor",
        choices=["superpoint", "disk"],
        default="superpoint",
        help="Extractor type to export (default: superpoint)",
    )
    args = parser.parse_args()

    export_trt_compatible_models(args.extractor)
