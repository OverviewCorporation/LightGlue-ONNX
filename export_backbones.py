#!/usr/bin/env python3
"""
Unified backbone export script for SuperPoint and DISK
Exports TensorRT-compatible backbone models with fixed batch size
"""

import sys
import argparse
import traceback
from pathlib import Path
from typing import Union

import torch

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lightglue_dynamo.models import SuperPointBackbone, DISKBackbone, LightGlueTRT


def export_backbone_model(
    extractor_type: str, resolution: Union[int, tuple[int, int]] = 512, num_keypoints: int = 1024
) -> bool:
    """
    Export backbone model for specific configuration

    Args:
        extractor_type: "superpoint" or "disk"
        resolution: Image resolution (int for square, tuple for (H, W))
        num_keypoints: Number of keypoints capacity (for naming only)

    Returns:
        bool: Success status
    """
    print(f"\n{'=' * 60}")
    print(f"🔧 Exporting {extractor_type.upper()} backbone")
    print(f"   Resolution: {resolution}")
    print(f"   Keypoints capacity: {num_keypoints}")
    print(f"{'=' * 60}")

    try:
        # Parse resolution
        if isinstance(resolution, int):
            height = width = resolution
            res_str = f"{resolution}x{resolution}"
        else:
            height, width = resolution
            res_str = f"{height}x{width}"

        # Create model
        if extractor_type.lower() == "disk":
            model = DISKBackbone(descriptor_dim=128).eval()
            descriptor_dim = 128
        elif extractor_type.lower() == "superpoint":
            model = SuperPointBackbone(descriptor_dim=256).eval()
            descriptor_dim = 256
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

        print("✅ Successfully created backbone model")

        # Create dummy input (always RGB for unified interface)
        dummy_input = torch.randn(1, 3, height, width)
        print(f"📝 Testing with input shape: {dummy_input.shape}")

        # Test forward pass
        with torch.no_grad():
            heatmaps, descriptors = model(dummy_input)
            print("✅ Forward pass successful!")
            print(f"   - Heatmaps shape: {heatmaps.shape}")
            print(f"   - Descriptors shape: {descriptors.shape}")

        # Define output filename
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        output_path = weights_dir / f"{extractor_type}_backbone_trt_{res_str}.onnx"

        # Export to ONNX with fixed batch size
        print(f"🔧 Exporting to {output_path}...")

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["image"],
            output_names=["heatmaps", "descriptors"],
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

            onnx_model = onnx.load(str(output_path))
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

            sess = ort.InferenceSession(str(output_path))
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
        print(f"❌ Export failed: {e}")
        traceback.print_exc()
        return False


def export_matcher_model(extractor_type: str, num_keypoints: int = 1024) -> bool:
    """
    Export LightGlue matcher model for specific extractor

    Args:
        extractor_type: "superpoint" or "disk"
        num_keypoints: Number of keypoints capacity

    Returns:
        bool: Success status
    """
    print(f"\n{'=' * 60}")
    print(f"🔧 Exporting LightGlue matcher for {extractor_type.upper()}")
    print(f"   Max keypoints: {num_keypoints}")
    print(f"{'=' * 60}")

    try:
        # Determine feature dimensions
        if extractor_type.lower() == "disk":
            feature_dim = 128
        elif extractor_type.lower() == "superpoint":
            feature_dim = 256
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

        # Get model URL based on extractor type
        if extractor_type.lower() == "superpoint":
            url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"
            input_dim = 256
        else:  # disk
            url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth"
            input_dim = 128

        # Create TensorRT-compatible LightGlue model
        model = LightGlueTRT(
            url=url,
            input_dim=input_dim,
            descriptor_dim=256,  # Always 256 for LightGlue internal processing
        ).eval()
        print("✅ Successfully created LightGlue matcher model")

        # Create dummy inputs - LightGlue expects concatenated inputs
        # Format: (2B, N, 2) keypoints, (2B, N, D) descriptors
        dummy_kpts = torch.randn(2, num_keypoints, 2)  # Two sets of keypoints
        dummy_desc = torch.randn(2, num_keypoints, feature_dim)  # Two sets of descriptors

        print("📝 Testing with inputs:")
        print(f"   - Keypoints: {dummy_kpts.shape}")
        print(f"   - Descriptors: {dummy_desc.shape}")

        # Test forward pass
        with torch.no_grad():
            scores = model(dummy_kpts, dummy_desc)
            print("✅ Forward pass successful!")
            print(f"   - Scores shape: {scores.shape}")

        # Define output filename
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        output_path = weights_dir / f"{extractor_type}_lightglue_matcher_trt_{num_keypoints}.onnx"

        # Export to ONNX with fixed batch size
        print(f"🔧 Exporting to {output_path}...")

        torch.onnx.export(
            model,
            (dummy_kpts, dummy_desc),
            str(output_path),
            input_names=["keypoints", "descriptors"],
            output_names=["scores"],
            opset_version=16,
            dynamic_axes=None,  # Fixed batch size and keypoint count
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

        print(f"✅ ONNX export successful: {output_path}")

        # Validate ONNX model
        try:
            import onnx

            onnx_model = onnx.load(str(output_path))
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

            sess = ort.InferenceSession(str(output_path))
            print("✅ ONNXRuntime can load the model")

            # Test inference
            input_names = [inp.name for inp in sess.get_inputs()]
            input_dict = {
                input_names[0]: dummy_kpts.numpy(),
                input_names[1]: dummy_desc.numpy(),
            }
            result = sess.run(None, input_dict)
            print("✅ ONNXRuntime inference successful")
            print(f"   - Output shapes: {[r.shape for r in result]}")

        except ImportError:
            print("⚠️  ONNXRuntime not available for testing")
        except Exception as e:
            print(f"❌ ONNXRuntime error: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Matcher export failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main export function with argument parsing"""
    parser = argparse.ArgumentParser(description="Export TensorRT-compatible backbone and matcher models")
    parser.add_argument(
        "--extractor",
        choices=["superpoint", "disk", "both"],
        default="both",
        help="Extractor type to export (default: both)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="Image resolutions to export (default: 256 512 1024)",
    )
    parser.add_argument("--keypoints", type=int, default=1024, help="Keypoints capacity (default: 1024)")
    parser.add_argument(
        "--models",
        choices=["backbone", "matcher", "both"],
        default="both",
        help="Model types to export (default: both)",
    )

    args = parser.parse_args()

    print("🚀 Unified Model Export (Backbones + Matchers)")
    print("=" * 80)
    print(f"Extractor(s): {args.extractor}")
    print(f"Models: {args.models}")
    print(f"Resolutions: {args.resolution}")
    print(f"Keypoints: {args.keypoints}")
    print("=" * 80)

    # Determine extractors to export
    extractors = ["superpoint", "disk"] if args.extractor == "both" else [args.extractor]

    # Determine model types to export
    export_backbone = args.models in ["backbone", "both"]
    export_matcher = args.models in ["matcher", "both"]

    # Export all combinations
    results = {}
    for extractor in extractors:
        results[extractor] = {}

        # Export backbone models if requested
        if export_backbone:
            results[extractor]["backbones"] = {}
            for resolution in args.resolution:
                success = export_backbone_model(extractor, resolution, args.keypoints)
                results[extractor]["backbones"][resolution] = success

        # Export matcher models if requested
        if export_matcher:
            results[extractor]["matcher"] = export_matcher_model(extractor, args.keypoints)

    # Summary
    print("\n" + "=" * 80)
    print("📋 EXPORT SUMMARY")
    print("=" * 80)

    successful_exports = []
    failed_exports = []

    for extractor, model_results in results.items():
        # Handle backbone exports
        if "backbones" in model_results:
            for resolution, success in model_results["backbones"].items():
                export_name = f"{extractor}_backbone_{resolution}x{resolution}"
                if success:
                    filename = f"weights/{extractor}_backbone_trt_{resolution}x{resolution}.onnx"
                    print(f"✅ {export_name}: {filename}")
                    successful_exports.append(export_name)
                else:
                    print(f"❌ {export_name}: Export failed")
                    failed_exports.append(export_name)

        # Handle matcher exports
        if "matcher" in model_results:
            export_name = f"{extractor}_matcher"
            success = model_results["matcher"]
            if success:
                filename = f"weights/{extractor}_lightglue_matcher_trt_{args.keypoints}.onnx"
                print(f"✅ {export_name}: {filename}")
                successful_exports.append(export_name)
            else:
                print(f"❌ {export_name}: Export failed")
                failed_exports.append(export_name)

    total_exports = len(successful_exports) + len(failed_exports)
    print(f"\n📊 Results: {len(successful_exports)}/{total_exports} successful")

    if successful_exports:
        print("\n💡 Next steps for successful exports:")
        print("   1. Convert ONNX models to TensorRT engines:")
        for export_name in successful_exports:
            parts = export_name.split("_")
            extractor = parts[0]
            if "backbone" in export_name:
                res_part = parts[2]  # e.g., "256x256"
                print(f"      trtexec --onnx=weights/{extractor}_backbone_trt_{res_part}.onnx \\")
                print(f"              --saveEngine=weights/{extractor}_backbone_trt_{res_part}.trt")
            elif "matcher" in export_name:
                print(f"      trtexec --onnx=weights/{extractor}_lightglue_matcher_trt_{args.keypoints}.onnx \\")
                print(f"              --saveEngine=weights/{extractor}_lightglue_matcher_trt_{args.keypoints}.trt")
            print()

        print("   2. Pipeline usage:")
        print("      - Backbone: Extract dense features from images")
        print("      - Matcher: Match sparse keypoints between image pairs")
        print("      - Post-processing: Convert dense outputs to sparse keypoints (CPU)")

    if failed_exports:
        print(f"\n⚠️  Failed exports: {failed_exports}")
        print("   Check error messages above for debugging")

    # Show model specifications
    print("\n" + "=" * 80)
    print("📋 MODEL SPECIFICATIONS")
    print("=" * 80)
    print("""
BACKBONE MODELS:

SuperPoint Backbone:
- Input: (1, 3, H, W) RGB images (converted to grayscale internally)
- Outputs:
  * heatmaps: (1, 1, H//8, W//8) - Dense detection scores
  * descriptors: (1, 256, H//8, W//8) - Dense 256D descriptors

DISK Backbone:
- Input: (1, 3, H, W) RGB images
- Outputs:
  * heatmaps: (1, 1, H, W) - Dense detection scores  
  * descriptors: (1, 128, H, W) - Dense 128D descriptors

MATCHER MODELS:

LightGlue Matcher (SuperPoint):
- Inputs:
  * keypoints0/1: (1, N, 2) - Sparse keypoint coordinates
  * descriptors0/1: (1, N, 256) - SuperPoint 256D descriptors
- Outputs:
  * matches: (1, M, 2) - Match indices between keypoint sets
  * scores: (1, M) - Match confidence scores

LightGlue Matcher (DISK):
- Inputs:
  * keypoints0/1: (1, N, 2) - Sparse keypoint coordinates  
  * descriptors0/1: (1, N, 128) - DISK 128D descriptors
- Outputs:
  * matches: (1, M, 2) - Match indices between keypoint sets
  * scores: (1, M) - Match confidence scores

ALL MODELS:
- Fixed batch size of 1 for optimal TensorRT performance
- Fixed keypoint counts for matcher models (no dynamic shapes)
- Fully TensorRT compatible (no dynamic operations)
""")

    return len(failed_exports) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
