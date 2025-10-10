#!/usr/bin/env python3
"""
Test script to check if DISK can be exported to ONNX and identify potential issues
"""

import sys
import traceback
from pathlib import Path

import torch


def test_disk_export():
    """Test DISK model export to ONNX"""
    # Add the current directory to the path
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        # Try importing DISK from the dynamo models
        from lightglue_dynamo.models.disk import DISK

        print("✅ Successfully imported DISK from lightglue_dynamo.models.disk")

        # Test model creation
        model = DISK(num_keypoints=512).eval()
        print("✅ Successfully created DISK model")

        # Test forward pass
        dummy_input = torch.randn(1, 3, 512, 512)
        print(f"📝 Testing forward pass with input shape: {dummy_input.shape}")

        with torch.no_grad():
            keypoints, scores, descriptors = model(dummy_input)
            print("✅ Forward pass successful!")
            print(f"   - Keypoints shape: {keypoints.shape}")
            print(f"   - Scores shape: {scores.shape}")
            print(f"   - Descriptors shape: {descriptors.shape}")

        # Test ONNX export
        print("\n🔧 Testing ONNX export...")

        # Dynamic axes for ONNX
        dynamic_axes = {
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "keypoints": {0: "batch_size", 1: "num_keypoints"},
            "scores": {0: "batch_size", 1: "num_keypoints"},
            "descriptors": {0: "batch_size", 1: "num_keypoints"},
        }

        output_path = "test_disk_export.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["image"],
                output_names=["keypoints", "scores", "descriptors"],
                opset_version=16,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True,
            )
            print(f"✅ ONNX export successful: {output_path}")

            # Try to load and validate the ONNX model
            import onnx

            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model validation successful")

            # Test with onnxruntime
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

        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            print("📋 Full traceback:")
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📋 Trying alternative import...")
        try:
            from lightglue_onnx import DISK

            print("✅ Successfully imported DISK from lightglue_onnx")

            model = DISK(max_num_keypoints=512).eval()
            print("✅ Successfully created DISK model from lightglue_onnx")

        except Exception as e2:
            print(f"❌ Alternative import also failed: {e2}")
            return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

    print("\n🎉 DISK export test completed!")
    return True


if __name__ == "__main__":
    success = test_disk_export()
    sys.exit(0 if success else 1)
