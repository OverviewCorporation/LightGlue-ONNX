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

        # Create a modified DISK class that loads weights on CPU
        class DISKCPUTest(DISK):
            def __init__(self, **kwargs):
                # Initialize without calling parent __init__ to avoid weight loading
                torch.nn.Module.__init__(self)

                # Set up attributes
                descriptor_dim = kwargs.get("descriptor_dim", 128)
                nms_window_size = kwargs.get("nms_window_size", 5)
                num_keypoints = kwargs.get("num_keypoints", 1024)

                if nms_window_size % 2 != 1:
                    raise ValueError(f"window_size has to be odd, got {nms_window_size}")

                self.descriptor_dim = descriptor_dim
                self.nms_window_size = nms_window_size
                self.num_keypoints = num_keypoints

                # Import and create U-Net
                from lightglue_dynamo.models.disk.unet import Unet

                self.unet = Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64], up=[64, 64, 64, descriptor_dim + 1])

                # Load weights with CPU mapping
                try:
                    state_dict = torch.hub.load_state_dict_from_url(self.url, map_location=torch.device("cpu"))[
                        "extractor"
                    ]
                    self.load_state_dict(state_dict)
                    print("✅ Successfully loaded DISK weights")
                except Exception as e:
                    print(f"⚠️  Could not load pre-trained weights: {e}")
                    print("   Proceeding with random weights for architecture testing...")

        # Test model creation
        model = DISKCPUTest(num_keypoints=512).eval()
        print("✅ Successfully created DISK model")

        # Test forward pass
        dummy_input = torch.randn(1, 3, 512, 512)
        print(f"📝 Testing forward pass with input shape: {dummy_input.shape}")

        with torch.no_grad():
            try:
                keypoints, scores, descriptors = model(dummy_input)
                print("✅ Forward pass successful!")
                print(f"   - Keypoints shape: {keypoints.shape}")
                print(f"   - Scores shape: {scores.shape}")
                print(f"   - Descriptors shape: {descriptors.shape}")
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                print("📋 Full traceback:")
                traceback.print_exc()
                return False

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
                verbose=True,  # Enable verbose output to see what operations are problematic
            )
            print(f"✅ ONNX export successful: {output_path}")

            # Try to load and validate the ONNX model
            try:
                import onnx

                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                print("✅ ONNX model validation successful")

                # Print model info
                print(f"   - Model IR version: {onnx_model.ir_version}")
                print(f"   - Number of nodes: {len(onnx_model.graph.node)}")
                print(f"   - Inputs: {[inp.name for inp in onnx_model.graph.input]}")
                print(f"   - Outputs: {[out.name for out in onnx_model.graph.output]}")

            except ImportError:
                print("⚠️  ONNX not available for model validation")

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
                print("   This indicates TensorRT incompatible operations")

        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            print("📋 Full traceback:")
            traceback.print_exc()

            # Analyze the error for TensorRT compatibility insights
            error_str = str(e).lower()
            if "topk" in error_str:
                print("🔍 TensorRT Issue: topk operation detected - known TensorRT problem")
            if "index" in error_str or "gather" in error_str:
                print("🔍 TensorRT Issue: indexing/gather operations - problematic for TensorRT")
            if "scatter" in error_str:
                print("🔍 TensorRT Issue: scatter operations - not well supported")
            if "floor" in error_str or "ceil" in error_str:
                print("🔍 TensorRT Issue: integer arithmetic operations")

            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

    print("\n🎉 DISK export test completed!")
    return True


if __name__ == "__main__":
    success = test_disk_export()
    print("\n" + "=" * 80)
    if success:
        print("✅ DISK CAN BE EXPORTED TO ONNX!")
        print("📋 Next step: Test conversion to TensorRT")
        print("⚠️  However, expect TensorRT compatibility issues due to:")
        print("   - topk operations for keypoint selection")
        print("   - Advanced indexing for descriptor sampling")
        print("   - Dynamic tensor operations")
    else:
        print("❌ DISK CANNOT BE EXPORTED TO ONNX")
        print("   TensorRT export is not feasible")

    sys.exit(0 if success else 1)
