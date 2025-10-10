#!/usr/bin/env python3
"""
Export DISK U-Net backbone without post-processing for TensorRT compatibility
"""

import sys
import traceback
from pathlib import Path

import torch


def export_disk_backbone():
    """Export DISK U-Net backbone without keypoint selection post-processing"""
    # Add the current directory to the path
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from lightglue_dynamo.models.disk import DISK

        print("✅ Successfully imported DISK from lightglue_dynamo.models.disk")

        # Create a DISK backbone class that only exports the U-Net
        class DISKBackbone(DISK):
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

            def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """
                Forward pass that only runs the U-Net backbone
                Returns:
                    descriptors: (B, 128, H, W) - Dense descriptor maps
                    heatmaps: (B, 1, H, W) - Dense detection heatmaps
                """
                unet_output = self.unet(image)
                descriptors = unet_output[:, : self.descriptor_dim]  # (B, 128, H, W)
                heatmaps = unet_output[:, self.descriptor_dim :]  # (B, 1, H, W)

                return descriptors, heatmaps

        # Test model creation
        model = DISKBackbone(descriptor_dim=128).eval()
        print("✅ Successfully created DISK backbone model")

        # Test forward pass
        dummy_input = torch.randn(1, 3, 512, 512)
        print(f"📝 Testing forward pass with input shape: {dummy_input.shape}")

        with torch.no_grad():
            descriptors, heatmaps = model(dummy_input)
            print("✅ Forward pass successful!")
            print(f"   - Descriptors shape: {descriptors.shape}")
            print(f"   - Heatmaps shape: {heatmaps.shape}")

        # Test ONNX export
        print("\n🔧 Testing ONNX export of DISK backbone...")

        # Dynamic axes for ONNX - much simpler now!
        dynamic_axes = {
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "descriptors": {0: "batch_size", 2: "height", 3: "width"},
            "heatmaps": {0: "batch_size", 2: "height", 3: "width"},
        }

        output_path = "disk_backbone_trt_compatible.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["image"],
                output_names=["descriptors", "heatmaps"],
                opset_version=16,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True,
                verbose=False,  # Less verbose since we expect this to work
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
                return False

        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            print("📋 Full traceback:")
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

    return True


def show_numpy_postprocessing_example():
    """Show how the post-processing can be done in NumPy"""
    print("\n" + "=" * 80)
    print("📋 POST-PROCESSING IN NUMPY")
    print("=" * 80)
    print("""
The exported DISK backbone outputs:
- descriptors: (B, 128, H, W) - Dense feature maps
- heatmaps: (B, 1, H, W) - Detection scores at each pixel

Post-processing pipeline in NumPy/CPU:

```python
import numpy as np
from scipy.ndimage import maximum_filter

def disk_postprocess(descriptors, heatmaps, num_keypoints=1024, nms_size=5):
    '''
    Post-process DISK outputs to extract keypoints
    
    Args:
        descriptors: (B, 128, H, W) numpy array
        heatmaps: (B, 1, H, W) numpy array  
        num_keypoints: number of keypoints to extract
        nms_size: NMS window size
    
    Returns:
        keypoints: (B, N, 2) - keypoint coordinates
        scores: (B, N) - keypoint scores
        desc: (B, N, 128) - keypoint descriptors
    '''
    B, C, H, W = heatmaps.shape
    
    # NMS using scipy maximum filter
    nms_heatmaps = maximum_filter(heatmaps, size=nms_size, mode='constant')
    mask = (heatmaps == nms_heatmaps)
    
    results_kpts = []
    results_scores = []
    results_desc = []
    
    for b in range(B):
        # Get valid keypoints after NMS
        valid_mask = mask[b, 0]
        scores_flat = heatmaps[b, 0][valid_mask]
        
        # Get coordinates
        y_coords, x_coords = np.where(valid_mask)
        
        # Select top-k keypoints
        if len(scores_flat) > num_keypoints:
            top_indices = np.argpartition(scores_flat, -num_keypoints)[-num_keypoints:]
            top_indices = top_indices[np.argsort(scores_flat[top_indices])[::-1]]
            
            x_coords = x_coords[top_indices]
            y_coords = y_coords[top_indices]
            scores_flat = scores_flat[top_indices]
        
        # Stack coordinates (x, y format)
        keypoints = np.stack([x_coords, y_coords], axis=1)
        
        # Sample descriptors at keypoint locations
        desc = descriptors[b, :, y_coords, x_coords].T  # (N, 128)
        desc = desc / np.linalg.norm(desc, axis=1, keepdims=True)  # L2 normalize
        
        results_kpts.append(keypoints)
        results_scores.append(scores_flat)
        results_desc.append(desc)
    
    return results_kpts, results_scores, results_desc
```

Benefits of this approach:
✅ TensorRT-compatible backbone (no problematic operations)
✅ Flexible post-processing (easy to modify NMS parameters)
✅ Better CPU utilization (post-processing doesn't need GPU)
✅ Easier debugging and validation
✅ Can optimize post-processing independently
""")


if __name__ == "__main__":
    print("🚀 Exporting DISK U-Net Backbone for TensorRT")
    print("=" * 80)

    success = export_disk_backbone()

    if success:
        show_numpy_postprocessing_example()

        print("\n" + "=" * 80)
        print("✅ SUCCESS: DISK BACKBONE EXPORTED!")
        print("=" * 80)
        print("📋 Summary:")
        print("   • Exported: disk_backbone_trt_compatible.onnx")
        print("   • Contains: U-Net feature extractor only")
        print("   • Outputs: Dense descriptors + heatmaps")
        print("   • TensorRT: Should be fully compatible!")
        print("   • Post-processing: Implement in NumPy (see example above)")
        print("\n💡 Next steps:")
        print("   1. Convert disk_backbone_trt_compatible.onnx to TensorRT")
        print("   2. Implement NumPy post-processing pipeline")
        print("   3. Benchmark end-to-end performance")

    else:
        print("\n❌ FAILED: Could not export DISK backbone")

    sys.exit(0 if success else 1)
