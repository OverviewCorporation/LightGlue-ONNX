# DISK TensorRT 8.2 Compatibility Analysis

## Overview

This document analyzes the feasibility of exporting DISK feature extractor to ONNX and subsequently to TensorRT 8.2, based on the architectural components and operations used in the model.

## DISK Architecture Analysis

### Core Components

1. **U-Net Backbone**: DISK uses a U-Net architecture with encoder-decoder structure
2. **Feature Extractor**: Outputs descriptors (128-dim) and heatmaps for keypoint detection
3. **NMS Processing**: Non-maximum suppression for keypoint selection
4. **Keypoint Sampling**: Bilinear sampling of descriptors at keypoint locations

### Potential TensorRT Compatibility Issues

#### 1. InstanceNorm2d ✅ LIKELY COMPATIBLE
- **Operation**: Used in Conv blocks for normalization
- **TRT 8.2 Support**: InstanceNorm is generally well-supported in TensorRT
- **Risk Level**: LOW - Should work with proper ONNX export

#### 2. PReLU Activation ✅ COMPATIBLE  
- **Operation**: Parametric ReLU activation function
- **TRT 8.2 Support**: PReLU is natively supported in TensorRT
- **Risk Level**: LOW

#### 3. F.interpolate (Bilinear Upsampling) ⚠️ MODERATE RISK
- **Operation**: Used in TrivialUpsample for 2x upsampling
- **Configuration**: `mode="bilinear", align_corners=False`
- **TRT 8.2 Support**: Generally supported but can be problematic with dynamic shapes
- **Potential Issues**: 
  - May require specific ONNX opset versions
  - Dynamic shape handling might need careful configuration
- **Risk Level**: MODERATE

#### 4. F.avg_pool2d ✅ COMPATIBLE
- **Operation**: Average pooling for downsampling
- **TRT 8.2 Support**: Fully supported
- **Risk Level**: LOW

#### 5. torch.topk ❌ HIGH RISK
- **Operation**: Used in `heatmap_to_keypoints` for selecting top-K keypoints
- **TRT 8.2 Support**: PROBLEMATIC - topk can be challenging for TensorRT
- **Issues**:
  - Dynamic K values may not be supported
  - Output indices might require special handling
  - Could cause significant performance bottlenecks
- **Risk Level**: HIGH

#### 6. Complex Indexing Operations ❌ HIGH RISK
- **Operations**: 
  - `floor_divide`, `%` (modulo), tensor slicing
  - Advanced indexing: `descriptors[(batches, keypoints[:, :, 1], keypoints[:, :, 0])]`
- **TRT 8.2 Support**: VERY PROBLEMATIC
- **Issues**:
  - Advanced indexing with dynamic indices is poorly supported
  - Integer operations (floor_divide, modulo) can be problematic
  - Gather operations with computed indices are complex
- **Risk Level**: VERY HIGH

#### 7. torch.where ⚠️ MODERATE RISK
- **Operation**: Conditional selection in NMS
- **TRT 8.2 Support**: Supported but can be inefficient
- **Risk Level**: MODERATE

#### 8. Dynamic Tensor Operations ❌ HIGH RISK
- **Operations**: Tensor reshaping, slicing with runtime-computed indices
- **Issues**:
  - `heatmap.reshape(b, h * w).topk(n)` - dynamic reshaping
  - Runtime index computation for keypoint extraction
- **Risk Level**: HIGH

## Comparison with SuperPoint

| Component | SuperPoint | DISK | TRT Compatibility |
|-----------|------------|------|------------------|
| Backbone | VGG-style CNN | U-Net | DISK more complex |
| Normalization | BatchNorm | InstanceNorm | Both OK |
| Upsampling | None/Simple | Bilinear interpolation | SuperPoint better |
| NMS | Simple | Complex with topk | SuperPoint simpler |
| Keypoint Extraction | Grid-based | Advanced indexing | SuperPoint much better |
| Dynamic Operations | Minimal | Extensive | SuperPoint much better |

## Key Challenges for DISK → TensorRT

### 1. Keypoint Selection Pipeline
The most problematic part is the keypoint selection in `heatmap_to_keypoints`:

```python
# These operations are TensorRT-unfriendly:
top_scores, top_indices = heatmap.reshape(b, h * w).topk(n)  # topk with dynamic shapes
top_indices = top_indices.unsqueeze(2).floor_divide(...)     # integer arithmetic
top_keypoints = top_indices.flip(2)                          # tensor manipulation
```

### 2. Descriptor Sampling
The descriptor extraction uses advanced indexing:

```python
# This is very problematic for TensorRT:
descriptors = descriptors[(batches, keypoints[:, :, 1], keypoints[:, :, 0])]
```

This requires:
- Dynamic indexing with computed coordinates
- Multi-dimensional gather operations
- Runtime shape computations

### 3. U-Net Architecture Complexity
While U-Net itself can work with TensorRT, DISK's specific implementation has:
- Multiple skip connections requiring careful concatenation
- Bilinear upsampling operations
- Complex tensor routing

## Potential Solutions

### Option 1: Architecture Modifications ⚠️ SIGNIFICANT EFFORT
- Replace topk with fixed-size operations
- Implement TensorRT-friendly NMS
- Simplify keypoint extraction pipeline
- Pre-allocate tensor shapes where possible

### Option 2: Two-Stage Export 🔄 COMPLEX
- Export feature extraction (U-Net) to TensorRT
- Keep keypoint selection/NMS on CPU/GPU separately
- Would require pipeline modifications

### Option 3: Alternative Implementations 💡 RECOMMENDED
- Use existing TensorRT-compatible NMS implementations
- Replace advanced indexing with simpler operations
- Consider DISK-inspired architecture with TensorRT constraints

## Recommendation: ❌ NOT RECOMMENDED

**DISK export to TensorRT 8.2 is NOT recommended** for the following reasons:

1. **High Complexity**: Multiple TensorRT-incompatible operations
2. **Dynamic Operations**: Extensive use of runtime shape computations
3. **Advanced Indexing**: Core functionality relies on operations poorly supported by TensorRT
4. **Development Effort**: Would require significant architecture modifications
5. **Performance Risk**: Even if exported, performance might be poor due to fallback operations

## Alternative Recommendation: ✅ STICK WITH SUPERPOINT

SuperPoint is much better suited for TensorRT because:
- Simpler architecture with CNN backbone
- Minimal dynamic operations
- TensorRT-friendly NMS can be implemented
- Already proven to work with your TensorRT pipeline
- Much lower development and maintenance overhead

## Conclusion

While DISK is a powerful feature extractor, its architecture is fundamentally incompatible with TensorRT 8.2's limitations. The extensive use of dynamic indexing, topk operations, and complex tensor manipulations makes it a poor candidate for TensorRT optimization.

**Recommendation**: Continue using SuperPoint for your TensorRT pipeline, as it provides excellent performance with much lower complexity and proven TensorRT compatibility.