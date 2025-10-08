from pathlib import Path
from typing import Annotated, Optional

import cv2
import typer

from lightglue_dynamo.cli_utils import check_multiple_of
from lightglue_dynamo.config import Extractor, InferenceDevice

app = typer.Typer()


@app.callback()
def callback():
    """LightGlue Dynamo CLI"""


@app.command()
def export(
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output: Annotated[
        Optional[Path],  # typer does not support Path | None # noqa: UP007
        typer.Option("-o", "--output", dir_okay=False, writable=True, help="Path to save exported model."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "-b", "--batch-size", min=0, help="Batch size of exported ONNX model. Set to 0 to mark as dynamic."
        ),
    ] = 1,
    height: Annotated[
        int, typer.Option("-h", "--height", min=0, help="Height of input image. Set to 0 to mark as dynamic.")
    ] = 0,
    width: Annotated[
        int, typer.Option("-w", "--width", min=0, help="Width of input image. Set to 0 to mark as dynamic.")
    ] = 0,
    num_keypoints: Annotated[
        int, typer.Option(min=128, help="Number of keypoints outputted by feature extractor.")
    ] = 1024,
    fuse_multi_head_attention: Annotated[
        bool,
        typer.Option(
            "--fuse-multi-head-attention",
            help="Fuse multi-head attention subgraph into one optimized operation. (ONNX Runtime-only).",
        ),
    ] = False,
    opset: Annotated[int, typer.Option(min=16, max=20, help="ONNX opset version of exported model.")] = 17,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether to also convert to FP16.")] = False,
):
    """Export LightGlue extractor and matcher as separate ONNX models for template matching."""
    import onnx
    import torch
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    from onnxruntime.transformers.float16 import convert_float_to_float16

    from lightglue_dynamo.models import DISK, LightGlue, SuperPoint
    from lightglue_dynamo.ops import use_fused_multi_head_attention

    match extractor_type:
        case Extractor.superpoint:
            extractor = SuperPoint(num_keypoints=num_keypoints)
        case Extractor.disk:
            extractor = DISK(num_keypoints=num_keypoints)
    matcher = LightGlue(**extractor_type.lightglue_config)
    
    extractor = extractor.eval()
    matcher = matcher.eval()

    # Set output paths
    if output is None:
        extractor_output = Path(f"weights/{extractor_type}_extractor.onnx")
        matcher_output = Path(f"weights/{extractor_type}_lightglue_matcher.onnx")
    else:
        base_path = output.with_suffix('')
        extractor_output = Path(f"{base_path}_extractor.onnx")
        matcher_output = Path(f"{base_path}_matcher.onnx")

    check_multiple_of(height, extractor_type.input_dim_divisor)
    check_multiple_of(width, extractor_type.input_dim_divisor)

    if height > 0 and width > 0 and num_keypoints > height * width:
        raise typer.BadParameter("num_keypoints cannot be greater than height * width.")

    if fuse_multi_head_attention:
        typer.echo(
            "Warning: Multi-head attention nodes will be fused. Exported model will only work with ONNX Runtime CPU & CUDA execution providers."
        )
        if torch.__version__ < "2.4":
            raise typer.Abort("Fused multi-head attention requires PyTorch 2.4 or later.")
        use_fused_multi_head_attention()

    # Export extractor model
    typer.echo(f"Exporting extractor model...")
    extractor_dynamic_axes = {"image": {}}
    if batch_size == 0:
        extractor_dynamic_axes["image"][0] = "batch_size"
    if height == 0:
        extractor_dynamic_axes["image"][2] = "height"
    if width == 0:
        extractor_dynamic_axes["image"][3] = "width"
    extractor_dynamic_axes |= {
        "keypoints": {0: "batch_size", 1: "num_keypoints"},
        "scores": {0: "batch_size", 1: "num_keypoints"},
        "descriptors": {0: "batch_size", 1: "descriptor_dim", 2: "num_keypoints"}
    }
    
    dummy_input = {"image": torch.zeros(batch_size or 1, extractor_type.input_channels, height or 256, width or 256)}
    torch.onnx.export(
        extractor,
        (dummy_input,),
        str(extractor_output),
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
        opset_version=opset,
        dynamic_axes=extractor_dynamic_axes,
    )
    onnx.checker.check_model(extractor_output)
    onnx.save_model(SymbolicShapeInference.infer_shapes(onnx.load_model(extractor_output), auto_merge=True), extractor_output)  # type: ignore
    typer.echo(f"✓ Successfully exported extractor to {extractor_output}")

    # Export matcher model
    typer.echo(f"Exporting matcher model...")
    # Create dummy features for matcher
    with torch.no_grad():
        dummy_kpts, dummy_scores, dummy_desc = extractor(dummy_input["image"])
    
    # Normalize keypoints like in Pipeline: 2 * keypoints / size - 1
    # LightGlue expects normalized coordinates in [-1, 1]
    size = torch.tensor([width, height], device=dummy_kpts.device, dtype=torch.float32)
    dummy_kpts_normalized = 2 * dummy_kpts.float() / size - 1
    
    # LightGlue expects interleaved batch: (2B, N, 2) for keypoints and (2B, N, D) for descriptors
    # Concatenate features from two "images" (using same features for both)
    dummy_kpts_interleaved = torch.cat([dummy_kpts_normalized, dummy_kpts_normalized], dim=0)  # (2B, N, 2)
    dummy_desc_interleaved = torch.cat([dummy_desc, dummy_desc], dim=0)  # (2B, N, D)
    
    matcher_dynamic_axes = {
        "keypoints": {0: "batch_x2", 1: "num_keypoints"},
        "descriptors": {0: "batch_x2", 1: "num_keypoints"},
        "matches": {0: "num_matches"},
        "mscores": {0: "num_matches"}
    }
    
    torch.onnx.export(
        matcher,
        (dummy_kpts_interleaved, dummy_desc_interleaved),
        str(matcher_output),
        input_names=["keypoints", "descriptors"],
        output_names=["matches", "mscores"],
        opset_version=opset,
        dynamic_axes=matcher_dynamic_axes,
    )
    onnx.checker.check_model(matcher_output)
    onnx.save_model(SymbolicShapeInference.infer_shapes(onnx.load_model(matcher_output), auto_merge=True), matcher_output)  # type: ignore
    typer.echo(f"✓ Successfully exported matcher to {matcher_output}")
    
    if fp16:
        typer.echo("Converting to FP16. Warning: This FP16 model should NOT be used for TensorRT. TRT provides its own fp16 option.")
        onnx.save_model(convert_float_to_float16(onnx.load_model(extractor_output)), extractor_output.with_suffix(".fp16.onnx"))
        onnx.save_model(convert_float_to_float16(onnx.load_model(matcher_output)), matcher_output.with_suffix(".fp16.onnx"))
    
    typer.echo(f"\n✓ Export complete! Two models created:")
    typer.echo(f"  1. Extractor: {extractor_output}")
    typer.echo(f"  2. Matcher: {matcher_output}")


@app.command()
def infer(
    model_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to ONNX model.")],
    left_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to first image.")
    ],
    right_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to second image.")
    ],
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output_path: Annotated[
        Optional[Path],  # noqa: UP007
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save output matches figure. If not given, show visualization.",
        ),
    ] = None,
    height: Annotated[
        int,
        typer.Option("-h", "--height", min=1, help="Height of input image at which to perform inference."),
    ] = 1024,
    width: Annotated[
        int,
        typer.Option("-w", "--width", min=1, help="Width of input image at which to perform inference."),
    ] = 1024,
    device: Annotated[
        InferenceDevice, typer.Option("-d", "--device", help="Device to run inference on.")
    ] = InferenceDevice.cpu,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether model uses FP16 precision.")] = False,
    profile: Annotated[bool, typer.Option("--profile", help="Whether to profile model execution.")] = False,
):
    """Run inference for LightGlue ONNX model."""
    import numpy as np
    import onnxruntime as ort

    from lightglue_dynamo import viz
    from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor

    raw_images = [left_image_path, right_image_path]
    raw_images = [cv2.resize(cv2.imread(str(i)), (width, height)) for i in raw_images]
    images = np.stack(raw_images)
    match extractor_type:
        case Extractor.superpoint:
            images = SuperPointPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
    images = images.astype(np.float16 if fp16 and device != InferenceDevice.tensorrt else np.float32)

    session_options = ort.SessionOptions()
    session_options.enable_profiling = profile
    # session_options.optimized_model_filepath = "weights/ort_optimized.onnx"

    providers = [("CPUExecutionProvider", {})]
    if device == InferenceDevice.cuda:
        providers.insert(0, ("CUDAExecutionProvider", {}))
    elif device == InferenceDevice.tensorrt:
        providers.insert(0, ("CUDAExecutionProvider", {}))
        providers.insert(
            0,
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "weights/.trtcache_engines",
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": "weights/.trtcache_timings",
                    "trt_fp16_enable": fp16,
                },
            ),
        )
    elif device == InferenceDevice.openvino:
        providers.insert(0, ("OpenVINOExecutionProvider", {}))

    session = ort.InferenceSession(model_path, session_options, providers)

    for _ in range(100 if profile else 1):
        keypoints, matches, mscores = session.run(None, {"images": images})

    viz.plot_images(raw_images)
    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
    if output_path is None:
        viz.plt.show()
    else:
        viz.save_plot(output_path)


@app.command()
def trtexec(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to ONNX model or built TensorRT engine."),
    ],
    left_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to first image.")
    ],
    right_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to second image.")
    ],
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output_path: Annotated[
        Optional[Path],  # noqa: UP007
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save output matches figure. If not given, show visualization.",
        ),
    ] = None,
    height: Annotated[
        int,
        typer.Option("-h", "--height", min=1, help="Height of input image at which to perform inference."),
    ] = 1024,
    width: Annotated[
        int,
        typer.Option("-w", "--width", min=1, help="Width of input image at which to perform inference."),
    ] = 1024,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether model uses FP16 precision.")] = False,
    profile: Annotated[bool, typer.Option("--profile", help="Whether to profile model execution.")] = False,
):
    """Run pure TensorRT inference for LightGlue model using Polygraphy (requires TensorRT to be installed)."""
    import numpy as np
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import (
        CreateConfig,
        EngineFromBytes,
        EngineFromNetwork,
        NetworkFromOnnxPath,
        SaveEngine,
        TrtRunner,
    )

    from lightglue_dynamo import viz
    from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor

    raw_images = [left_image_path, right_image_path]
    raw_images = [cv2.resize(cv2.imread(str(i)), (width, height)) for i in raw_images]
    images = np.stack(raw_images)
    match extractor_type:
        case Extractor.superpoint:
            images = SuperPointPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
    images = images.astype(np.float32)

    # Build TensorRT engine
    if model_path.suffix == ".engine":
        build_engine = EngineFromBytes(BytesFromPath(str(model_path)))
    else:  # .onnx
        build_engine = EngineFromNetwork(NetworkFromOnnxPath(str(model_path)), config=CreateConfig(fp16=fp16))
        build_engine = SaveEngine(build_engine, str(model_path.with_suffix(".engine")))

    with TrtRunner(build_engine) as runner:
        for _ in range(10 if profile else 1):  # Warm-up if profiling
            outputs = runner.infer(feed_dict={"images": images})
            keypoints, matches, mscores = outputs["keypoints"], outputs["matches"], outputs["mscores"]  # noqa: F841

        if profile:
            typer.echo(f"Inference Time: {runner.last_inference_time():.3f} s")

    viz.plot_images(raw_images)
    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
    if output_path is None:
        viz.plt.show()
    else:
        viz.save_plot(output_path)


if __name__ == "__main__":
    app()
