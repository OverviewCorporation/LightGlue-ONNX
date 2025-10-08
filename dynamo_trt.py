from pathlib import Path
from typing import Annotated, Optional

import cv2
import typer

from lightglue_dynamo.cli_utils import check_multiple_of
from lightglue_dynamo.config import Extractor, InferenceDevice

app = typer.Typer()


@app.callback()
def callback():
    """LightGlue Dynamo CLI for TensorRT-compatible export"""


@app.command()
def export_trt_compatible(
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
    opset: Annotated[int, typer.Option(min=14, max=17, help="ONNX opset version of exported model.")] = 14,
):
    """Export TensorRT-compatible LightGlue extractor and matcher as separate ONNX models."""
    import onnx
    import torch
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    from lightglue_dynamo.models import DISK, SuperPoint
    from lightglue_dynamo.models.lightglue_trt import LightGlueTRT

    match extractor_type:
        case Extractor.superpoint:
            extractor = SuperPoint(num_keypoints=num_keypoints)
        case Extractor.disk:
            extractor = DISK(num_keypoints=num_keypoints)
    
    # Use TensorRT-compatible version of LightGlue
    matcher = LightGlueTRT(**extractor_type.lightglue_config)
    
    extractor = extractor.eval()
    matcher = matcher.eval()

    # Set output paths
    if output is None:
        extractor_output = Path(f"weights/{extractor_type}_extractor_trt.onnx")
        matcher_output = Path(f"weights/{extractor_type}_lightglue_matcher_trt.onnx")
    else:
        base_path = output.with_suffix('')
        extractor_output = Path(f"{base_path}_extractor_trt.onnx")
        matcher_output = Path(f"{base_path}_matcher_trt.onnx")

    check_multiple_of(height, extractor_type.input_dim_divisor)
    check_multiple_of(width, extractor_type.input_dim_divisor)

    if height > 0 and width > 0 and num_keypoints > height * width:
        raise typer.BadParameter("num_keypoints cannot be greater than height * width.")

    # Export extractor model
    typer.echo(f"Exporting TensorRT-compatible extractor model...")
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
    
    # Export with specific settings for TensorRT compatibility
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
    onnx.checker.check_model(extractor_output)
    onnx.save_model(SymbolicShapeInference.infer_shapes(onnx.load_model(extractor_output), auto_merge=True), extractor_output)  # type: ignore
    typer.echo(f"✓ Successfully exported extractor to {extractor_output}")

    # Export matcher model
    typer.echo(f"Exporting TensorRT-compatible matcher model...")
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
    
    # Export with TensorRT-friendly settings
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
        # Avoid INT64 operations that cause issues with TensorRT
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    onnx.checker.check_model(matcher_output)
    onnx.save_model(SymbolicShapeInference.infer_shapes(onnx.load_model(matcher_output), auto_merge=True), matcher_output)  # type: ignore
    typer.echo(f"✓ Successfully exported matcher to {matcher_output}")
    
    typer.echo(f"\n✓ TensorRT-compatible export complete! Two models created:")
    typer.echo(f"  1. Extractor: {extractor_output}")
    typer.echo(f"  2. Matcher: {matcher_output}")
    typer.echo(f"\nThese models should be compatible with TensorRT conversion.")


@app.command() 
def optimize_for_trt(
    input_model: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Input ONNX model path.")],
    output_model: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", dir_okay=False, writable=True, help="Output optimized ONNX model path."),
    ] = None,
):
    """Post-process ONNX model to make it more TensorRT-compatible."""
    import onnx
    from onnx import helper, numpy_helper
    import numpy as np

    if output_model is None:
        output_model = input_model.with_suffix('.trt_optimized.onnx')
    
    typer.echo(f"Loading ONNX model from {input_model}")
    model = onnx.load(str(input_model))
    
    # Remove LayerNormalization nodes and replace with equivalent operations
    graph = model.graph
    nodes_to_remove = []
    nodes_to_add = []
    
    for i, node in enumerate(graph.node):
        if node.op_type == "LayerNormalization":
            typer.echo(f"Found LayerNormalization node: {node.name}")
            
            # Get attributes
            axis = -1
            epsilon = 1e-5
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
                elif attr.name == "epsilon":
                    epsilon = attr.f
            
            input_name = node.input[0]
            weight_name = node.input[1] if len(node.input) > 1 else None
            bias_name = node.input[2] if len(node.input) > 2 else None
            output_name = node.output[0]
            
            # Create equivalent operations
            prefix = f"{node.name}_decomposed"
            
            # ReduceMean for mean calculation
            mean_output = f"{prefix}_mean"
            mean_node = helper.make_node(
                "ReduceMean",
                inputs=[input_name],
                outputs=[mean_output],
                axes=[axis],
                keepdims=1,
                name=f"{prefix}_mean_op"
            )
            
            # Sub for centering
            centered_output = f"{prefix}_centered"
            sub_node = helper.make_node(
                "Sub",
                inputs=[input_name, mean_output],
                outputs=[centered_output],
                name=f"{prefix}_sub_op"
            )
            
            # Pow for squaring
            squared_output = f"{prefix}_squared"
            two_const = f"{prefix}_two"
            # Add constant 2
            two_tensor = helper.make_tensor(two_const, onnx.TensorProto.FLOAT, [], [2.0])
            graph.initializer.append(two_tensor)
            
            pow_node = helper.make_node(
                "Pow",
                inputs=[centered_output, two_const],
                outputs=[squared_output],
                name=f"{prefix}_pow_op"
            )
            
            # ReduceMean for variance
            var_output = f"{prefix}_var"
            var_node = helper.make_node(
                "ReduceMean",
                inputs=[squared_output],
                outputs=[var_output],
                axes=[axis],
                keepdims=1,
                name=f"{prefix}_var_op"
            )
            
            # Add epsilon
            var_eps_output = f"{prefix}_var_eps"
            eps_const = f"{prefix}_eps"
            eps_tensor = helper.make_tensor(eps_const, onnx.TensorProto.FLOAT, [], [epsilon])
            graph.initializer.append(eps_tensor)
            
            add_eps_node = helper.make_node(
                "Add",
                inputs=[var_output, eps_const],
                outputs=[var_eps_output],
                name=f"{prefix}_add_eps_op"
            )
            
            # Sqrt
            std_output = f"{prefix}_std"
            sqrt_node = helper.make_node(
                "Sqrt",
                inputs=[var_eps_output],
                outputs=[std_output],
                name=f"{prefix}_sqrt_op"
            )
            
            # Div for normalization
            normalized_output = f"{prefix}_normalized"
            div_node = helper.make_node(
                "Div",
                inputs=[centered_output, std_output],
                outputs=[normalized_output],
                name=f"{prefix}_div_op"
            )
            
            # Apply weight and bias if present
            final_output = normalized_output
            if weight_name:
                weighted_output = f"{prefix}_weighted"
                mul_node = helper.make_node(
                    "Mul",
                    inputs=[normalized_output, weight_name],
                    outputs=[weighted_output],
                    name=f"{prefix}_mul_op"
                )
                nodes_to_add.append(mul_node)
                final_output = weighted_output
            
            if bias_name:
                final_add_node = helper.make_node(
                    "Add",
                    inputs=[final_output, bias_name],
                    outputs=[output_name],
                    name=f"{prefix}_add_bias_op"
                )
                nodes_to_add.append(final_add_node)
            else:
                # Rename final output
                if final_output != output_name:
                    identity_node = helper.make_node(
                        "Identity",
                        inputs=[final_output],
                        outputs=[output_name],
                        name=f"{prefix}_identity_op"
                    )
                    nodes_to_add.append(identity_node)
            
            # Add all decomposed nodes
            nodes_to_add.extend([mean_node, sub_node, pow_node, var_node, add_eps_node, sqrt_node, div_node])
            nodes_to_remove.append(i)
    
    # Remove original LayerNormalization nodes (in reverse order to maintain indices)
    for i in reversed(nodes_to_remove):
        del graph.node[i]
    
    # Add new decomposed nodes
    graph.node.extend(nodes_to_add)
    
    typer.echo(f"Saving optimized model to {output_model}")
    onnx.save(model, str(output_model))
    onnx.checker.check_model(str(output_model))
    typer.echo(f"✓ Model optimization complete!")


if __name__ == "__main__":
    app()