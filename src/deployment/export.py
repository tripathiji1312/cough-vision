"""
ONNX export and optimisation for edge deployment.

Exports the TBEnsemble to:
  1. ONNX (opset 17) with input/output names
  2. Optimised ONNX (graph simplification via onnxoptimizer)
  3. INT8 quantised ONNX for battery-constrained ARM devices

Deployment stack per research plan:
  - ONNX Runtime (CPU / GPU, cross-platform default)
  - TensorRT (NVIDIA Jetson Orin Nano — best GPU latency)
  - TFLite (ARM CPU — best battery / portability)

Target: <1 second inference on a mid-range laptop GPU or Jetson Orin Nano.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

_TORCH_AVAILABLE = False
try:
    import torch as _t  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _t = None  # type: ignore[assignment]

_ONNX_AVAILABLE = False
try:
    import onnx as _onnx  # type: ignore[import-untyped]
    _ONNX_AVAILABLE = True
except ImportError:
    _onnx = None  # type: ignore[assignment]

_ORT_AVAILABLE = False
try:
    import onnxruntime as _ort  # type: ignore[import-untyped]
    _ORT_AVAILABLE = True
except ImportError:
    _ort = None  # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(
    model: Any,
    output_path: str | Path,
    cnn_input_size: int = 224,
    vit_input_size: int = 384,
    opset: int = 17,
    dynamic_axes: bool = True,
    model_version: str = "v1.0.0",
) -> Path:
    """
    Export the TBEnsemble (or any nn.Module) to ONNX.

    The model is exported in inference mode (no Grad-CAM hooks, eval mode).
    Uses a single-input interface: the CNN image tensor (the ViT branch
    receives a resized copy internally), simplifying edge deployment.

    Args:
        model:           TBEnsemble nn.Module.
        output_path:     Destination .onnx file path.
        cnn_input_size:  CNN branch spatial resolution (224).
        vit_input_size:  ViT branch spatial resolution (384). Not used as
                         a separate input — the model resizes internally.
        opset:           ONNX opset version (17 recommended for TRT 8.6+).
        dynamic_axes:    Allow variable batch size.
        model_version:   Embedded in ONNX metadata for post-market tracking.

    Returns:
        Path to the exported .onnx file.
    """
    _require(_TORCH_AVAILABLE, "torch")
    _require(_ONNX_AVAILABLE, "onnx")
    import torch  # type: ignore[import-untyped]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    # Disable Grad-CAM hooks during export
    try:
        model.disable_gradcam()
    except AttributeError:
        pass

    dummy_cnn = torch.zeros(1, 3, cnn_input_size, cnn_input_size)

    dyn: dict[str, dict] = {}
    if dynamic_axes:
        dyn = {"cxr_image": {0: "batch_size"}, "tb_prob": {0: "batch_size"}}

    try:
        torch.onnx.export(
            model,
            (dummy_cnn,),
            str(output_path),
            input_names=["cxr_image"],
            output_names=["tb_logits", "tb_prob"],
            dynamic_axes=dyn,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
        )
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc

    # Embed metadata for post-market drift tracking
    try:
        import onnx  # type: ignore[import-untyped]
        onnx_model = onnx.load(str(output_path))
        meta = onnx_model.metadata_props.add()
        meta.key   = "model_version"
        meta.value = model_version
        meta2 = onnx_model.metadata_props.add()
        meta2.key   = "task"
        meta2.value = "TB_screening_CXR"
        onnx.save(onnx_model, str(output_path))
    except Exception as exc:
        warnings.warn(f"Could not embed ONNX metadata: {exc}", stacklevel=2)

    print(f"ONNX model exported → {output_path}")
    return output_path


def optimise_onnx(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """
    Apply onnxoptimizer passes to reduce graph complexity.

    Args:
        input_path:  Path to the raw exported .onnx file.
        output_path: Destination for the optimised model (default: overwrite).

    Returns:
        Path to the optimised .onnx file.
    """
    try:
        import onnxoptimizer  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("onnxoptimizer is required. pip install onnxoptimizer") from exc

    _require(_ONNX_AVAILABLE, "onnx")
    import onnx  # type: ignore[import-untyped]

    input_path  = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    try:
        model = onnx.load(str(input_path))
        passes = [
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_unused_initializer",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
        ]
        optimised = onnxoptimizer.optimize(model, passes)
        onnx.save(optimised, str(output_path))
        print(f"Optimised ONNX saved → {output_path}")
        return output_path
    except Exception as exc:
        raise RuntimeError(f"ONNX optimisation failed: {exc}") from exc


def quantise_onnx_dynamic(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """
    Apply dynamic INT8 quantisation for ARM/edge deployment.

    Args:
        input_path:  Path to the (optionally optimised) .onnx file.
        output_path: Destination for the quantised model.

    Returns:
        Path to the quantised .onnx file.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for quantisation. pip install onnxruntime"
        ) from exc

    input_path  = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_int8.onnx")
    output_path = Path(output_path)

    try:
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
        print(f"INT8 quantised ONNX saved → {output_path}")
        return output_path
    except Exception as exc:
        raise RuntimeError(f"ONNX quantisation failed: {exc}") from exc
