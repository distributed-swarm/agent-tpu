# agent-tpu/ops/map_classify_tpu.py

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import os
import time

OP_NAME = "map_classify_tpu"

# Lazy-initialized globals (keep simple; avoid new files)
_INTERPRETER = None
_INPUT_DETAILS = None
_OUTPUT_DETAILS = None
_MODEL_PATH = None


def _init_interpreter(model_path: str):
    """
    Initialize a Coral Edge TPU interpreter once per process.
    """
    global _INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS, _MODEL_PATH

    # Import here so agent can still start even if pycoral isn't present.
    from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter

    tpus = list_edge_tpus()
    if not tpus:
        raise RuntimeError("No Edge TPU detected (pycoral list_edge_tpus() returned empty).")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TPU model not found: {model_path}")

    interp = make_interpreter(model_path)
    interp.allocate_tensors()

    _INTERPRETER = interp
    _INPUT_DETAILS = interp.get_input_details()
    _OUTPUT_DETAILS = interp.get_output_details()
    _MODEL_PATH = model_path


def _ensure_ready(payload: Dict[str, Any]) -> str:
    """
    Ensure TPU interpreter is initialized; return chosen model_path.
    """
    model_path = (
        payload.get("model_path")
        or os.environ.get("TPU_MODEL_PATH")
        or "/models/model_edgetpu.tflite"
    )

    global _INTERPRETER, _MODEL_PATH
    if _INTERPRETER is None or _MODEL_PATH != model_path:
        _init_interpreter(model_path)

    return model_path


def run(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    TPU-backed INT8 TFLite classification.

    Contract (v0, minimal):
      payload = {
        "input": <flat list[int] OR nested list matching input tensor shape>,
        "model_path": (optional) "/path/to/model_edgetpu.tflite",
        "topk": (optional) int
      }

    Returns:
      {
        "op": "map_classify_tpu",
        "model_path": "...",
        "topk": [{"index": int, "score": float}, ...],
        "elapsed_ms": float
      }
    """
    t0 = time.time()
    model_path = _ensure_ready(payload)

    if "input" not in payload:
        raise ValueError('payload missing required key: "input"')

    # Local import keeps global import surface small
    import numpy as np

    # Build input tensor
    inp = payload["input"]
    input_details = _INPUT_DETAILS[0]
    shape = tuple(input_details["shape"])
    dtype = input_details["dtype"]

    arr = np.array(inp, dtype=dtype)

    # Ensure shape matches model input
    if arr.size != int(np.prod(shape)):
        raise ValueError(f'Input size mismatch. Got {arr.size} values, expected {int(np.prod(shape))} for shape {shape}.')

    arr = arr.reshape(shape)

    # Run inference
    _INTERPRETER.set_tensor(input_details["index"], arr)
    _INTERPRETER.invoke()

    # Read output (assume single output tensor, typical for classifiers)
    out_details = _OUTPUT_DETAILS[0]
    out = _INTERPRETER.get_tensor(out_details["index"]).flatten()

    # Convert to Python floats for JSON friendliness
    out_f = out.astype("float32")

    # Top-k
    topk = int(payload.get("topk", 5))
    topk = max(1, min(topk, out_f.shape[0]))
    idx = np.argpartition(-out_f, topk - 1)[:topk]
    idx = idx[np.argsort(-out_f[idx])]

    result = [{"index": int(i), "score": float(out_f[i])} for i in idx]

    elapsed_ms = (time.time() - t0) * 1000.0
    return {
        "op": OP_NAME,
        "model_path": model_path,
        "topk": result,
        "elapsed_ms": elapsed_ms,
    }
