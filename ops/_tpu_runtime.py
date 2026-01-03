# ops/_tpu_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

# Lazy, process-wide cache
_INTERPRETER = None
_INPUT_DETAILS = None
_OUTPUT_DETAILS = None
_MODEL_PATH = None


@dataclass(frozen=True)
class TPUHandle:
    interpreter: object
    input_details: object
    output_details: object
    model_path: str


def get_model_path(requested: Optional[str] = None) -> str:
    """
    Resolve model path with sane precedence.
    """
    return (
        requested
        or os.environ.get("TPU_MODEL_PATH")
        or "/models/model_edgetpu.tflite"
    )


def get_tpu_handle(model_path: str) -> TPUHandle:
    """
    Create/reuse a Coral Edge TPU interpreter for the given model_path.
    Imports pycoral only when needed.
    """
    global _INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS, _MODEL_PATH

    # Reuse if same model already loaded
    if _INTERPRETER is not None and _MODEL_PATH == model_path:
        return TPUHandle(_INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS, _MODEL_PATH)

    # Import inside function so agent can still start without pycoral installed
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

    return TPUHandle(_INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS, _MODEL_PATH)
