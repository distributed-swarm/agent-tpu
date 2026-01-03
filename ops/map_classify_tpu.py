# ops/map_classify_tpu.py
from __future__ import annotations

from typing import Dict, Any
import time

import numpy as np

from ._tpu_runtime import get_model_path, get_tpu_handle

OP_NAME = "map_classify_tpu"


def _topk(scores: np.ndarray, k: int):
    k = max(1, min(int(k), int(scores.shape[0])))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [{"index": int(i), "score": float(scores[i])} for i in idx]


def _cpu_fallback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal, safe fallback:
    - returns 'fallback' flag
    - doesn't pretend it ran inference
    """
    return {
        "op": OP_NAME,
        "fallback": "cpu",
        "reason": payload.get("fallback_reason", "TPU unavailable"),
        "topk": [],
    }


def run(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    TPU-backed INT8 TFLite classification.

    Uses:
      - payload["model_path"] OR env TPU_MODEL_PATH OR /models/model_edgetpu.tflite
      - payload["topk"] default 5
      - payload["input"] required (flat list[int] of int8 values)
      - payload["allow_fallback"] default True
    """
    t0 = time.time()

    allow_fallback = payload.get("allow_fallback", True)

    try:
        model_path = get_model_path(payload.get("model_path"))
        h = get_tpu_handle(model_path)

        if "input" not in payload:
            raise ValueError('payload missing required key: "input"')

        input_details = h.input_details[0]
        out_details = h.output_details[0]

        shape = tuple(input_details["shape"])
        dtype = input_details["dtype"]

        arr = np.array(payload["input"], dtype=dtype)

        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"Input size mismatch. Got {arr.size}, expected {expected} for shape {shape}.")

        arr = arr.reshape(shape)

        h.interpreter.set_tensor(input_details["index"], arr)
        h.interpreter.invoke()

        out = h.interpreter.get_tensor(out_details["index"]).flatten().astype("float32")
        topk = _topk(out, payload.get("topk", 5))

        return {
            "op": OP_NAME,
            "model_path": h.model_path,
            "topk": topk,
            "elapsed_ms": (time.time() - t0) * 1000.0,
        }

    except Exception as e:
        if allow_fallback:
            return {
                **_cpu_fallback({"fallback_reason": str(e), **payload}),
                "elapsed_ms": (time.time() - t0) * 1000.0,
            }
        raise
