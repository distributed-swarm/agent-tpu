# agent-tpu/ops/map_classify_tpu.py

from typing import Dict, Any

OP_NAME = "map_classify_tpu"

def run(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    TPU-backed INT8 TFLite classification.
    Payload should already be preprocessed.
    """

    # TEMP: stub until TPU wiring is complete
    return {
        "label": "stub",
        "score": 0.0,
        "op": OP_NAME,
    }
