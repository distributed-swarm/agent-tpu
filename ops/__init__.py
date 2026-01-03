# ops/__init__.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, List, Tuple
import importlib
import os
import traceback

# Global registry of ops (populated by @register_op decorators)
OPS_REGISTRY: Dict[str, Callable[..., Any]] = {}

# Track import/load errors so runtime + CI can diagnose quickly
OPS_LOAD_ERRORS: List[Tuple[str, str]] = []  # (module, error_string)


def register_op(name: str):
    """
    Decorator used by op modules to register their handler.
    """
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        OPS_REGISTRY[name] = fn
        return fn
    return _wrap


def _import_op_module(mod: str) -> None:
    """
    Import an op module by name and record failures.
    """
    try:
        importlib.import_module(f"{__name__}.{mod}")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        OPS_LOAD_ERRORS.append((mod, msg))
        print(f"[ops] ERROR: failed to import ops.{mod}: {msg}", flush=True)
        traceback.print_exc()


# Map op name -> module filename (without .py)
# This is the allow-list used by the agent.
OP_TO_MODULE: Dict[str, str] = {
    # Core ops
    "echo": "echo",
    "map_tokenize": "map_tokenize",
    "map_summarize": "map_summarize",

    # CSV shard
    "csv_shard": "csv_shard",
    "read_csv_shard": "csv_shard",

    # Math / IO
    "risk_accumulate": "risk_accumulate",
    "fibonacci": "fibonacci",
    "prime_factor": "prime_factor",

    # Stress / verification
    "sat_verify": "sat_verify",
    "subset_sum": "subset_sum",

    # TPU ops
    "map_classify_tpu": "map_classify_tpu",
}


def get_op(name: str) -> Callable[..., Any]:
    """
    Return the handler function for a given op name.

    Lazy-loads the underlying module ONLY when the op is requested,
    and ONLY if the op is enabled by TASKS and present in OP_TO_MODULE.
    """
    if name not in OPS_REGISTRY:
        mod = OP_TO_MODULE.get(name)
        if mod is None:
            raise ValueError(
                f"Unknown op {name!r}. Allowed ops: {sorted(OP_TO_MODULE.keys())}"
            )
        _import_op_module(mod)

    fn = OPS_REGISTRY.get(name)
    if fn is None:
        if OPS_LOAD_ERRORS:
            errs = "; ".join([f"{m} => {e}" for (m, e) in OPS_LOAD_ERRORS[:10]])
            more = "" if len(OPS_LOAD_ERRORS) <= 10 else f" (+{len(OPS_LOAD_ERRORS)-10} more)"
            raise ValueError(
                f"Unknown or failed op {name!r}. Registered ops: {sorted(OPS_REGISTRY.keys())}. "
                f"Also saw op import errors: {errs}{more}"
            )
        raise ValueError(
            f"Unknown op {name!r}. Registered ops: {sorted(OPS_REGISTRY.keys())}"
        )

    return fn
