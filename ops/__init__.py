# ops/__init__.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import importlib
import os

OPS_REGISTRY: Dict[str, Callable[..., Any]] = {}
OPS_LOAD_ERRORS: List[Tuple[str, str]] = []  # (module, error_string)

# IMPORTANT: Only include ops that actually exist in this repo under ops/
OP_TO_MODULE: Dict[str, str] = {
    "echo": "echo",
    "map_tokenize": "map_tokenize",

    # TPU op
    "map_classify_tpu": "map_classify_tpu",

    # util ops that exist in this repo
    "csv_shard": "csv_shard",
    "fibonacci": "fibonacci",
    "prime_factor": "prime_factor",
    "risk_accumulate": "risk_accumulate",
    "sat_verify": "sat_verify",
    "subset_sum": "subset_sum",

    # NOTE: map_summarize exists, but requires torch; keep it mapped
    # only if you build torch into this image. Otherwise don't put it in TASKS.
    "map_summarize": "map_summarize",
}

_IMPORTED_MODULES: Set[str] = set()


def register_op(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        OPS_REGISTRY[name] = fn
        return fn
    return _decorator


def _parse_tasks_env() -> Optional[Set[str]]:
    raw = os.getenv("TASKS", "").strip()
    if not raw:
        return None

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None

    lowered = {p.lower() for p in parts}
    if "*" in lowered or "all" in lowered:
        return None
    if "none" in lowered:
        return set()

    return set(parts)


def list_ops() -> List[str]:
    enabled = _parse_tasks_env()
    if enabled is None:
        return sorted(OP_TO_MODULE.keys())
    return sorted([name for name in OP_TO_MODULE.keys() if name in enabled])


def _is_enabled(name: str) -> bool:
    enabled = _parse_tasks_env()
    if enabled is None:
        return True
    return name in enabled


def _import_op_module(module_name: str) -> None:
    if module_name in _IMPORTED_MODULES:
        return

    try:
        importlib.import_module(f"ops.{module_name}")
        _IMPORTED_MODULES.add(module_name)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        OPS_LOAD_ERRORS.append((module_name, msg))
        print(f"[ops] ERROR: failed to import ops.{module_name}: {msg}", flush=True)


def get_op(name: str) -> Callable[..., Any]:
    if not _is_enabled(name):
        raise ValueError(f"Op {name!r} is not enabled by TASKS. Enabled ops: {list_ops()}")

    module = OP_TO_MODULE.get(name)
    if not module:
        raise ValueError(f"Unknown op {name!r}. Allowed ops: {sorted(OP_TO_MODULE.keys())}")

    _import_op_module(module)

    fn = OPS_REGISTRY.get(name)
    if fn is None:
        if OPS_LOAD_ERRORS:
            errs = "; ".join([f"{m} => {e}" for (m, e) in OPS_LOAD_ERRORS[:10]])
            more = "" if len(OPS_LOAD_ERRORS) <= 10 else f" (+{len(OPS_LOAD_ERRORS)-10} more)"
            raise ValueError(
                f"Unknown or failed op {name!r}. Registered ops: {sorted(OPS_REGISTRY.keys())}. "
                f"Also saw op import errors: {errs}{more}"
            )
        raise ValueError(f"Unknown op {name!r}. Registered ops: {sorted(OPS_REGISTRY.keys())}")

    return fn


__all__ = ["OPS_REGISTRY", "OPS_LOAD_ERRORS", "OP_TO_MODULE", "register_op", "list_ops", "get_op"]
