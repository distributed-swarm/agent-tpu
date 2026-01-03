# ops/__init__.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import importlib
import os

# Global registry of ops (populated by @register_op decorators)
OPS_REGISTRY: Dict[str, Callable[..., Any]] = {}

# Track import/load errors so runtime + CI can diagnose quickly
OPS_LOAD_ERRORS: List[Tuple[str, str]] = []  # (module, error_string)

# Allow-list map: op_name -> module_name (under ops/)
# Example: "echo" -> "echo" means module path "ops.echo"
OP_TO_MODULE: Dict[str, str] = {
    # Keep this list aligned with your repo ops modules.
    # Add TPU ops here too, e.g. "map_classify_tpu": "map_classify_tpu"
    "echo": "echo",
    "map_tokenize": "map_tokenize",
    "map_summarize": "map_summarize",
    "map_classify": "map_classify",
    "map_classify_tpu": "map_classify_tpu",
    "read_csv_shard": "read_csv_shard",
    # Add other ops as needed:
    # "fibonacci": "fibonacci",
    # "prime_factor": "prime_factor",
    # "risk_accumulate": "risk_accumulate",
}

# Cache of modules we've successfully imported
_IMPORTED_MODULES: Set[str] = set()


def register_op(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator used by ops modules to register an op function.
    Example in ops/echo.py:
        @register_op("echo")
        def echo(payload): ...
    """
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        OPS_REGISTRY[name] = fn
        return fn
    return _decorator


def _parse_tasks_env() -> Optional[Set[str]]:
    """
    Returns:
      - None  => treat as "all enabled" (TASKS not set / empty / includes '*')
      - set() => enabled ops (intersection will be taken with OP_TO_MODULE keys)
    """
    raw = os.getenv("TASKS", "").strip()
    if not raw:
        return None  # no gating provided => allow all allow-listed ops

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
    """
    Return the ops enabled for this agent (based on TASKS and OP_TO_MODULE).
    This is what your capabilities reporting / ops_loader should use.
    """
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
    """
    Import ops.<module_name> once.
    Modules should call @register_op(...) at import time.
    """
    if module_name in _IMPORTED_MODULES:
        return

    try:
        importlib.import_module(f"ops.{module_name}")
        _IMPORTED_MODULES.add(module_name)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        OPS_LOAD_ERRORS.append((module_name, msg))
        # keep stdout noisy but not fatal here; get_op will raise nicely
        print(f"[ops] ERROR: failed to import ops.{module_name}: {msg}", flush=True)


def get_op(name: str) -> Callable[..., Any]:
    """
    Return a callable op function for `name`.

    Behavior:
      - Enforces TASKS gating (only enabled ops can be requested)
      - Enforces allow-list (OP_TO_MODULE)
      - Lazily imports the module to populate OPS_REGISTRY
      - Raises ValueError with diagnostic info on failure
    """
    if not _is_enabled(name):
        raise ValueError(
            f"Op {name!r} is not enabled by TASKS. Enabled ops: {list_ops()}"
        )

    module = OP_TO_MODULE.get(name)
    if not module:
        raise ValueError(
            f"Unknown op {name!r}. Allowed ops: {sorted(OP_TO_MODULE.keys())}"
        )

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
        raise ValueError(
            f"Unknown op {name!r}. Registered ops: {sorted(OPS_REGISTRY.keys())}"
        )

    return fn


__all__ = [
    "OPS_REGISTRY",
    "OPS_LOAD_ERRORS",
    "OP_TO_MODULE",
    "register_op",
    "list_ops",
    "get_op",
]
