#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import socket
import signal
import traceback
from typing import Any, Dict, Optional, List, Tuple

import requests

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# ---------------- config ----------------

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "").rstrip("/")
if not CONTROLLER_URL:
    CONTROLLER_URL = "http://10.11.12.54:8080"  # safe default; override in env

AGENT_NAME = os.getenv("AGENT_NAME", socket.gethostname())

HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "10"))
IDLE_SLEEP_SEC = float(os.getenv("IDLE_SLEEP_SEC", "0.25"))

# TPU agents should usually lease 1 task at a time
MAX_TASKS = int(os.getenv("MAX_TASKS", "1"))
LEASE_TIMEOUT_MS = int(os.getenv("LEASE_TIMEOUT_MS", "3000"))

ERROR_LOG_EVERY_SEC = float(os.getenv("ERROR_LOG_EVERY_SEC", "10"))
ERROR_BACKOFF_SEC = float(os.getenv("ERROR_BACKOFF_SEC", "1.0"))

# Comma-separated
TASKS_RAW = os.getenv("TASKS", "echo,map_classify_tpu")

# Optional: "k=v,k2=v2"
AGENT_LABELS_RAW = os.getenv("AGENT_LABELS", "")

_running = True
_err_last: Dict[str, float] = {}


# ---------------- utils ----------------

def _parse_labels(raw: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    raw = (raw or "").strip()
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[part] = True
    return out


def _log_err_ratelimited(key: str, msg: str) -> None:
    now = time.time()
    last = _err_last.get(key, 0.0)
    if now - last >= ERROR_LOG_EVERY_SEC:
        _err_last[key] = now
        print(msg, flush=True)


def _collect_metrics() -> Dict[str, Any]:
    if psutil is None:
        return {}
    try:
        return {
            "cpu_util": float(psutil.cpu_percent(interval=None)) / 100.0,
            "ram_mb": float(psutil.virtual_memory().used) / (1024 * 1024),
        }
    except Exception:
        return {}


def _capabilities_list() -> List[str]:
    ops = [x.strip() for x in (TASKS_RAW or "").split(",") if x.strip()]
    # de-dup while preserving order
    seen = set()
    out: List[str] = []
    for o in ops:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


CAPS_LIST = _capabilities_list()
BASE_LABELS = _parse_labels(AGENT_LABELS_RAW)

# Honest, conservative TPU profile
WORKER_PROFILE: Dict[str, Any] = {
    "tier": "ultra-lite",
    "cpu": {"total_cores": 4, "reserved_cores": 3, "usable_cores": 1, "min_cpu_workers": 1, "max_cpu_workers": 1},
    "gpu": {"gpu_present": False, "gpu_count": 0, "vram_gb": None, "devices": [], "max_gpu_workers": 0},
    "tpu": {"tpu_present": True, "tpu_kind": os.getenv("TPU_KIND", "unknown"), "devices": [], "max_tpu_workers": 1},
    "workers": {"max_total_workers": 1, "current_workers": 0},
    "limits": {"max_payload_bytes": 262144, "max_tokens": 2048},
}


# ---------------- ops ----------------

def op_echo(payload: Any) -> Any:
    return {"ok": True, "echo": payload}


# Put your TPU implementation in tpu_ops.py next to app.py:
#   def map_classify_tpu(payload: dict) -> dict: ...
try:
    from tpu_ops import map_classify_tpu  # type: ignore
except Exception:
    map_classify_tpu = None


def _require_tpu_impl(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(
        "TPU op is not available.\n"
        "Fix: create tpu_ops.py next to app.py with:\n"
        "  def map_classify_tpu(payload: dict) -> dict:\n"
        "and include your working TPU inference code.\n"
    )


OPS: Dict[str, Any] = {
    "echo": op_echo,
    "map_classify_tpu": map_classify_tpu if map_classify_tpu is not None else _require_tpu_impl,
}


# ---------------- v1 http ----------------

def _post_json(path: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
    url = f"{CONTROLLER_URL}{path}"
    try:
        r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_SEC)
    except Exception as e:
        return 0, {"error": str(e), "url": url}

    if r.status_code == 204:
        return 204, None

    try:
        body = r.json()
    except Exception:
        body = r.text

    return r.status_code, body


def _lease_once() -> Optional[Tuple[str, Dict[str, Any]]]:
    payload: Dict[str, Any] = {
        "agent": AGENT_NAME,
        "capabilities": {"ops": CAPS_LIST},
        "max_tasks": MAX_TASKS,
        "timeout_ms": LEASE_TIMEOUT_MS,
        "labels": BASE_LABELS,
        "worker_profile": WORKER_PROFILE,
        "metrics": _collect_metrics(),
    }

    code, body = _post_json("/v1/leases", payload)
    if code == 204:
        return None
    if code == 0:
        raise RuntimeError(f"lease failed: {body}")
    if code >= 400:
        raise RuntimeError(f"lease HTTP {code}: {body}")

    if not isinstance(body, dict):
        raise RuntimeError(f"lease body not dict: {body!r}")

    lease_id = body.get("lease_id")
    tasks = body.get("tasks")

    if not isinstance(lease_id, str) or not lease_id:
        raise RuntimeError(f"lease missing lease_id: {body!r}")
    if not isinstance(tasks, list) or not tasks:
        return None

    task = tasks[0]
    if not isinstance(task, dict):
        raise RuntimeError(f"task not dict: {task!r}")

    return lease_id, task


def _post_result(
    lease_id: str,
    job_id: str,
    job_epoch: Optional[int],
    status: str,
    result: Any = None,
    error: Any = None,
) -> None:
    payload: Dict[str, Any] = {
        "lease_id": lease_id,
        "job_id": job_id,
        "job_epoch": job_epoch,
        "status": status,   # "succeeded" or "failed"
        "result": result,
        "error": error,
    }
    code, body = _post_json("/v1/results", payload)
    if code == 0:
        raise RuntimeError(f"result failed: {body}")
    if code >= 400:
        raise RuntimeError(f"result HTTP {code}: {body}")


def _extract_task(task: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any], Optional[int]]:
    job_id = task.get("id") or task.get("job_id")
    op = task.get("op")
    payload = task.get("payload") or {}
    job_epoch = task.get("job_epoch")

    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError(f"task missing job id: {task!r}")
    if not isinstance(op, str) or not op:
        raise RuntimeError(f"task missing op: {task!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"task payload not dict: {task!r}")

    return job_id, op, payload, job_epoch


# ---------------- runtime ----------------

def _shutdown(signum: int, _frame: Any) -> None:
    global _running
    _running = False
    print(f"[agent-tpu-v1] shutdown signal {signum}", flush=True)


def main() -> int:
    global _running

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if not CAPS_LIST:
        print("[agent-tpu-v1] no TASKS configured; exiting", flush=True)
        return 2

    print(f"[agent-tpu-v1] starting name={AGENT_NAME} controller={CONTROLLER_URL} ops={CAPS_LIST}", flush=True)

    while _running:
        try:
            leased = _lease_once()
        except Exception as e:
            _log_err_ratelimited("lease", f"[agent-tpu-v1] lease error: {e}")
            time.sleep(ERROR_BACKOFF_SEC)
            continue

        if not leased:
            time.sleep(IDLE_SLEEP_SEC)
            continue

        lease_id, task = leased

        try:
            job_id, op, payload, job_epoch = _extract_task(task)
        except Exception as e:
            _log_err_ratelimited("task:bad", f"[agent-tpu-v1] bad task: {e} task={repr(task)[:300]}")
            continue

        start_ts = time.time()
        ok = True
        out: Any = None
        err: Any = None

        try:
            fn = OPS.get(op)
            if fn is None:
                raise RuntimeError(f"Unknown op '{op}'")
            # TPU RULE: run inline (no fork / no process pool)
            out = fn(payload)
        except Exception as e:
            ok = False
            err = {
                "type": type(e).__name__,
                "message": str(e),
                "trace": traceback.format_exc(limit=12),
            }

        duration_ms = (time.time() - start_ts) * 1000.0

        try:
            _post_result(
                lease_id,
                job_id,
                job_epoch,
                "succeeded" if ok else "failed",
                result=(out if ok else None),
                error=(None if ok else err),
            )
        except Exception as e:
            _log_err_ratelimited("result", f"[agent-tpu-v1] post result error: {e}")

        if ok:
            print(f"[agent-tpu-v1] ok job={job_id} op={op} ms={duration_ms:.1f}", flush=True)
        else:
            _log_err_ratelimited("exec", f"[agent-tpu-v1] FAIL job={job_id} op={op} ms={duration_ms:.1f} err={err}")

    print("[agent-tpu-v1] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
