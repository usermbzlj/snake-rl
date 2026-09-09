from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_run_ref(raw: Any) -> str:
    if isinstance(raw, dict):
        raw = raw.get("name") or ""
    return str(raw or "").strip()


def monitor_logdir_for(run_name: str, runs_dir: Path) -> Path:
    name = normalize_run_ref(run_name)
    root = runs_dir.resolve()
    if not name:
        return root
    if name in {".", ".."}:
        raise ValueError("非法运行名")
    run_dir = (root / name).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("非法运行名") from exc
    return run_dir


def should_restart_monitor(current_logdir: str | None, desired: Path, alive: bool) -> bool:
    if not alive:
        return False
    if not current_logdir:
        return True
    return Path(current_logdir).resolve() != desired.resolve()
