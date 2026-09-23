"""On-disk experiment layout under experiments/<id>/."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from snake_rl.core.config import ExperimentConfig

log = logging.getLogger(__name__)

STATUSES = frozenset({"created", "running", "paused", "stopped", "finished", "error"})

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def default_experiments_root() -> Path:
    """Repo-root/experiments (two levels up from snake_rl/lab/)."""
    return Path(__file__).resolve().parents[2] / "experiments"


def _now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def make_experiment_id(name: str, when: datetime | None = None) -> str:
    dt = when or datetime.now().astimezone()
    stamp = dt.strftime("%Y%m%d-%H%M%S")
    slug = _SLUG_RE.sub("-", name.strip()).strip("-").lower()
    if not slug:
        slug = "exp"
    slug = slug[:40]
    return f"{stamp}-{slug}"


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    last_err: Exception | None = None
    for attempt in range(20):
        try:
            tmp.write_text(payload, encoding="utf-8")
            # On Windows, replace can fail if another process has the target open.
            try:
                tmp.replace(path)
            except PermissionError:
                # Fallback: overwrite in place
                path.write_text(payload, encoding="utf-8")
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.05 * (attempt + 1))
    if last_err:
        raise last_err
    raise RuntimeError(f"无法写入 {path}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def downsample_evenly(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep first/last and evenly spaced middle rows; always keep last."""
    n = len(rows)
    if n <= limit or limit <= 0:
        return rows
    if limit == 1:
        return [rows[-1]]
    indices = {0, n - 1}
    for i in range(1, limit - 1):
        indices.add(round(i * (n - 1) / (limit - 1)))
    return [rows[i] for i in sorted(indices)]


def sparkline(metrics: list[dict[str, Any]], key: str = "score_mean", limit: int = 60) -> list[float]:
    vals = [float(r[key]) for r in metrics if key in r and r[key] is not None]
    if len(vals) <= limit:
        return vals
    sampled = downsample_evenly([{"v": v} for v in vals], limit)
    return [float(r["v"]) for r in sampled]


class ExperimentStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_experiments_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def exp_dir(self, exp_id: str) -> Path:
        return self.root / exp_id

    def meta_path(self, exp_id: str) -> Path:
        return self.exp_dir(exp_id) / "experiment.json"

    def metrics_path(self, exp_id: str) -> Path:
        return self.exp_dir(exp_id) / "metrics.jsonl"

    def events_path(self, exp_id: str) -> Path:
        return self.exp_dir(exp_id) / "events.jsonl"

    def ckpt_dir(self, exp_id: str) -> Path:
        return self.exp_dir(exp_id) / "checkpoints"

    def ckpt_path(self, exp_id: str, name: str) -> Path:
        return self.ckpt_dir(exp_id) / f"{name}.pt"

    def exists(self, exp_id: str) -> bool:
        return self.meta_path(exp_id).is_file()

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        ids = [p.name for p in self.root.iterdir() if p.is_dir() and (p / "experiment.json").is_file()]
        return sorted(ids, reverse=True)

    def create(
        self,
        config: ExperimentConfig,
        *,
        parent_id: str | None = None,
        notes: str | None = None,
        exp_id: str | None = None,
    ) -> dict[str, Any]:
        eid = exp_id or make_experiment_id(f"{config.algo} {config.name}")
        # Ensure uniqueness
        base = eid
        n = 1
        while self.exists(eid):
            eid = f"{base}-{n}"
            n += 1
        d = self.exp_dir(eid)
        d.mkdir(parents=True, exist_ok=False)
        self.ckpt_dir(eid).mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "id": eid,
            "name": config.name,
            "algo": config.algo,
            "created_at": _now_iso(),
            "status": "created",
            "config": config.model_dump(),
            "parent_id": parent_id,
            "notes": notes,
            "error": None,
            "best_eval_score": None,
            "env_steps": 0,
            "started_at": None,
            "elapsed_s": 0.0,
        }
        _atomic_write_json(self.meta_path(eid), meta)
        self.metrics_path(eid).write_text("", encoding="utf-8")
        self.events_path(eid).write_text("", encoding="utf-8")
        return meta

    def read_meta(self, exp_id: str) -> dict[str, Any]:
        path = self.meta_path(exp_id)
        if not path.is_file():
            raise FileNotFoundError(f"实验不存在: {exp_id}")
        return _read_json(path)

    def write_meta(self, exp_id: str, meta: dict[str, Any]) -> None:
        _atomic_write_json(self.meta_path(exp_id), meta)

    def update_status(
        self,
        exp_id: str,
        status: str,
        *,
        error: str | None = None,
        clear_error: bool = False,
    ) -> dict[str, Any]:
        if status not in STATUSES:
            raise ValueError(f"非法状态: {status}")
        meta = self.read_meta(exp_id)
        meta["status"] = status
        if clear_error:
            meta["error"] = None
        if error is not None:
            meta["error"] = error
        self.write_meta(exp_id, meta)
        return meta

    def update_config(self, exp_id: str, config: ExperimentConfig | dict[str, Any]) -> dict[str, Any]:
        meta = self.read_meta(exp_id)
        if isinstance(config, ExperimentConfig):
            meta["config"] = config.model_dump()
            meta["name"] = config.name
            meta["algo"] = config.algo
        else:
            meta["config"] = config
        self.write_meta(exp_id, meta)
        return meta

    def append_metric(self, exp_id: str, row: dict[str, Any]) -> None:
        _append_jsonl(self.metrics_path(exp_id), row)
        # Soft-update counters without fighting concurrent writers too hard
        try:
            meta = self.read_meta(exp_id)
            dirty = False
            if "env_steps" in row:
                meta["env_steps"] = int(row["env_steps"])
                dirty = True
            if "eval_score_mean" in row:
                best = meta.get("best_eval_score")
                score = float(row["eval_score_mean"])
                if best is None or score > float(best):
                    meta["best_eval_score"] = score
                    dirty = True
            if dirty:
                self.write_meta(exp_id, meta)
        except (PermissionError, OSError, json.JSONDecodeError):
            log.debug("append_metric: soft meta update skipped for %s", exp_id, exc_info=True)

    def append_event(self, exp_id: str, event: dict[str, Any]) -> dict[str, Any]:
        if "t" not in event:
            event = {**event, "t": time.time()}
        _append_jsonl(self.events_path(exp_id), event)
        return event

    def read_metrics(self, exp_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.metrics_path(exp_id))
        if limit is not None:
            rows = downsample_evenly(rows, limit)
        return rows

    def read_events(self, exp_id: str) -> list[dict[str, Any]]:
        return _read_jsonl(self.events_path(exp_id))

    def read_config(self, exp_id: str) -> ExperimentConfig:
        meta = self.read_meta(exp_id)
        return ExperimentConfig.model_validate(meta["config"])

    def mark_stale_workers_stopped(self) -> list[str]:
        """On server start: running/paused without a live worker -> stopped."""
        changed: list[str] = []
        for eid in self.list_ids():
            meta = self.read_meta(eid)
            if meta.get("status") in ("running", "paused"):
                meta["status"] = "stopped"
                meta["error"] = None
                self.write_meta(eid, meta)
                changed.append(eid)
        return changed

    def delete(self, exp_id: str) -> None:
        d = self.exp_dir(exp_id)
        if d.is_dir():
            shutil.rmtree(d)

    def clone(
        self,
        exp_id: str,
        *,
        name: str,
        with_weights: bool,
    ) -> dict[str, Any]:
        src = self.read_meta(exp_id)
        cfg = ExperimentConfig.model_validate(src["config"])
        cfg = cfg.model_copy(update={"name": name})
        meta = self.create(cfg, parent_id=exp_id)
        if with_weights:
            for ckpt_name in ("latest", "best"):
                src_path = self.ckpt_path(exp_id, ckpt_name)
                if src_path.is_file():
                    dst = self.ckpt_path(meta["id"], ckpt_name)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst)
        return meta

    def checkpoint_infos(self, exp_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name in ("latest", "best"):
            path = self.ckpt_path(exp_id, name)
            if not path.is_file():
                continue
            env_steps = 0
            try:
                import torch

                data = torch.load(path, map_location="cpu", weights_only=False)
                trainer = data.get("trainer", {})
                env_steps = int(trainer.get("env_steps", data.get("meta", {}).get("env_steps", 0)))
            except Exception:
                log.warning("checkpoint_infos: failed reading %s", path, exc_info=True)
                env_steps = 0
            mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            out.append({"name": name, "env_steps": env_steps, "saved_at": mtime})
        return out

    def summary(self, exp_id: str) -> dict[str, Any]:
        meta = self.read_meta(exp_id)
        cfg = meta.get("config") or {}
        env = cfg.get("env") or {}
        metrics = self.read_metrics(exp_id)
        last_row = metrics[-1] if metrics else None
        last = None
        if last_row is not None:
            last = {
                "score_mean": float(last_row.get("score_mean", 0.0) or 0.0),
                "score_max": float(last_row.get("score_max", 0.0) or 0.0),
                "sps": float(last_row.get("sps", 0.0) or 0.0),
            }
        return {
            "id": meta["id"],
            "name": meta["name"],
            "algo": meta["algo"],
            "status": meta["status"],
            "created_at": meta["created_at"],
            "board": [int(env.get("min_size", 8)), int(env.get("max_size", 8))],
            "env_steps": int(meta.get("env_steps") or 0),
            "elapsed_s": float(meta.get("elapsed_s") or 0.0),
            "best_eval_score": meta.get("best_eval_score"),
            "last": last,
            "spark": sparkline(metrics),
        }

    def detail(self, exp_id: str, *, metrics_limit: int = 3000) -> dict[str, Any]:
        meta = self.read_meta(exp_id)
        experiment = {
            "id": meta["id"],
            "name": meta["name"],
            "algo": meta["algo"],
            "created_at": meta["created_at"],
            "status": meta["status"],
            "config": meta["config"],
            "parent_id": meta.get("parent_id"),
            "notes": meta.get("notes"),
            "error": meta.get("error"),
        }
        return {
            "experiment": experiment,
            "metrics": self.read_metrics(exp_id, limit=metrics_limit),
            "events": self.read_events(exp_id),
        }
