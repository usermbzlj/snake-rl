from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from ..schemes import SCHEME_INFO
from .paths import GUI_STATE_PATH, default_custom_path


def load_gui_state() -> dict[str, Any]:
    try:
        if GUI_STATE_PATH.exists():
            data = json.loads(GUI_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_gui_state(data: dict[str, Any]) -> None:
    try:
        GUI_STATE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._clients:
            self._clients.remove(websocket)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class RuntimeState:
    """进程与进度（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.training_proc: subprocess.Popen | None = None
        self.monitor_proc: subprocess.Popen | None = None
        self.inference_proc: subprocess.Popen | None = None
        self.estimate_proc: subprocess.Popen | None = None
        self.user_stop_training = False
        self.estimating = False
        self.monitor_logdir: str | None = None

        gs = load_gui_state()
        self.scheme = str(gs.get("scheme", "custom"))
        if self.scheme not in SCHEME_INFO:
            self.scheme = "custom"
        self.parallel = bool(gs.get("parallel", False))
        self.parallel_workers = max(1, min(64, int(gs.get("parallel_workers", 4))))
        self.parallel_sync = max(16, min(100000, int(gs.get("parallel_sync_interval", 512))))
        self.custom_config_path = str(gs.get("custom_config_path", default_custom_path())).strip() or default_custom_path()
        self.monitor_port = max(1024, min(65535, int(gs.get("monitor_port", 6006))))
        self.inference_port = max(1024, min(65535, int(gs.get("inference_port", 8765))))

        self.progress_total = 0
        self.progress_current = 0
        self.progress_stage = "-"
        self.progress_avg_reward: float | None = None
        self.progress_epsilon: float | None = None
        self.stage_prefix: dict[int, int] = {}
        self.training_run_dir: str | None = None
        self.train_started_at: float | None = None

    def snapshot_gui_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scheme": self.scheme,
                "parallel": self.parallel,
                "parallel_workers": self.parallel_workers,
                "parallel_sync_interval": self.parallel_sync,
                "custom_config_path": self.custom_config_path,
                "monitor_port": self.monitor_port,
                "inference_port": self.inference_port,
            }

    def persist(self) -> None:
        base = load_gui_state()
        base.update(self.snapshot_gui_state())
        save_gui_state(base)

    def training_alive(self) -> bool:
        with self._lock:
            return self.training_proc is not None and self.training_proc.poll() is None

    def monitor_alive(self) -> bool:
        with self._lock:
            return self.monitor_proc is not None and self.monitor_proc.poll() is None

    def infer_alive(self) -> bool:
        with self._lock:
            return self.inference_proc is not None and self.inference_proc.poll() is None
