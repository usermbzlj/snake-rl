"""
贪吃蛇 AI Web 控制台：FastAPI + WebSocket，替代原 tkinter GUI。

启动：snake-webui  或  uv run python -m snake_rl.web_server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .form_field_tips import form_meta
from .process_supervisor import terminate_process
from .run_meta import list_run_metas_sorted, run_meta_to_gui_row
from .schemes import SCHEME_INFO, default_custom_train_config, get_config
from .train_config_json import parse_and_validate_train_config_json, train_config_to_json_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
DOCS_DIR = PROJECT_ROOT / "docs"
RUNS_DIR = PROJECT_ROOT / "runs"
GUI_STATE_PATH = PROJECT_ROOT / ".snake_gui_state.json"

EPISODE_LINE_RE = re.compile(r"\[Episode\s+(\d+)\]")
STAGE_LINE_RE = re.compile(r"\[Stage\s+(\d+)\s+\|\s+Ep\s+(\d+)/(\d+)\]")
STAGE_HEADER_RE = re.compile(r"Curriculum Stage\s+(\d+)/(\d+)")
TOTAL_EPISODES_RE = re.compile(r"总局数上限[：:]\s*(\d+)")
AVG_REWARD_RE = re.compile(r"avg_reward=\s*([-+]?\d+(?:\.\d+)?)")
EPSILON_RE = re.compile(r"\beps=\s*([-+]?\d+(?:\.\d+)?)")

def _load_gui_state() -> dict[str, Any]:
    try:
        if GUI_STATE_PATH.exists():
            data = json.loads(GUI_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_gui_state(data: dict[str, Any]) -> None:
    try:
        GUI_STATE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _default_custom_path() -> str:
    return str(PROJECT_ROOT / "custom_train_config.json")


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
        self.user_stop_training = False
        self.estimating = False

        gs = _load_gui_state()
        self.scheme = str(gs.get("scheme", "custom"))
        if self.scheme not in SCHEME_INFO:
            self.scheme = "custom"
        self.parallel = bool(gs.get("parallel", False))
        self.parallel_workers = max(1, min(64, int(gs.get("parallel_workers", 4))))
        self.parallel_sync = max(16, min(100000, int(gs.get("parallel_sync_interval", 512))))
        self.custom_config_path = str(gs.get("custom_config_path", _default_custom_path())).strip() or _default_custom_path()
        self.monitor_port = max(1024, min(65535, int(gs.get("monitor_port", 6006))))
        self.inference_port = max(1024, min(65535, int(gs.get("inference_port", 8765))))

        self.progress_total = 0
        self.progress_current = 0
        self.progress_stage = "-"
        self.progress_avg_reward: float | None = None
        self.progress_epsilon: float | None = None
        self.stage_prefix: dict[int, int] = {}

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
        base = _load_gui_state()
        base.update(self.snapshot_gui_state())
        _save_gui_state(base)

    def training_alive(self) -> bool:
        with self._lock:
            return self.training_proc is not None and self.training_proc.poll() is None

    def monitor_alive(self) -> bool:
        with self._lock:
            return self.monitor_proc is not None and self.monitor_proc.poll() is None

    def infer_alive(self) -> bool:
        with self._lock:
            return self.inference_proc is not None and self.inference_proc.poll() is None


state = RuntimeState()
manager = ConnectionManager()
_main_loop: asyncio.AbstractEventLoop | None = None


def _schedule_coro(coro: Any) -> None:
    loop = _main_loop
    if loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception:
        pass


def _win_flags() -> int:
    return subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0


def _tcp_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200)) == 200
    except Exception:
        return False


def _prepare_progress(scheme: str, custom_path: str) -> None:
    state.progress_total = 0
    state.progress_current = 0
    state.progress_stage = "-"
    state.progress_avg_reward = None
    state.progress_epsilon = None
    state.stage_prefix.clear()
    try:
        if scheme == "custom":
            cfg = get_config(scheme="custom", custom_config_path=Path(custom_path))
        else:
            cfg = get_config(scheme=scheme)
        if cfg.curriculum is not None and cfg.curriculum.stages:
            total = 0
            for idx, st in enumerate(cfg.curriculum.stages, start=1):
                state.stage_prefix[idx] = total
                total += int(st.episodes)
            state.progress_total = max(0, total)
        else:
            state.progress_total = max(0, int(cfg.episodes))
    except Exception:
        state.progress_total = 0


def _update_progress_from_line(line: str) -> None:
    total_match = TOTAL_EPISODES_RE.search(line)
    if total_match:
        try:
            state.progress_total = max(state.progress_total, int(total_match.group(1)))
        except Exception:
            pass

    episode_match = EPISODE_LINE_RE.search(line)
    if episode_match:
        try:
            state.progress_current = max(state.progress_current, int(episode_match.group(1)))
        except Exception:
            pass

    stage_match = STAGE_LINE_RE.search(line)
    if stage_match:
        try:
            stage_index = int(stage_match.group(1))
            stage_episode = int(stage_match.group(2))
            stage_total = int(stage_match.group(3))
            prefix = state.stage_prefix.get(stage_index)
            if prefix is not None:
                state.progress_current = max(
                    state.progress_current,
                    prefix + stage_episode,
                )
            state.progress_stage = f"{stage_index} ({stage_episode}/{stage_total})"
        except Exception:
            pass
    else:
        stage_header_match = STAGE_HEADER_RE.search(line)
        if stage_header_match:
            state.progress_stage = f"{stage_header_match.group(1)} (准备中)"

    avg_match = AVG_REWARD_RE.search(line)
    if avg_match:
        try:
            state.progress_avg_reward = float(avg_match.group(1))
        except Exception:
            pass

    eps_match = EPSILON_RE.search(line)
    if eps_match:
        try:
            state.progress_epsilon = float(eps_match.group(1))
        except Exception:
            pass


def _progress_payload() -> dict[str, Any]:
    total = state.progress_total
    cur = state.progress_current
    pct = 0.0
    if total > 0:
        pct = max(0.0, min(100.0, 100.0 * cur / total))
    return {
        "type": "progress",
        "episode": cur,
        "total": total,
        "percent": pct,
        "avg_reward": state.progress_avg_reward,
        "epsilon": state.progress_epsilon,
        "stage": state.progress_stage,
    }


def _status_payload() -> dict[str, Any]:
    return {
        "type": "status",
        "training": state.training_alive(),
        "monitor": state.monitor_alive(),
        "infer": state.infer_alive(),
        "monitor_port": state.monitor_port,
        "infer_port": state.inference_port,
        "estimating": state.estimating,
    }


def _ensure_default_config_file() -> None:
    p = Path(state.custom_config_path)
    if not p.is_file():
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(train_config_to_json_text(default_custom_train_config()), encoding="utf-8")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    _ensure_default_config_file()
    yield
    for proc_attr in ("training_proc", "monitor_proc", "inference_proc"):
        with state._lock:
            proc = getattr(state, proc_attr)
        if proc is not None and proc.poll() is None:
            terminate_process(proc, timeout_s=2.0)


app = FastAPI(title="Snake RL Web UI", lifespan=lifespan)

if DOCS_DIR.is_dir():
    app.mount("/doc-static", StaticFiles(directory=str(DOCS_DIR)), name="doc_static")
if WEB_DIR.is_dir():
    app.mount("/play", StaticFiles(directory=str(WEB_DIR), html=True), name="play")


@app.get("/")
async def index() -> FileResponse:
    html = WEB_DIR / "app.html"
    if not html.is_file():
        raise HTTPException(500, "web/app.html 缺失")
    return FileResponse(html)


@app.get("/api/form-meta")
async def api_form_meta() -> dict[str, Any]:
    return form_meta()


@app.get("/api/schemes")
async def api_schemes() -> dict[str, str]:
    return dict(SCHEME_INFO)


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    return {
        **state.snapshot_gui_state(),
        "training": state.training_alive(),
        "monitor": state.monitor_alive(),
        "infer": state.infer_alive(),
        "estimating": state.estimating,
        "progress": _progress_payload(),
    }


@app.get("/api/config")
async def api_config_get() -> dict[str, Any]:
    _ensure_default_config_file()
    p = Path(state.custom_config_path)
    if not p.is_file():
        raise HTTPException(404, f"配置文件不存在: {p}")
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("根不是对象")
    except Exception as exc:
        raise HTTPException(400, f"读取失败: {exc}") from exc
    return {"path": str(p), "config": data}


@app.post("/api/config")
async def api_config_post(body: dict[str, Any]) -> dict[str, str]:
    cfg_obj = body.get("config")
    if not isinstance(cfg_obj, dict):
        raise HTTPException(400, "body.config 必须是对象")
    path_str = body.get("path") or state.custom_config_path
    p = Path(str(path_str))
    try:
        raw = json.dumps(cfg_obj, ensure_ascii=False, indent=2)
        parse_and_validate_train_config_json(raw)
    except Exception as exc:
        raise HTTPException(400, f"校验失败: {exc}") from exc
    try:
        if not p.parent.is_dir():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(raw, encoding="utf-8")
    except Exception as exc:
        raise HTTPException(500, f"写入失败: {exc}") from exc
    with state._lock:
        state.custom_config_path = str(p.resolve())
    state.persist()
    return {"ok": "true", "path": str(p)}


@app.post("/api/ui-settings")
async def api_ui_settings(body: dict[str, Any]) -> dict[str, str]:
    """更新 scheme / parallel / ports / custom_config_path（不写 TrainConfig）。"""
    with state._lock:
        if "scheme" in body and str(body["scheme"]) in SCHEME_INFO:
            state.scheme = str(body["scheme"])
        if "parallel" in body:
            state.parallel = bool(body["parallel"])
        if "parallel_workers" in body:
            state.parallel_workers = max(1, min(64, int(body["parallel_workers"])))
        if "parallel_sync_interval" in body:
            state.parallel_sync = max(16, min(100000, int(body["parallel_sync_interval"])))
        if "custom_config_path" in body:
            state.custom_config_path = str(body["custom_config_path"]).strip() or _default_custom_path()
        if "monitor_port" in body:
            state.monitor_port = max(1024, min(65535, int(body["monitor_port"])))
        if "inference_port" in body:
            state.inference_port = max(1024, min(65535, int(body["inference_port"])))
    state.persist()
    _schedule_coro(manager.broadcast_json(_status_payload()))
    return {"ok": "true"}


@app.post("/api/train/start")
async def api_train_start() -> dict[str, str]:
    if state.training_alive():
        raise HTTPException(409, "训练已在进行中")
    scheme = state.scheme
    if scheme == "custom":
        cp = Path(state.custom_config_path)
        if not cp.is_file():
            raise HTTPException(400, "custom 模式需要有效的配置文件路径")
        try:
            parse_and_validate_train_config_json(cp.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(400, f"配置校验失败: {exc}") from exc

    _prepare_progress(scheme, state.custom_config_path)
    _schedule_coro(manager.broadcast_json(_progress_payload()))

    cmd = [sys.executable, "-m", "snake_rl.cli", "train", "--scheme", scheme]
    if scheme == "custom":
        cmd.extend(["--custom-config", str(Path(state.custom_config_path).resolve())])
    if state.parallel:
        cmd.extend(
            [
                "--parallel",
                "--parallel-workers",
                str(state.parallel_workers),
                "--parallel-sync-interval",
                str(state.parallel_sync),
            ]
        )

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            creationflags=_win_flags(),
        )
    except Exception as exc:
        raise HTTPException(500, f"启动失败: {exc}") from exc

    with state._lock:
        state.training_proc = proc
        state.user_stop_training = False
    state.persist()
    async def _start_logs() -> None:
        await manager.broadcast_json(_status_payload())
        await manager.broadcast_json({"type": "log", "text": f"[训练] 启动: {' '.join(cmd)}"})

    _schedule_coro(_start_logs())

    threading.Thread(target=_training_reader_thread, args=(proc,), daemon=True).start()
    return {"ok": "true"}


def _training_reader_thread(proc: subprocess.Popen) -> None:
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n\r")
            if not stripped:
                continue
            _update_progress_from_line(stripped)

            async def _push() -> None:
                await manager.broadcast_json({"type": "log", "text": stripped})
                await manager.broadcast_json(_progress_payload())

            _schedule_coro(_push())
    except Exception as exc:
        _schedule_coro(
            manager.broadcast_json({"type": "log", "text": f"[训练] 读取输出失败: {exc}"})
        )
    finally:
        code = proc.wait()
        with state._lock:
            state.training_proc = None
            user_stop = state.user_stop_training
            state.user_stop_training = False
        if code == 0 and state.progress_total > 0 and not user_stop:
            state.progress_current = state.progress_total
        msg = (
            "训练已中断（用户停止）"
            if user_stop
            else ("训练完成" if code == 0 else f"训练结束 (exit {code})")
        )
        async def _done() -> None:
            await manager.broadcast_json({"type": "log", "text": f"[训练] {msg}"})
            await manager.broadcast_json(_progress_payload())
            await manager.broadcast_json(_status_payload())
            await manager.broadcast_json({"type": "runs_reload"})

        _schedule_coro(_done())


@app.post("/api/train/stop")
async def api_train_stop() -> dict[str, str]:
    with state._lock:
        proc = state.training_proc
    if proc is None or proc.poll() is not None:
        return {"ok": "true", "message": "未在训练"}
    state.user_stop_training = True
    terminate_process(proc)
    _schedule_coro(manager.broadcast_json({"type": "log", "text": "[训练] 正在停止…"}))
    return {"ok": "true"}


@app.get("/api/runs")
async def api_runs() -> list[dict[str, str]]:
    if not RUNS_DIR.is_dir():
        return []
    out: list[dict[str, str]] = []
    try:
        for meta in list_run_metas_sorted(RUNS_DIR):
            out.append(run_meta_to_gui_row(meta))
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return out


@app.post("/api/runs/{name}/reveal")
async def api_run_reveal(name: str) -> dict[str, str]:
    run_dir = RUNS_DIR / name
    if not run_dir.is_dir():
        raise HTTPException(404, "运行不存在")
    try:
        if os.name == "nt":
            os.startfile(str(run_dir))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(run_dir)])
        else:
            subprocess.Popen(["xdg-open", str(run_dir)])
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": "true"}


@app.delete("/api/runs/{name}")
async def api_run_delete(name: str) -> dict[str, str]:
    run_dir = RUNS_DIR / name
    if not run_dir.is_dir():
        raise HTTPException(404, "运行不存在")
    try:
        shutil.rmtree(run_dir)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": "true"}


@app.post("/api/runs/clear-all")
async def api_runs_clear_all(body: dict[str, Any]) -> dict[str, str]:
    if not body.get("confirm"):
        raise HTTPException(400, "需要 JSON body: {\"confirm\": true}")
    try:
        if RUNS_DIR.is_dir():
            shutil.rmtree(RUNS_DIR)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": "true"}


def _service_stderr_reader(proc: subprocess.Popen, tag: str) -> None:
    def read() -> None:
        try:
            if proc.stderr is not None:
                for line in proc.stderr:
                    s = line.rstrip("\n\r")
                    if s:
                        _schedule_coro(
                            manager.broadcast_json({"type": "log", "text": f"[{tag}] {s}"})
                        )
        except Exception as exc:
            _schedule_coro(
                manager.broadcast_json({"type": "log", "text": f"[{tag}] stderr 异常: {exc}"})
            )

    threading.Thread(target=read, daemon=True).start()


def _popen_service(args: list[str], tag: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        args,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        creationflags=_win_flags(),
    )
    _service_stderr_reader(proc, tag)
    return proc


@app.post("/api/monitor/start")
async def api_monitor_start() -> dict[str, Any]:
    base = f"http://127.0.0.1:{state.monitor_port}"
    health = f"{base}/"

    if state.monitor_alive():
        return {"ok": "true", "url": f"{base}/", "already": True}

    if _tcp_port_open("127.0.0.1", state.monitor_port) and _http_ok(health):
        return {"ok": "true", "url": f"{base}/", "external": True}

    if _tcp_port_open("127.0.0.1", state.monitor_port) and not _http_ok(health):
        raise HTTPException(409, f"端口 {state.monitor_port} 被占用且非可访问 Web 服务")

    try:
        proc = _popen_service(
            [
                sys.executable,
                "-m",
                "snake_rl.cli",
                "monitor",
                "--port",
                str(state.monitor_port),
            ],
            "monitor",
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    with state._lock:
        state.monitor_proc = proc
    state.persist()
    if proc.poll() is not None:
        with state._lock:
            state.monitor_proc = None
        raise HTTPException(500, "TensorBoard 进程立即退出，请查看日志")
    _schedule_coro(manager.broadcast_json(_status_payload()))
    return {"ok": "true", "url": f"{base}/"}


@app.post("/api/monitor/stop")
async def api_monitor_stop() -> dict[str, str]:
    with state._lock:
        proc = state.monitor_proc
        state.monitor_proc = None
    if proc is not None and proc.poll() is None:
        terminate_process(proc)
    _schedule_coro(manager.broadcast_json(_status_payload()))
    return {"ok": "true"}


@app.post("/api/infer/start")
async def api_infer_start(body: dict[str, Any]) -> dict[str, Any]:
    run_name = str(body.get("run_name", "")).strip()
    if not run_name:
        raise HTTPException(400, "需要 run_name")
    run_dir = RUNS_DIR / run_name
    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.is_file():
        ckpt = run_dir / "checkpoints" / "latest.pt"
    if not ckpt.is_file():
        raise HTTPException(404, "未找到 best.pt / latest.pt")

    port = state.inference_port
    health_url = f"http://127.0.0.1:{port}/health"

    if _tcp_port_open("127.0.0.1", port):
        if _http_ok(health_url):
            return {
                "ok": "true",
                "play_url": "/play/",
                "health": health_url,
                "already": True,
            }
        raise HTTPException(409, f"端口 {port} 被占用且非本推理服务")

    with state._lock:
        old = state.inference_proc
    if old is not None and old.poll() is None:
        terminate_process(old)
    with state._lock:
        state.inference_proc = None

    try:
        proc = _popen_service(
            [
                sys.executable,
                "-m",
                "snake_rl.cli",
                "serve-model",
                "--port",
                str(port),
                "--checkpoint",
                str(ckpt.resolve()),
            ],
            "infer",
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    with state._lock:
        state.inference_proc = proc
    state.persist()
    if proc.poll() is not None:
        with state._lock:
            state.inference_proc = None
        raise HTTPException(500, "推理进程立即退出")
    _schedule_coro(manager.broadcast_json(_status_payload()))
    return {"ok": "true", "play_url": "/play/", "checkpoint": str(ckpt)}


@app.post("/api/infer/stop")
async def api_infer_stop() -> dict[str, str]:
    with state._lock:
        proc = state.inference_proc
        state.inference_proc = None
    if proc is not None and proc.poll() is None:
        terminate_process(proc)
    _schedule_coro(manager.broadcast_json(_status_payload()))
    return {"ok": "true"}


@app.post("/api/estimate/start")
async def api_estimate_start() -> dict[str, str]:
    if state.estimating:
        raise HTTPException(409, "已有估算任务在进行中")
    scheme = state.scheme
    if scheme == "custom":
        cp = Path(state.custom_config_path)
        if not cp.is_file():
            raise HTTPException(400, "custom 模式需要有效配置文件")

    state.estimating = True
    _schedule_coro(manager.broadcast_json(_status_payload()))
    threading.Thread(target=_estimate_worker, daemon=True).start()
    return {"ok": "true"}


def _estimate_worker() -> None:
    try:
        cmd = [sys.executable, "-u", "-m", "snake_rl.cli", "estimate", "--scheme", state.scheme]
        if state.scheme == "custom":
            cmd.extend(["--custom-config", str(Path(state.custom_config_path).resolve())])
        if state.parallel:
            cmd.extend(
                [
                    "--parallel",
                    "--parallel-workers",
                    str(state.parallel_workers),
                    "--parallel-sync-interval",
                    str(state.parallel_sync),
                ]
            )
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                s = line.rstrip("\n\r")
                if s:
                    _schedule_coro(manager.broadcast_json({"type": "log", "text": s}))
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            terminate_process(proc)
            _schedule_coro(
                manager.broadcast_json({"type": "log", "text": "[估算] 超时已终止"})
            )
    except Exception as exc:
        _schedule_coro(
            manager.broadcast_json({"type": "log", "text": f"[估算] 失败: {exc}"})
        )
    finally:
        state.estimating = False

        async def _est_done() -> None:
            await manager.broadcast_json({"type": "log", "text": "[估算] --- 结束 ---"})
            await manager.broadcast_json(_status_payload())

        _schedule_coro(_est_done())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        await websocket.send_json(_status_payload())
        await websocket.send_json(_progress_payload())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def main() -> None:
    parser = argparse.ArgumentParser(description="Snake RL Web 控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
