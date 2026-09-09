"""
贪吃蛇 AI Web 控制台：FastAPI + WebSocket，替代原 tkinter GUI。

启动：snake-webui  或  uv run python -m snake_rl.web_server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..form_field_tips import form_meta
from ..netutil import get_lan_ip
from ..process_supervisor import popen_service, terminate_process, win_creation_flags
from ..run_meta import build_run_meta, list_run_metas_sorted, run_meta_to_api_item, run_meta_to_gui_row
from ..run_context import checkpoint_run_dir
from ..schemes import SCHEME_INFO, default_custom_train_config, get_config
from ..train_config_json import parse_and_validate_train_config_json, train_config_to_json_text
from .monitor import monitor_logdir_for, normalize_run_ref, should_restart_monitor
from .net import http_get_json, http_ok, tcp_port_open, wait_for_port_closed
from .paths import DOCS_DIR, PROJECT_ROOT, RUNS_DIR, WEB_DIR, default_custom_path
from .progress import progress_payload, status_payload, update_progress_from_line
from .state import ConnectionManager, RuntimeState

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


def _progress_payload() -> dict[str, Any]:
    return progress_payload(state)


def _status_payload() -> dict[str, Any]:
    return status_payload(state)


def _ensure_runs_mutable() -> None:
    if state.training_alive():
        raise HTTPException(409, "训练进行中时不能删除或清空运行记录")


def _training_output_run_dir(scheme: str, custom_config_path: str) -> Path | None:
    try:
        cp = Path(custom_config_path) if scheme == "custom" else None
        cfg = get_config(scheme=scheme, custom_config_path=cp)
        root = cfg.output_root
        if not root.is_absolute():
            root = (PROJECT_ROOT / root).resolve()
        return (root / cfg.run_name).resolve()
    except Exception:
        return None


def _monitor_stop_sync() -> None:
    with state._lock:
        proc = state.monitor_proc
        state.monitor_proc = None
        state.monitor_logdir = None
    if proc is not None and proc.poll() is None:
        terminate_process(proc)


def _resolve_run_dir(name: str) -> Path:
    raw = normalize_run_ref(name)
    if not raw:
        raise HTTPException(400, "运行名不能为空")
    try:
        return monitor_logdir_for(raw, RUNS_DIR)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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
    for proc_attr in ("training_proc", "monitor_proc", "inference_proc", "estimate_proc"):
        with state._lock:
            proc = getattr(state, proc_attr)
        if proc is not None and proc.poll() is None:
            terminate_process(proc, timeout_s=2.0)


app = FastAPI(title="Snake RL Web UI", lifespan=lifespan)

if DOCS_DIR.is_dir():
    app.mount("/doc-static", StaticFiles(directory=str(DOCS_DIR)), name="doc_static")
if WEB_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")
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
        "lan_ip": get_lan_ip(),
        "training_run": Path(state.training_run_dir).name if state.training_run_dir else None,
        "train_started_at": state.train_started_at,
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
            v = body["parallel_workers"]
            if v != "":
                state.parallel_workers = max(1, min(64, int(v)))
        if "parallel_sync_interval" in body:
            v = body["parallel_sync_interval"]
            if v != "":
                state.parallel_sync = max(16, min(100000, int(v)))
        if "custom_config_path" in body:
            state.custom_config_path = str(body["custom_config_path"]).strip() or default_custom_path()
        if "monitor_port" in body:
            v = body["monitor_port"]
            if v != "":
                state.monitor_port = max(1024, min(65535, int(v)))
        if "inference_port" in body:
            v = body["inference_port"]
            if v != "":
                state.inference_port = max(1024, min(65535, int(v)))
    state.persist()
    _schedule_coro(manager.broadcast_json(_status_payload()))
    return {"ok": "true"}


@app.post("/api/train/start")
async def api_train_start(body: dict[str, Any] | None = None) -> dict[str, str]:
    if body is None:
        body = {}
    if state.training_alive():
        raise HTTPException(409, "训练已在进行中")

    tb_was_on = state.monitor_alive()
    if tb_was_on:
        _monitor_stop_sync()

    resume_state_path = str(body.get("resume_state", "")).strip()
    warm_start_path = str(body.get("warm_start", "")).strip()
    if resume_state_path and warm_start_path:
        raise HTTPException(400, "不能同时使用完整恢复和热加载")
    if resume_state_path:
        rp = Path(resume_state_path)
        if not rp.is_file():
            raise HTTPException(400, f"恢复文件不存在: {rp}")
    if warm_start_path:
        wp = Path(warm_start_path)
        if not wp.is_file():
            raise HTTPException(400, f"权重文件不存在: {wp}")

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
    if resume_state_path:
        cmd.extend(["--resume-state", str(Path(resume_state_path).resolve())])
        extra_ep = body.get("extra_episodes")
        if extra_ep is not None:
            extra_ep = int(extra_ep)
            if extra_ep > 0:
                cmd.extend(["--extra-episodes", str(extra_ep)])
    elif warm_start_path:
        cmd.extend(["--warm-start", str(Path(warm_start_path).resolve())])
        wsg_raw = body.get("warm_start_global_step")
        if wsg_raw is not None and str(wsg_raw).strip() != "":
            cmd.extend(["--warm-start-global-step", str(int(wsg_raw))])
        elif body.get("warm_start_inherit_global_step") is False:
            cmd.extend(["--warm-start-global-step", "0"])
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
            creationflags=win_creation_flags(),
        )
    except Exception as exc:
        raise HTTPException(500, f"启动失败: {exc}") from exc

    with state._lock:
        state.training_proc = proc
        state.user_stop_training = False
        state.train_started_at = time.time()
        try:
            if resume_state_path:
                # resume_state 位于 runs/<name>/state/training.pt，向上两级即 run 目录
                state.training_run_dir = str(Path(resume_state_path).resolve().parent.parent)
            else:
                tr = _training_output_run_dir(scheme, state.custom_config_path)
                state.training_run_dir = str(tr) if tr is not None else None
        except Exception:
            state.training_run_dir = None
    state.persist()

    mode_label = "完整恢复" if resume_state_path else ("热加载" if warm_start_path else "全新训练")

    async def _start_logs() -> None:
        await manager.broadcast_json(_status_payload())
        await manager.broadcast_json(
            {"type": "log", "text": f"[训练] [{mode_label}] 启动: {' '.join(cmd)}"}
        )

    _schedule_coro(_start_logs())

    threading.Thread(target=_training_reader_thread, args=(proc,), daemon=True).start()

    if tb_was_on:

        async def _restart_tb() -> None:
            await asyncio.sleep(1.2)
            try:
                await api_monitor_start()
            except HTTPException:
                pass
            except Exception:
                pass

        asyncio.create_task(_restart_tb())

    return {"ok": "true"}


def _training_reader_thread(proc: subprocess.Popen) -> None:
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n\r")
            if not stripped:
                continue
            update_progress_from_line(state, stripped)

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
            state.training_run_dir = None
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


@app.get("/api/train/live-preview-info")
async def api_train_live_preview_info(run: str | None = None) -> dict[str, Any]:
    if run:
        run_path = _resolve_run_dir(run)
    else:
        rd = state.training_run_dir
        if not rd:
            raise HTTPException(503, "无法解析训练输出目录，请指定 ?run=<run_name>")
        run_path = Path(rd)
    if not run_path.is_dir():
        raise HTTPException(404, "训练目录不存在")
    ckpt_dir = run_path / "checkpoints"
    if not ckpt_dir.is_dir():
        raise HTTPException(404, "尚无 checkpoints 目录")
    pts = [p for p in ckpt_dir.glob("*.pt") if p.is_file()]
    if not pts:
        raise HTTPException(404, "尚无权重文件")
    latest = max(pts, key=lambda p: p.stat().st_mtime)
    model_type: str | None = None
    rc = run_path / "run_config.json"
    if rc.is_file():
        try:
            model_type = json.loads(rc.read_text(encoding="utf-8")).get("model_type")
            if model_type is not None:
                model_type = str(model_type)
        except Exception:
            model_type = None
    return {
        "ok": True,
        "run_name": run_path.name,
        "run_dir": str(run_path.resolve()),
        "checkpoint": str(latest.resolve()),
        "mtime": latest.stat().st_mtime,
        "model_type": model_type,
        "inference_port": state.inference_port,
    }


@app.post("/api/checkpoint/inspect")
async def api_checkpoint_inspect(body: dict[str, Any]) -> dict[str, Any]:
    """
    智能识别一个 .pt 路径：是否存在、是否属于本项目历史训练 run、推荐什么操作。
    输入：{"path": "<checkpoint 路径>"}
    """
    raw = str(body.get("path", "")).strip()
    if not raw:
        raise HTTPException(400, "需要 path 字段")

    ckpt_path = Path(raw).expanduser()
    # 尝试解析为绝对路径
    if not ckpt_path.is_absolute():
        ckpt_path = (PROJECT_ROOT / ckpt_path).resolve()
    else:
        ckpt_path = ckpt_path.resolve()

    result: dict[str, Any] = {
        "path": str(ckpt_path),
        "exists": ckpt_path.is_file(),
        "is_pt": ckpt_path.suffix.lower() == ".pt",
        "run": None,
        "recommendation": None,
        "hint": "",
    }

    if not result["exists"]:
        result["hint"] = "文件不存在，请确认路径是否正确。"
        result["recommendation"] = "invalid"
        return result

    if not result["is_pt"]:
        result["hint"] = "不是 .pt 文件，可能无法作为模型 checkpoint 加载。"
        result["recommendation"] = "unknown"
        return result

    # 尝试关联到本项目 runs/ 下的历史 run
    run_dir = checkpoint_run_dir(ckpt_path)
    runs_root = RUNS_DIR.resolve()
    if run_dir is not None:
        try:
            run_dir.relative_to(runs_root)
            is_in_project = True
        except ValueError:
            is_in_project = False
    else:
        is_in_project = False

    if run_dir is not None and is_in_project:
        try:
            meta = build_run_meta(run_dir)
            run_info = run_meta_to_api_item(meta)
            run_info["name"] = run_dir.name
            run_info["run_dir"] = str(run_dir)
            result["run"] = run_info

            ckpt_name = ckpt_path.name
            status_key = meta.status_key
            has_state = (run_dir / "state" / "training.pt").is_file()

            if status_key == "completed":
                if ckpt_name == "best.pt":
                    result["recommendation"] = "demo"
                    result["hint"] = f"这是「{run_dir.name}」训练完成后的最佳模型，适合直接演示。"
                else:
                    result["recommendation"] = "demo"
                    result["hint"] = f"这是「{run_dir.name}」训练完成后的权重，可直接演示。"
            elif status_key == "interrupted":
                if has_state and ckpt_name in ("latest.pt", "best.pt"):
                    result["recommendation"] = "resume"
                    result["hint"] = (
                        f"「{run_dir.name}」上次训练中断，共 {meta.episodes_logged} 局。"
                        f" 可「原样继续练」（完整恢复），也可直接演示当前权重。"
                    )
                else:
                    result["recommendation"] = "warm"
                    result["hint"] = (
                        f"「{run_dir.name}」上次训练中断（{meta.episodes_logged} 局），"
                        f" 可「继承权重继续练」（热加载），也可直接演示。"
                    )
            elif status_key == "running":
                result["recommendation"] = "demo"
                result["hint"] = f"「{run_dir.name}」正在训练中，可演示当前权重或等待训练完成。"
            else:
                result["recommendation"] = "demo"
                result["hint"] = f"这是「{run_dir.name}」的权重文件。"
        except Exception:
            result["run"] = {"name": run_dir.name, "run_dir": str(run_dir)}
            result["recommendation"] = "demo"
            result["hint"] = f"已找到关联训练 run「{run_dir.name}」，可直接演示。"
    else:
        result["recommendation"] = "demo_external"
        result["hint"] = "该权重不属于本项目 runs/ 目录，仍可尝试加载演示，但无法关联历史训练记录。"

    return result


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
    run_dir = _resolve_run_dir(name)
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


@app.get("/api/runs/{name}/artifacts")
async def api_run_artifacts(name: str) -> dict[str, Any]:
    """返回运行目录中可用的恢复文件与权重文件列表。"""
    run_dir = _resolve_run_dir(name)
    if not run_dir.is_dir():
        raise HTTPException(404, "运行不存在")
    result: dict[str, Any] = {"state_file": None, "checkpoints": [], "model_type": None}
    state_file = run_dir / "state" / "training.pt"
    if state_file.is_file():
        result["state_file"] = str(state_file.resolve())
    run_cfg = run_dir / "run_config.json"
    if run_cfg.is_file():
        try:
            rc = json.loads(run_cfg.read_text(encoding="utf-8"))
            result["model_type"] = rc.get("model_type")
        except Exception:
            pass
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        for f in sorted(ckpt_dir.glob("*.pt")):
            result["checkpoints"].append({"name": f.name, "path": str(f.resolve())})
    return result


@app.delete("/api/runs/{name}")
async def api_run_delete(name: str) -> dict[str, str]:
    _ensure_runs_mutable()
    run_dir = _resolve_run_dir(name)
    if not run_dir.is_dir():
        raise HTTPException(404, "运行不存在")
    try:
        shutil.rmtree(run_dir)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": "true"}


@app.post("/api/runs/clear-all")
async def api_runs_clear_all(body: dict[str, Any]) -> dict[str, str]:
    _ensure_runs_mutable()
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
    proc = popen_service(args, cwd=PROJECT_ROOT)
    _service_stderr_reader(proc, tag)
    return proc


@app.post("/api/monitor/start")
async def api_monitor_start(body: dict[str, Any] | None = None) -> dict[str, Any]:
    base = f"http://127.0.0.1:{state.monitor_port}"
    health = f"{base}/"
    if body is None:
        body = {}
    run_name = normalize_run_ref(body.get("run"))
    try:
        desired = monitor_logdir_for(run_name, RUNS_DIR)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    runs_dir = str(desired)

    if state.monitor_alive():
        if not should_restart_monitor(state.monitor_logdir, desired, True):
            return {"ok": "true", "url": f"{base}/", "already": True}
        _monitor_stop_sync()
        wait_for_port_closed("127.0.0.1", state.monitor_port)

    if tcp_port_open("127.0.0.1", state.monitor_port) and http_ok(health):
        if run_name:
            raise HTTPException(409, "端口上已有外部 TensorBoard，无法切换到指定 run")
        return {"ok": "true", "url": f"{base}/", "external": True}

    if tcp_port_open("127.0.0.1", state.monitor_port) and not http_ok(health):
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
                "--runs-dir",
                runs_dir,
            ],
            "monitor",
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    with state._lock:
        state.monitor_proc = proc
        state.monitor_logdir = runs_dir
    state.persist()
    if proc.poll() is not None:
        with state._lock:
            state.monitor_proc = None
            state.monitor_logdir = None
        raise HTTPException(500, "TensorBoard 进程立即退出，请查看日志")
    _schedule_coro(manager.broadcast_json(_status_payload()))
    return {"ok": "true", "url": f"{base}/"}


@app.post("/api/monitor/stop")
async def api_monitor_stop() -> dict[str, str]:
    with state._lock:
        proc = state.monitor_proc
        state.monitor_proc = None
        state.monitor_logdir = None
    if proc is not None and proc.poll() is None:
        terminate_process(proc)
    _schedule_coro(manager.broadcast_json(_status_payload()))
    return {"ok": "true"}


@app.post("/api/infer/start")
async def api_infer_start(body: dict[str, Any]) -> dict[str, Any]:
    run_name = str(body.get("run_name", "")).strip()
    if not run_name:
        raise HTTPException(400, "需要 run_name")
    run_dir = _resolve_run_dir(run_name)
    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.is_file():
        ckpt = run_dir / "checkpoints" / "latest.pt"
    if not ckpt.is_file():
        raise HTTPException(404, "未找到 best.pt / latest.pt")

    port = state.inference_port
    health_url = f"http://127.0.0.1:{port}/health"
    desired_checkpoint = ckpt.resolve()

    if tcp_port_open("127.0.0.1", port):
        health_payload = http_get_json(health_url)
        if health_payload is not None and health_payload.get("ok"):
            loaded_checkpoint = str(health_payload.get("checkpoint", "")).strip()
            if loaded_checkpoint:
                try:
                    if Path(loaded_checkpoint).expanduser().resolve() == desired_checkpoint:
                        return {
                            "ok": "true",
                            "play_url": "/play/",
                            "health": health_url,
                            "already": True,
                            "checkpoint": str(desired_checkpoint),
                        }
                except OSError:
                    pass
            with state._lock:
                old = state.inference_proc
            if old is not None and old.poll() is None:
                terminate_process(old)
                if not wait_for_port_closed("127.0.0.1", port):
                    raise HTTPException(409, "旧推理服务未能及时退出，请稍后重试")
                with state._lock:
                    state.inference_proc = None
            else:
                raise HTTPException(409, "推理端口已有其他模型服务，请先停止后再切换")
        else:
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


@app.post("/api/infer/ensure-running")
async def api_infer_ensure_running() -> dict[str, Any]:
    """确保推理服务在配置端口上运行（不预加载模型），供训练实况自动启动使用。"""
    port = state.inference_port
    health_url = f"http://127.0.0.1:{port}/health"

    if state.infer_alive():
        return {"ok": "true", "already": True, "port": port}

    if tcp_port_open("127.0.0.1", port):
        payload = http_get_json(health_url)
        if payload is not None and payload.get("ok"):
            return {"ok": "true", "already": True, "external": True, "port": port}
        raise HTTPException(409, f"端口 {port} 被占用且非本推理服务，请先停止占用该端口的进程")

    try:
        proc = _popen_service(
            [sys.executable, "-m", "snake_rl.cli", "serve-model", "--port", str(port)],
            "infer",
        )
    except Exception as exc:
        raise HTTPException(500, f"推理服务启动失败: {exc}") from exc

    with state._lock:
        state.inference_proc = proc
    state.persist()
    if proc.poll() is not None:
        with state._lock:
            state.inference_proc = None
        raise HTTPException(500, "推理进程立即退出，请查看日志")
    _schedule_coro(manager.broadcast_json(_status_payload()))
    return {"ok": "true", "port": port}


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


@app.post("/api/estimate/stop")
async def api_estimate_stop() -> dict[str, str]:
    proc = state.estimate_proc
    if proc is None:
        state.estimating = False
        return {"ok": "true"}
    terminate_process(proc)
    state.estimate_proc = None
    state.estimating = False
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
        state.estimate_proc = proc
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
        state.estimate_proc = None

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
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0 方便手机同网访问")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    open_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{open_host}:{args.port}/"
    lan = get_lan_ip()
    print(f"控制台  {url}")
    print(f"局域网  http://{lan}:{args.port}/")
    print(f"游戏页  http://{open_host}:{args.port}/play/")
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
