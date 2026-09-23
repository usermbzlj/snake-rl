"""FastAPI application: REST + WebSocket + SPA static files."""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from snake_rl import __version__
from snake_rl.core.config import ExperimentConfig, ui_schema
from snake_rl.lab.inspect import compare as compare_episodes
from snake_rl.lab.inspect import inspect_episode
from snake_rl.lab.manager import ExperimentManager
from snake_rl.lab.storage import ExperimentStore
from snake_rl.lab.viewer import WatchSession

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _lan_urls(port: int) -> list[str]:
    urls: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = str(info[4][0])
            if ip.startswith("127."):
                continue
            urls.append(f"http://{ip}:{port}")
    except Exception:
        pass
    # Fallback: connect UDP to discover primary interface
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = str(s.getsockname()[0])
        s.close()
        u = f"http://{ip}:{port}"
        if u not in urls and not ip.startswith("127."):
            urls.insert(0, u)
    except Exception:
        pass
    return urls


def _device_info() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    name = ""
    if cuda:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA"
    else:
        name = "CPU"
    return {"cuda": cuda, "name": name}


class CreateBody(BaseModel):
    config: ExperimentConfig
    start: bool = True


class LiveBody(BaseModel):
    patch: dict[str, float] = Field(default_factory=dict)


class CloneBody(BaseModel):
    name: str
    with_weights: bool = False


class InspectBody(BaseModel):
    checkpoint: Literal["latest", "best"] = "latest"
    board_size: int = 8
    seed: int | None = None
    greedy: bool = True


class CompareEntry(BaseModel):
    experiment_id: str
    checkpoint: Literal["latest", "best"] = "latest"


class CompareBody(BaseModel):
    entries: list[CompareEntry]
    board_size: int = 8
    seed: int | None = None


def create_app(
    *,
    manager: ExperimentManager | None = None,
    experiments_root: Path | str | None = None,
    port: int = 7860,
) -> FastAPI:
    store = ExperimentStore(experiments_root) if experiments_root is not None else ExperimentStore()
    mgr = manager or ExperimentManager(store)
    app_state: dict[str, Any] = {"manager": mgr, "port": port}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        mgr.bind_loop(loop)
        mgr.startup()
        app.state.manager = mgr
        app.state.port = app_state["port"]
        try:
            yield
        finally:
            mgr.shutdown()

    app = FastAPI(title="贪吃蛇 AI 训练实验室", version=__version__, lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_exc_handler(_request: Request, exc: HTTPException):
        detail = exc.detail
        if not isinstance(detail, str):
            detail = str(detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    def _mgr() -> ExperimentManager:
        return app.state.manager

    def _not_found(exp_id: str) -> None:
        if not _mgr().store.exists(exp_id):
            raise HTTPException(404, f"实验不存在: {exp_id}")

    @app.get("/api/meta")
    def api_meta() -> dict[str, Any]:
        port_ = int(getattr(app.state, "port", port))
        return {
            "version": __version__,
            "device": _device_info(),
            "lan_urls": _lan_urls(port_),
            "port": port_,
        }

    @app.get("/api/config-schema")
    def api_schema() -> dict[str, Any]:
        return ui_schema()

    @app.get("/api/experiments")
    def api_list() -> list[dict[str, Any]]:
        return _mgr().list()

    @app.post("/api/experiments")
    def api_create(body: CreateBody) -> dict[str, Any]:
        try:
            return _mgr().create(body.config, start=body.start)
        except ValidationError as e:
            raise HTTPException(400, f"配置无效: {e}") from e
        except Exception as e:
            raise HTTPException(400, f"创建实验失败: {e}") from e

    @app.get("/api/experiments/{exp_id}")
    def api_get(exp_id: str) -> dict[str, Any]:
        _not_found(exp_id)
        return _mgr().get(exp_id)

    @app.patch("/api/experiments/{exp_id}/live")
    def api_live(exp_id: str, body: LiveBody) -> dict[str, Any]:
        _not_found(exp_id)
        try:
            return _mgr().live_patch(exp_id, body.patch)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            raise HTTPException(400, f"实时调参失败: {e}") from e

    @app.post("/api/experiments/{exp_id}/clone")
    def api_clone(exp_id: str, body: CloneBody) -> dict[str, Any]:
        _not_found(exp_id)
        try:
            return _mgr().clone(exp_id, name=body.name, with_weights=body.with_weights)
        except Exception as e:
            raise HTTPException(400, f"克隆失败: {e}") from e

    @app.delete("/api/experiments/{exp_id}")
    def api_delete(exp_id: str) -> Response:
        _not_found(exp_id)
        try:
            _mgr().delete(exp_id)
        except Exception as e:
            raise HTTPException(400, f"删除失败: {e}") from e
        return Response(status_code=204)

    @app.get("/api/experiments/{exp_id}/checkpoints")
    def api_ckpts(exp_id: str) -> list[dict[str, Any]]:
        _not_found(exp_id)
        return _mgr().store.checkpoint_infos(exp_id)

    @app.post("/api/experiments/{exp_id}/inspect")
    async def api_inspect(exp_id: str, body: InspectBody) -> dict[str, Any]:
        _not_found(exp_id)
        path = _mgr().store.ckpt_path(exp_id, body.checkpoint)
        if not path.is_file():
            raise HTTPException(400, f"检查点不存在: {body.checkpoint}")
        try:
            traj = await asyncio.to_thread(
                inspect_episode,
                path,
                body.board_size,
                body.seed,
                body.greedy,
                5000,
                True,
            )
            traj["experiment_id"] = exp_id
            meta = _mgr().store.read_meta(exp_id)
            traj["name"] = meta.get("name")
            return traj
        except Exception as e:
            raise HTTPException(400, f"分析失败: {e}") from e

    @app.post("/api/experiments/{exp_id}/{action}")
    def api_action(exp_id: str, action: str) -> dict[str, Any]:
        _not_found(exp_id)
        if action not in ("start", "pause", "resume", "stop"):
            raise HTTPException(404, f"未知操作: {action}")
        try:
            if action == "start":
                return _mgr().start(exp_id)
            if action == "pause":
                return _mgr().pause(exp_id)
            if action == "resume":
                return _mgr().resume(exp_id)
            return _mgr().stop(exp_id)
        except RuntimeError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            raise HTTPException(400, f"操作失败: {e}") from e

    @app.post("/api/compare")
    async def api_compare(body: CompareBody) -> dict[str, Any]:
        if len(body.entries) < 2 or len(body.entries) > 4:
            raise HTTPException(400, "请选择 2–4 个实验进行对比")
        pairs: list[tuple[str, str]] = []
        labels: list[tuple[str, str]] = []
        for ent in body.entries:
            if not _mgr().store.exists(ent.experiment_id):
                raise HTTPException(404, f"实验不存在: {ent.experiment_id}")
            path = _mgr().store.ckpt_path(ent.experiment_id, ent.checkpoint)
            if not path.is_file():
                raise HTTPException(400, f"检查点不存在: {ent.experiment_id}/{ent.checkpoint}")
            meta = _mgr().store.read_meta(ent.experiment_id)
            pairs.append((str(path), meta.get("name", ent.experiment_id)))
            labels.append((ent.experiment_id, meta.get("name", ent.experiment_id)))
        seed = body.seed if body.seed is not None else 0
        try:
            trajs = await asyncio.to_thread(compare_episodes, pairs, body.board_size, seed)
            for traj, (eid, name) in zip(trajs, labels, strict=True):
                traj["experiment_id"] = eid
                traj["name"] = name
            return {"seed": seed, "trajectories": trajs}
        except Exception as e:
            raise HTTPException(400, f"对比失败: {e}") from e

    @app.websocket("/ws/experiments/{exp_id}")
    async def ws_exp(ws: WebSocket, exp_id: str) -> None:
        await ws.accept()
        mgr_ = _mgr()
        if not mgr_.store.exists(exp_id):
            await ws.close(code=4404)
            return
        meta = mgr_.store.read_meta(exp_id)
        status = meta.get("status", "created")
        h = mgr_._workers.get(exp_id)
        if h is not None:
            status = h.status
        await ws.send_json({"type": "hello", "status": status})
        q = mgr_.subscribe(exp_id)
        try:
            while True:
                # Also detect client disconnect via receive
                getter = asyncio.create_task(q.get())
                recv = asyncio.create_task(ws.receive_text())
                done, pending = await asyncio.wait({getter, recv}, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                if recv in done:
                    # client message or disconnect
                    exc = recv.exception()
                    if exc is not None:
                        break
                    continue
                if getter in done:
                    msg = getter.result()
                    await ws.send_json(msg)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            mgr_.unsubscribe(exp_id, q)

    @app.websocket("/ws/experiments/{exp_id}/watch")
    async def ws_watch(ws: WebSocket, exp_id: str) -> None:
        await ws.accept()
        mgr_ = _mgr()
        if not mgr_.store.exists(exp_id):
            await ws.close(code=4404)
            return

        async def send_json(obj: dict[str, Any]) -> None:
            await ws.send_json(obj)

        session = WatchSession(mgr_, exp_id, send_json=send_json)
        task = asyncio.create_task(session.run())
        try:
            while True:
                raw = await ws.receive_json()
                if isinstance(raw, dict) and raw.get("type") == "config":
                    session.apply_config(raw)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            session.stop()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    # Static SPA
    if STATIC_DIR.is_dir():
        assets = STATIC_DIR / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/favicon.svg")
        def favicon():
            path = STATIC_DIR / "favicon.svg"
            if path.is_file():
                return FileResponse(path)
            raise HTTPException(404, "未找到")

        @app.get("/")
        def index():
            return FileResponse(STATIC_DIR / "index.html")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            if full_path.startswith("api/") or full_path.startswith("ws/"):
                raise HTTPException(404, "未找到")
            candidate = STATIC_DIR / full_path
            if candidate.is_file() and candidate.resolve().is_relative_to(STATIC_DIR.resolve()):
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")

    return app
