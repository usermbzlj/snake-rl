"""FastAPI application: REST + WebSocket + SPA static files."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from snake_rl import __version__
from snake_rl.lab.manager import ExperimentManager
from snake_rl.lab.storage import ExperimentStore
from snake_rl.server.routes import experiments, meta, ws

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    *,
    manager: ExperimentManager | None = None,
    experiments_root: Path | str | None = None,
    port: int = 7860,
) -> FastAPI:
    store = ExperimentStore(experiments_root) if experiments_root is not None else ExperimentStore()
    mgr = manager or ExperimentManager(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        mgr.bind_loop(loop)
        mgr.startup()
        app.state.manager = mgr
        app.state.port = port
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

    app.include_router(meta.router)
    app.include_router(experiments.router)
    app.include_router(ws.router)

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
