"""WebSocket endpoints for experiment metrics and live watch."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from snake_rl.lab.viewer import WatchSession
from snake_rl.server.schemas import get_manager_from_ws

log = logging.getLogger(__name__)

router = APIRouter(tags=["ws"])


@router.websocket("/ws/experiments/{exp_id}")
async def ws_exp(ws: WebSocket, exp_id: str) -> None:
    await ws.accept()
    mgr = get_manager_from_ws(ws)
    if not mgr.store.exists(exp_id):
        await ws.close(code=4404)
        return
    meta = mgr.store.read_meta(exp_id)
    status = meta.get("status", "created")
    h = mgr._workers.get(exp_id)
    if h is not None:
        status = h.status
    await ws.send_json({"type": "hello", "status": status})
    q = mgr.subscribe(exp_id)
    try:
        while True:
            getter = asyncio.create_task(q.get())
            recv = asyncio.create_task(ws.receive_text())
            done, pending = await asyncio.wait({getter, recv}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                with suppress(asyncio.CancelledError):
                    await t
            if recv in done:
                exc = recv.exception()
                if exc is not None:
                    break
                continue
            if getter in done:
                msg = getter.result()
                await ws.send_json(msg)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        log.exception("ws_exp disconnected with error for %s", exp_id)
    finally:
        mgr.unsubscribe(exp_id, q)


@router.websocket("/ws/experiments/{exp_id}/watch")
async def ws_watch(ws: WebSocket, exp_id: str) -> None:
    await ws.accept()
    mgr = get_manager_from_ws(ws)
    if not mgr.store.exists(exp_id):
        await ws.close(code=4404)
        return

    async def send_json(obj: dict[str, Any]) -> None:
        await ws.send_json(obj)

    session = WatchSession(mgr, exp_id, send_json=send_json)
    task = asyncio.create_task(session.run())
    try:
        while True:
            raw = await ws.receive_json()
            if isinstance(raw, dict) and raw.get("type") == "config":
                session.apply_config(raw)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        log.exception("ws_watch disconnected with error for %s", exp_id)
    finally:
        session.stop()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
