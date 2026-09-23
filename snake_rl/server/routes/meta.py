"""Meta and schema endpoints."""

from __future__ import annotations

import logging
import socket
from typing import Any

import torch
from fastapi import APIRouter, Request

from snake_rl import __version__
from snake_rl.core.config import ui_schema

log = logging.getLogger(__name__)

router = APIRouter(tags=["meta"])


def lan_urls(port: int) -> list[str]:
    urls: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = str(info[4][0])
            if ip.startswith("127."):
                continue
            urls.append(f"http://{ip}:{port}")
    except Exception:
        log.debug("lan_urls: getaddrinfo failed", exc_info=True)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = str(s.getsockname()[0])
        s.close()
        u = f"http://{ip}:{port}"
        if u not in urls and not ip.startswith("127."):
            urls.insert(0, u)
    except Exception:
        log.debug("lan_urls: UDP probe failed", exc_info=True)
    return urls


def device_info() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    name = ""
    if cuda:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            log.debug("device_info: get_device_name failed", exc_info=True)
            name = "CUDA"
    else:
        name = "CPU"
    return {"cuda": cuda, "name": name}


@router.get("/api/meta")
def api_meta(request: Request) -> dict[str, Any]:
    port_ = int(getattr(request.app.state, "port", 7860))
    return {
        "version": __version__,
        "device": device_info(),
        "lan_urls": lan_urls(port_),
        "port": port_,
    }


@router.get("/api/config-schema")
def api_schema() -> dict[str, Any]:
    return ui_schema()
