from __future__ import annotations

import json
import socket
import time
from typing import Any


def tcp_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200)) == 200
    except Exception:
        return False


def http_get_json(url: str, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        import urllib.request

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def wait_for_port_closed(host: str, port: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + max(0.1, float(timeout))
    while time.time() < deadline:
        if not tcp_port_open(host, port):
            return True
        time.sleep(0.1)
    return not tcp_port_open(host, port)
