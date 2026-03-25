"""
轻量 Web 训练监控后台（替代 TensorBoard）。

用法：
    uv run snake-rl monitor
    uv run snake-rl monitor --runs-dir runs --port 6006
    uv run snake-rl monitor --public-url https://xxxxx.trycloudflare.com
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import csv
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .run_meta import (
    build_run_meta,
    json_load_dict,
    list_run_metas_sorted,
    run_meta_to_api_item,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
DEFAULT_DASHBOARD = "dashboard.html"


def build_monitor_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve lightweight web monitor for snake training runs.",
        add_help=False,
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument(
        "--public-url",
        type=str,
        default="",
        help="如果你已用 frp/ngrok/cloudflared 映射到公网，可在这里填公网地址，脚本会一并打印。",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_monitor_arg_parser().parse_args(argv)


def get_lan_ip() -> str:
    """尽量推断本机在局域网中的 IP。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        sock.close()
    return ip


def _safe_run_name(value: str) -> str:
    run_name = value.strip()
    if not run_name or "/" in run_name or "\\" in run_name or run_name in {".", ".."}:
        raise ValueError("非法 run 名称")
    return run_name


def _read_last_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        block_size = 4096
        data = b""
        lines: list[bytes] = []
        while file_size > 0 and len(lines) <= max_lines:
            read_size = min(block_size, file_size)
            file_size -= read_size
            f.seek(file_size)
            data = f.read(read_size) + data
            lines = data.splitlines()
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    return [line.decode("utf-8", errors="replace") for line in tail]


def _episode_points_from_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw_lines = _read_last_lines(path, max_lines=max(limit * 2, 200))
    points: list[dict[str, Any]] = []
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            points.append(
                {
                    "episode": int(row.get("episode", 0)),
                    "reward": float(row.get("reward", 0.0)),
                    "avg_reward": float(row.get("avg_reward", 0.0)),
                    "steps": int(row.get("steps", 0)),
                    "foods": int(row.get("foods", 0)),
                    "score": int(row.get("score", 0)),
                    "epsilon": float(row.get("epsilon", 0.0)),
                    "stage_index": row.get("stage_index"),
                }
            )
        except Exception:
            continue
    points.sort(key=lambda x: int(x.get("episode", 0)))
    return points[-limit:]


def _episode_points_from_csv(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "episode": int(row.get("episode", "0")),
                        "reward": float(row.get("reward", "0")),
                        "avg_reward": float(row.get("avg_reward", "0")),
                        "steps": int(row.get("steps", "0")),
                        "foods": int(row.get("foods", "0")),
                        "score": int(row.get("score", "0")),
                        "epsilon": float(row.get("epsilon", "0")),
                        "stage_index": row.get("stage_index"),
                    }
                )
    except Exception:
        return []
    rows.sort(key=lambda x: int(x.get("episode", 0)))
    return rows[-limit:]


def _load_episode_points(run_dir: Path, limit: int) -> list[dict[str, Any]]:
    logs_dir = run_dir / "logs"
    points = _episode_points_from_jsonl(logs_dir / "episodes.jsonl", limit)
    if points:
        return points
    return _episode_points_from_csv(logs_dir / "episodes.csv", limit)


def _build_run_detail(run_dir: Path, points_limit: int) -> dict[str, Any]:
    item = run_meta_to_api_item(build_run_meta(run_dir))
    run_cfg = json_load_dict(run_dir / "run_config.json")
    if not run_cfg:
        run_cfg = json_load_dict(run_dir / "train_config.json")
    summary = json_load_dict(run_dir / "logs" / "summary.json")
    points = _load_episode_points(run_dir, points_limit)
    return {
        "run": item,
        "summary": summary,
        "config": run_cfg,
        "episodes": points,
        "latest": points[-1] if points else {},
    }


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "SnakeMonitorHTTP/1.0"

    @property
    def runs_dir(self) -> Path:
        return self.server.runs_dir  # type: ignore[attr-defined]

    @property
    def web_dir(self) -> Path:
        return self.server.web_dir  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "api_version": "1",
                        "runs_dir": str(self.runs_dir.resolve()),
                        "dashboard": "/dashboard",
                    },
                )
                return

            if path in {"/", ""}:
                self._redirect("/dashboard")
                return

            if path in {"/dashboard", "/dashboard.html"}:
                self._serve_static(DEFAULT_DASHBOARD, content_type="text/html; charset=utf-8")
                return

            if path == "/dashboard.js":
                self._serve_static("dashboard.js", content_type="application/javascript; charset=utf-8")
                return

            if path == "/dashboard.css":
                self._serve_static("dashboard.css", content_type="text/css; charset=utf-8")
                return

            if path == "/api/runs":
                runs = [run_meta_to_api_item(m) for m in list_run_metas_sorted(self.runs_dir)]
                self._json(
                    HTTPStatus.OK,
                    {
                        "runs": runs,
                        "count": len(runs),
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                return

            if path.startswith("/api/runs/"):
                parts = [p for p in path.split("/") if p]
                if len(parts) < 3:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                run_name = _safe_run_name(unquote(parts[2]))
                run_dir = (self.runs_dir / run_name).resolve()
                if not run_dir.exists() or not run_dir.is_dir():
                    self._json(HTTPStatus.NOT_FOUND, {"error": f"run 不存在: {run_name}"})
                    return

                limit_raw = query.get("limit", ["300"])[0]
                try:
                    limit = max(20, min(5000, int(limit_raw)))
                except Exception:
                    limit = 300

                if len(parts) == 3:
                    self._json(HTTPStatus.OK, _build_run_detail(run_dir, points_limit=limit))
                    return
                if len(parts) == 4 and parts[3] == "episodes":
                    self._json(
                        HTTPStatus.OK,
                        {"run": run_name, "episodes": _load_episode_points(run_dir, limit)},
                    )
                    return

                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return

            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _redirect(self, target: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", target)
        self.end_headers()

    def _serve_static(self, filename: str, content_type: str) -> None:
        path = (self.web_dir / filename).resolve()
        if not path.exists() or not path.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": f"静态文件不存在: {filename}"})
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_web_monitor(args: Namespace) -> None:
    runs_dir = args.runs_dir.expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    web_dir = WEB_DIR.resolve()
    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    server.runs_dir = runs_dir  # type: ignore[attr-defined]
    server.web_dir = web_dir  # type: ignore[attr-defined]

    lan_ip = get_lan_ip()
    print("Web 训练监控后台已启动。")
    print(f"运行目录: {runs_dir}")
    print(f"本机访问: http://127.0.0.1:{args.port}")
    print(f"本机面板: http://127.0.0.1:{args.port}/dashboard")
    print(f"局域网访问: http://{lan_ip}:{args.port}/dashboard")
    print()
    print("可用接口：")
    print("- GET /health")
    print("- GET /api/runs")
    print("- GET /api/runs/<run_name>")
    print("- GET /api/runs/<run_name>/episodes?limit=300")
    if args.public_url:
        print(f"公网访问: {args.public_url.rstrip('/')}/dashboard")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Web 训练监控后台已停止。")


def main(argv: list[str] | None = None) -> None:
    run_web_monitor(parse_args(argv))


if __name__ == "__main__":
    main()
