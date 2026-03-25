"""
TensorBoard 训练监控：指向 ``runs/`` 目录，自动聚合各次 run 根目录下的 tfevents。

用法：
    uv run snake-rl monitor
    uv run snake-rl monitor --runs-dir runs --port 6006
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import socket
import time
from pathlib import Path

from tensorboard import program


def build_monitor_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start TensorBoard for snake training runs (logdir = runs/).",
        add_help=False,
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument(
        "--bind-all",
        action="store_true",
        help="等价于 --host 0.0.0.0（已默认开放时通常可省略）",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_monitor_arg_parser().parse_args(argv)


def get_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        sock.close()
    return ip


def run_web_monitor(args: Namespace) -> None:
    runs_dir = args.runs_dir.expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    host = "0.0.0.0" if args.bind_all else args.host

    tb = program.TensorBoard()
    tb.configure(
        argv=[
            "tensorboard",
            "--logdir",
            str(runs_dir),
            "--port",
            str(int(args.port)),
            "--host",
            host,
            "--reload_interval",
            "5",
        ]
    )
    url = tb.launch()
    lan = get_lan_ip()
    port = int(args.port)
    print("TensorBoard 已启动。")
    print(f"  运行目录 (--logdir): {runs_dir}")
    print(f"  本机: {url}")
    if host == "0.0.0.0":
        print(f"  局域网（示例）: http://{lan}:{port}/")
    print("按 Ctrl+C 停止。")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nTensorBoard 已停止。")


def main(argv: list[str] | None = None) -> None:
    run_web_monitor(parse_args(argv))


if __name__ == "__main__":
    main()
