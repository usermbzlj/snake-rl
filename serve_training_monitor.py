"""
开放 TensorBoard 端口，方便在局域网或通过内网穿透观察训练过程。

用法：
    uv run python serve_training_monitor.py
    uv run python serve_training_monitor.py --logdir runs --port 6006
    uv run python serve_training_monitor.py --public-url https://xxxxx.trycloudflare.com

说明：
    - 默认监听 0.0.0.0，因此同一局域网中的其他设备可以直接访问
    - 如果需要外网访问，可配合 frp / ngrok / cloudflared tunnel / Tailscale Funnel 等使用
"""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import time

from tensorboard import program


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve TensorBoard for remote monitoring.")
    parser.add_argument("--logdir", type=Path, default=Path("runs"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument(
        "--public-url",
        type=str,
        default="",
        help="如果你已用 frp/ngrok/cloudflared 映射到公网，可在这里填公网地址，脚本会一并打印。",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    args.logdir.mkdir(parents=True, exist_ok=True)

    tb = program.TensorBoard()
    tb.configure(
        argv=[
            None,
            "--logdir",
            str(args.logdir),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    )
    _ = tb.launch()

    lan_ip = get_lan_ip()
    print("TensorBoard 已启动。")
    print(f"日志目录: {args.logdir.resolve()}")
    print(f"本机访问: http://127.0.0.1:{args.port}")
    print(f"局域网访问: http://{lan_ip}:{args.port}")
    print()
    print("如果其他设备无法访问，请检查：")
    print("1. Windows 防火墙是否放行该端口")
    print("2. 路由器/交换机是否允许局域网互访")
    print("3. 训练机和观测设备是否在同一网段")
    print()
    print("如果需要外网访问，可将本地端口映射出去，例如：")
    print(f"- frp / ngrok / cloudflared tunnel / Tailscale Funnel → {args.port}")
    if args.public_url:
        print(f"公网访问: {args.public_url}")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nTensorBoard 已停止。")


if __name__ == "__main__":
    main()
