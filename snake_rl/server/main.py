"""Launcher: uvicorn on 0.0.0.0:7860, print LAN URLs, open browser."""

from __future__ import annotations

import argparse
import contextlib
import socket
import threading
import time
import webbrowser


def _lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = str(s.getsockname()[0])
        s.close()
        if not ip.startswith("127."):
            ips.append(ip)
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = str(info[4][0])
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    return ips


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="贪吃蛇 AI 训练实验室")
    parser.add_argument("--port", type=int, default=7860, help="HTTP 端口（默认 7860）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args(argv)

    import uvicorn

    from snake_rl.server.app import create_app

    app = create_app(port=args.port)

    local = f"http://127.0.0.1:{args.port}"
    print("贪吃蛇 AI 训练实验室")
    print(f"  本机: {local}")
    for ip in _lan_ips():
        print(f"  局域网: http://{ip}:{args.port}")
    print("按 Ctrl+C 停止（将关闭所有训练进程）")

    if not args.no_open:

        def _open() -> None:
            time.sleep(1.2)
            with contextlib.suppress(Exception):
                webbrowser.open(local)

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
