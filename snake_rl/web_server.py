"""贪吃蛇 AI Web 控制台入口。实现见 snake_rl.web_console。"""

from __future__ import annotations

from .web_console.app import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
