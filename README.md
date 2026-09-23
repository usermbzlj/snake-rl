# 贪吃蛇 AI 训练实验室

GPU 向量化贪吃蛇环境 + PPO/DQN + FastAPI/Vue 本地训练控制台。

## 快速开始

```powershell
uv sync
uv run snake-lab
# 或
uv run python -m snake_rl
```

浏览器打开 `http://127.0.0.1:7860`。局域网手机可访问启动时打印的 LAN 地址。

## 开发

```powershell
# 后端
uv run python -m pytest tests -m "not slow"
uv run python -m ruff check snake_rl tests
uv run python -m ruff format --check snake_rl tests
uv run python -m pyright snake_rl tests

# 前端
cd web
npm install
npm run build   # 输出到 snake_rl/server/static/
npm run test
```

## 布局

- `snake_rl/core/` — 环境、网络、PPO/DQN、配置、检查点
- `snake_rl/lab/` — 实验存储、训练 worker、管理器、观战、分析
- `snake_rl/server/` — FastAPI + 静态 SPA
- `web/` — Vite + Vue 3 前端源码
- `experiments/` — 运行时实验数据（gitignored）
