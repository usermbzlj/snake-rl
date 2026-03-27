"""
启动本地模型推理服务，让浏览器里的正常游戏直接调用训练好的 .pt 模型。

典型用法：
    uv run snake-rl serve-model --port 8765
    uv run snake-rl serve-model --port 8765 --checkpoint runs/xxx/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np
import torch

from .config import resolve_device
from .env import SnakeEnv, SnakeEnvConfig
from .evaluate import build_agent, center_pad_chw, hwc_to_chw
from .run_context import checkpoint_run_dir, load_run_config_dict


def build_inference_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve local snake model inference over HTTP.",
        add_help=False,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_inference_arg_parser().parse_args(argv)


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


def browser_state_to_python_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    env_cfg = state.get("envConfig") or {}
    episode_stats = state.get("episodeStats") or {}
    bonus_food = state.get("bonusFood")

    py_snapshot: dict[str, Any] = {
        "config": {
            "difficulty": env_cfg.get("difficulty", "normal"),
            "mode": env_cfg.get("mode", "classic"),
            "board_size": int(env_cfg.get("boardSize", 22)),
            "enable_bonus_food": bool(env_cfg.get("enableBonusFood", True)),
            "enable_obstacles": bool(env_cfg.get("enableObstacles", True)),
            "allow_leveling": bool(env_cfg.get("allowLeveling", True)),
            "max_steps_without_food": int(env_cfg.get("maxStepsWithoutFood", 0)),
        },
        "reward_weights": state.get("rewardWeights", {}),
        "seed": state.get("seed"),
        "state": state.get("state", "running"),
        "direction": state.get("direction", "right"),
        "snake": state.get("snake", []),
        "food": state.get("food"),
        "bonus_food": None,
        "obstacles": state.get("obstacles", []),
        "score": int(state.get("score", 0)),
        "level": int(state.get("level", 1)),
        "foods_eaten": int(state.get("foodsEaten", 0)),
        "steps_since_last_food": int(state.get("stepsSinceLastFood", 0)),
        "episode_index": int(episode_stats.get("episode", 0)),
        "episode_steps": int(episode_stats.get("steps", 0)),
        "episode_total_reward": float(episode_stats.get("totalReward", 0.0)),
        "episode_foods": int(episode_stats.get("foods", 0)),
        "episode_bonus_foods": int(episode_stats.get("bonusFoods", 0)),
        "episode_level_ups": int(episode_stats.get("levelUps", 0)),
        "episode_max_length": int(episode_stats.get("maxLength", len(state.get("snake", [])))),
        "episode_terminal_reason": state.get("lastTerminalReason", ""),
    }
    if bonus_food:
        py_snapshot["bonus_food"] = {
            "x": int(bonus_food["x"]),
            "y": int(bonus_food["y"]),
            "expires_step": int(episode_stats.get("steps", 0)) + 1,
        }
    return py_snapshot


class ModelRunner:
    def __init__(self, device_name: str) -> None:
        self.device_name = device_name
        self.device = torch.device(resolve_device(device_name))
        self.checkpoint_path: Path | None = None
        self.agent = None
        self.env: SnakeEnv | None = None
        self.model_type = ""
        self.input_size = 0
        self.recommended_env_config: dict[str, Any] | None = None

    def load_checkpoint(self, checkpoint: str | Path) -> dict[str, Any]:
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")

        self.agent = build_agent(path, self.device)
        self.checkpoint_path = path
        self.model_type = self.agent.model_type
        obs_shape = tuple(self.agent.observation_shape)
        if self.model_type == "tiny":
            self.input_size = int(obs_shape[0]) if len(obs_shape) == 1 else int(obs_shape[-1])
        else:
            self.input_size = int(self.agent.observation_shape[1])
        self.env = SnakeEnv(config=SnakeEnvConfig(), seed=None)
        self.recommended_env_config = self._load_recommended_env(path)

        return {
            "checkpoint": str(path),
            "modelType": self.model_type,
            "inputSize": self.input_size,
            "supportsVariableBoard": self.model_type != "small_cnn",
            "recommendedEnvConfig": self.recommended_env_config,
        }

    def _load_recommended_env(self, checkpoint_path: Path) -> dict[str, Any] | None:
        run_dir = checkpoint_run_dir(checkpoint_path)
        if run_dir is None:
            return None
        payload = load_run_config_dict(run_dir)
        if payload is None:
            return None
        env = payload.get("env")
        return env if isinstance(env, dict) else None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.agent is not None,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else "",
            "modelType": self.model_type,
            "inputSize": self.input_size,
            "supportsVariableBoard": self.model_type not in ("small_cnn",) if self.agent is not None else False,
            "recommendedEnvConfig": self.recommended_env_config,
        }

    def act(self, browser_state: dict[str, Any], *, include_debug: bool = False) -> dict[str, Any]:
        if self.agent is None or self.env is None:
            raise RuntimeError("尚未加载模型，请先调用 /v1/load")

        py_snapshot = browser_state_to_python_snapshot(browser_state)
        self.env.set_state(py_snapshot)

        if self.model_type == "tiny":
            state = self.env.get_tiny_features()
            global_feat = None
        elif self.model_type == "hybrid":
            state = hwc_to_chw(self.env.get_local_patch(self.input_size))
            global_feat = self.env.get_global_features()
        else:
            obs = self.env.get_observation()
            state = hwc_to_chw(obs)
            if self.model_type == "adaptive_cnn":
                state = center_pad_chw(state, self.input_size)
            elif self.model_type == "small_cnn":
                if tuple(state.shape) != tuple(self.agent.observation_shape):
                    raise ValueError(
                        "固定尺寸模型与当前页面地图尺寸不匹配："
                        f" page={state.shape}, ckpt={self.agent.observation_shape}"
                    )
            global_feat = None

        start = time.perf_counter()
        action = int(
            self.agent.select_action(
                state,
                global_step=0,
                eval_mode=True,
                global_feat=global_feat,
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        out: dict[str, Any] = {
            "action": action,
            "modelType": self.model_type,
            "inferenceMs": round(elapsed_ms, 3),
        }
        if include_debug:
            q_raw = self.agent.compute_q_values(state, global_feat=global_feat)
            debug: dict[str, Any] = {
                "q_values": [float(x) for x in np.asarray(q_raw).flatten().tolist()],
            }
            if self.model_type == "tiny":
                debug["features"] = [float(x) for x in np.asarray(state).flatten().tolist()]
            elif self.model_type == "hybrid":
                debug["features"] = [float(x) for x in np.asarray(global_feat).flatten().tolist()]
                patch = np.asarray(state)
                debug["patch_shape"] = list(patch.shape)
                debug["patch_sample"] = [float(x) for x in patch.flatten()[:64].tolist()]
            out["debug"] = debug
        return out


class InferenceHandler(BaseHTTPRequestHandler):
    server_version = "SnakeInferenceHTTP/1.0"

    @property
    def runner(self) -> ModelRunner:
        return self.server.runner  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"ok": True, **self.runner.status()})
            return
        if self.path == "/v1/status":
            self._json(HTTPStatus.OK, self.runner.status())
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/v1/load":
                checkpoint = payload.get("checkpoint")
                if not checkpoint:
                    raise ValueError("checkpoint 不能为空")
                result = self.runner.load_checkpoint(checkpoint)
                self._json(HTTPStatus.OK, result)
                return

            if self.path == "/v1/act":
                state = payload.get("state")
                if not isinstance(state, dict):
                    raise ValueError("state 必须是对象")
                include_debug = bool(payload.get("include_debug"))
                result = self.runner.act(state, include_debug=include_debug)
                self._json(HTTPStatus.OK, result)
                return

            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except FileNotFoundError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_inference_http(args: Namespace) -> None:
    runner = ModelRunner(args.device)
    if args.checkpoint is not None:
        info = runner.load_checkpoint(args.checkpoint)
        print(f"已预加载模型: {info['checkpoint']}")

    server = ThreadingHTTPServer((args.host, args.port), InferenceHandler)
    server.runner = runner  # type: ignore[attr-defined]

    lan_ip = get_lan_ip()
    print("本地模型推理服务已启动。")
    print(f"本机访问: http://127.0.0.1:{args.port}")
    print(f"局域网访问: http://{lan_ip}:{args.port}")
    print("健康检查: /health")
    print("加载模型: POST /v1/load")
    print("请求动作: POST /v1/act")
    print()
    print("浏览器游戏页面可直接连接该服务，让训练好的模型接管正常游戏。")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("模型推理服务已停止。")


def main(argv: list[str] | None = None) -> None:
    serve_inference_http(parse_args(argv))


if __name__ == "__main__":
    main()
