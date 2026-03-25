from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .agent import AgentHyperParams, DDQNAgent
from .config import resolve_device
from .env import SnakeEnv, SnakeEnvConfig
from .run_context import RunContext


def build_eval_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained DDQN snake model.", add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ignore-run-config",
        action="store_true",
        help="忽略 checkpoint 同目录下的 run_config，仅用下方 CLI 构造环境",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps-per-episode", type=int, default=3000)
    parser.add_argument("--difficulty", type=str, default="normal")
    parser.add_argument("--mode", type=str, default="classic", choices=["classic", "wrap"])
    parser.add_argument("--board-size", type=int, default=22)
    parser.add_argument("--enable-bonus-food", action="store_true")
    parser.add_argument("--enable-obstacles", action="store_true")
    parser.add_argument("--allow-leveling", action="store_true")
    parser.add_argument("--max-steps-without-food", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-sleep-ms", type=int, default=80)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_eval_arg_parser().parse_args(argv)


def hwc_to_chw(obs_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(obs_hwc, (2, 0, 1)).astype(np.float32, copy=False)


def center_pad_chw(obs_chw: np.ndarray, target_size: int) -> np.ndarray:
    channels, height, width = obs_chw.shape
    if height == target_size and width == target_size:
        return obs_chw
    if height > target_size or width > target_size:
        raise ValueError(f"观测尺寸 {obs_chw.shape} 大于目标尺寸 {target_size}")
    out = np.zeros((channels, target_size, target_size), dtype=np.float32)
    top = (target_size - height) // 2
    left = (target_size - width) // 2
    out[:, top : top + height, left : left + width] = obs_chw
    return out


def build_agent(checkpoint: Path, device: torch.device) -> DDQNAgent:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    observation_shape = tuple(int(v) for v in payload["observation_shape"])
    num_actions = int(payload["num_actions"])
    hp_data = payload.get("hyper_params", {})
    model_type = payload.get("model_type", "small_cnn")
    hp = AgentHyperParams(**hp_data) if hp_data else AgentHyperParams()
    agent = DDQNAgent(
        observation_shape=observation_shape,
        num_actions=num_actions,
        device=device,
        hp=hp,
        model_type=model_type,
    )
    agent.load_checkpoint(checkpoint)
    return agent


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(resolve_device(args.device))
    ctx = RunContext.from_checkpoint(args.checkpoint)
    if args.ignore_run_config or ctx.env is None:
        env = SnakeEnv(
            config=SnakeEnvConfig(
                difficulty=args.difficulty,
                mode=args.mode,
                board_size=args.board_size,
                enable_bonus_food=args.enable_bonus_food,
                enable_obstacles=args.enable_obstacles,
                allow_leveling=args.allow_leveling,
                max_steps_without_food=args.max_steps_without_food,
            ),
            seed=args.seed,
        )
    else:
        e = ctx.env
        env = SnakeEnv(
            config=SnakeEnvConfig(
                difficulty=e.difficulty,
                mode=e.mode,
                board_size=e.board_size,
                enable_bonus_food=e.enable_bonus_food,
                enable_obstacles=e.enable_obstacles,
                allow_leveling=e.allow_leveling,
                max_steps_without_food=e.max_steps_without_food,
            ),
            reward_weights=ctx.reward_weights,
            seed=args.seed,
        )
    agent = build_agent(args.checkpoint, device=device)
    checkpoint_size = int(agent.observation_shape[1])

    check_obs, _ = env.reset()
    check_shape = hwc_to_chw(check_obs).shape
    if agent.model_type == "small_cnn" and tuple(check_shape) != tuple(agent.observation_shape):
        raise ValueError(
            "Observation shape mismatch between environment and checkpoint: "
            f"env={check_shape}, ckpt={agent.observation_shape}"
        )

    rewards: list[float] = []
    steps_list: list[int] = []
    foods_list: list[int] = []
    scores: list[int] = []
    reason_counter: dict[str, int] = {}

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        if agent.model_type == "hybrid":
            state = hwc_to_chw(env.get_local_patch(checkpoint_size))
            global_feat = env.get_global_features()
        else:
            state = hwc_to_chw(obs)
            if agent.model_type == "adaptive_cnn":
                state = center_pad_chw(state, checkpoint_size)
            global_feat = None
        episode_reward = 0.0
        info: dict[str, Any] = {"terminal_reason": ""}

        for _ in range(args.max_steps_per_episode):
            action = agent.select_action(state, global_step=0, eval_mode=True, global_feat=global_feat)
            next_obs, reward, done, info = env.step(action)
            if agent.model_type == "hybrid":
                state = hwc_to_chw(env.get_local_patch(checkpoint_size))
                global_feat = env.get_global_features()
            else:
                state = hwc_to_chw(next_obs)
                if agent.model_type == "adaptive_cnn":
                    state = center_pad_chw(state, checkpoint_size)
            episode_reward += float(reward)
            if args.render:
                print(f"\nEpisode {episode}, reward={episode_reward:.3f}")
                print(env.render(mode="ansi"))
                time.sleep(max(0, args.render_sleep_ms) / 1000.0)
            if done:
                break

        stats = env.get_episode_stats()
        terminal_reason = str(info.get("terminal_reason", "")) or "running"
        reason_counter[terminal_reason] = reason_counter.get(terminal_reason, 0) + 1
        rewards.append(episode_reward)
        steps_list.append(int(stats["steps"]))
        foods_list.append(int(stats["foods"]))
        scores.append(int(stats["score_end"]))

        print(
            f"[Eval {episode:3d}] reward={episode_reward:8.3f} "
            f"steps={stats['steps']:4d} foods={stats['foods']:3d} "
            f"score={stats['score_end']:5d} terminal={terminal_reason}"
        )

    result = {
        "episodes": args.episodes,
        "model_type": agent.model_type,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "avg_foods": float(np.mean(foods_list)) if foods_list else 0.0,
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "terminal_reason_counter": reason_counter,
    }
    env.close()
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_eval(args)
    print("\nEvaluation summary:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
