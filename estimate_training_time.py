"""
快速估算当前 train_config.py 配置的大致训练耗时。

特点：
    - 会读取当前 `get_config()` 返回的配置
    - 实际跑一小段“环境交互 + 动作选择 + 反向传播”来测 steps/s
    - 根据当前方案推导各阶段 / 各地图尺寸的训练量
    - 输出三档估算：保守 / 中等 / 偏长

注意：
    - 这是“估算脚本”，不是精确计时器
    - 真正耗时会受：
      - agent 学到的水平（强策略通常会活得更久）
      - CPU / GPU
      - batch_size / replay 容量
      - 地图尺寸分布
      影响
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from train_config import get_config
from snake_rl.train import (
    build_env_options,
    create_agent,
    create_replay,
    extract_model_inputs,
    get_agent_input_size,
    resolve_device,
    set_global_seed,
    validate_config,
)
from snake_rl.env import SnakeEnv, SnakeEnvConfig


@dataclass(slots=True)
class EstimateSlice:
    label: str
    board_size: int
    timeout: int
    episodes: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate training time for current config.")
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=80,
        help="每个地图尺寸实际 benchmark 的训练步数，越大越稳定但越慢。",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=96,
        help="每个地图尺寸预热回放池的步数。",
    )
    parser.add_argument(
        "--step-scales",
        type=str,
        default="0.6,1.2,2.0",
        help="按 timeout 乘这些倍数估算每局平均步数，例如 0.6,1.2,2.0。",
    )
    return parser.parse_args()


def parse_scales(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("step-scales 不能为空")
    return values


def normalize_weights(weights: list[float] | None, size: int) -> list[float]:
    if weights is None:
        return [1.0 / size] * size
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights 之和必须大于 0")
    return [float(weight) / total for weight in weights]


def build_estimate_slices(cfg) -> list[EstimateSlice]:
    slices: list[EstimateSlice] = []

    if cfg.curriculum is not None:
        for idx, stage in enumerate(cfg.curriculum.stages, start=1):
            if stage.board_sizes:
                probs = normalize_weights(stage.weights, len(stage.board_sizes))
                for board_size, prob in zip(stage.board_sizes, probs):
                    timeout = max(1, int(round(board_size * board_size * stage.max_steps_scale)))
                    slices.append(
                        EstimateSlice(
                            label=f"stage{idx}:{board_size}",
                            board_size=int(board_size),
                            timeout=timeout,
                            episodes=stage.episodes * prob,
                        )
                    )
            else:
                timeout = (
                    stage.board_size * stage.board_size
                    if cfg.curriculum.scale_timeout
                    else int(stage.max_steps_without_food)
                )
                slices.append(
                    EstimateSlice(
                        label=f"stage{idx}:{stage.board_size}",
                        board_size=int(stage.board_size),
                        timeout=int(timeout),
                        episodes=float(stage.episodes),
                    )
                )
        return slices

    if cfg.random_board is not None:
        probs = normalize_weights(cfg.random_board.weights, len(cfg.random_board.board_sizes))
        for board_size, prob in zip(cfg.random_board.board_sizes, probs):
            timeout = max(1, int(round(board_size * board_size * cfg.random_board.max_steps_scale)))
            slices.append(
                EstimateSlice(
                    label=f"random:{board_size}",
                    board_size=int(board_size),
                    timeout=timeout,
                    episodes=cfg.episodes * prob,
                )
            )
        return slices

    slices.append(
        EstimateSlice(
            label=f"fixed:{cfg.env.board_size}",
            board_size=int(cfg.env.board_size),
            timeout=int(cfg.env.max_steps_without_food),
            episodes=float(cfg.episodes),
        )
    )
    return slices


def benchmark_board(
    *,
    cfg,
    board_size: int,
    timeout: int,
    device: torch.device,
    benchmark_steps: int,
    warmup_steps: int,
) -> float:
    effective_batch_size = min(int(cfg.batch_size), max(1, int(warmup_steps)))

    env = SnakeEnv(
        config=SnakeEnvConfig(
            **build_env_options(
                cfg.env,
                board_size=board_size,
                max_steps_without_food=timeout,
            )
        ),
        seed=cfg.env.seed,
    )
    agent_input_size = get_agent_input_size(cfg)
    obs, _ = env.reset(seed=cfg.env.seed)
    state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)

    agent = create_agent(cfg, device, state.shape)
    replay = create_replay(
        cfg,
        device,
        state.shape,
        capacity=max(effective_batch_size * 4, warmup_steps * 2),
    )

    # 预热：填充足够经验，后续 benchmark 才能测到真实 update 开销
    for step_idx in range(warmup_steps):
        action = env.sample_action()
        next_obs, reward, done, _ = env.step(
            action,
            lightweight_info=cfg.lightweight_step_info,
        )
        next_state, next_global_feat = extract_model_inputs(env, next_obs, cfg, agent_input_size)
        replay.add(
            state,
            action,
            reward,
            next_state,
            done,
            global_feat=global_feat,
            next_global_feat=next_global_feat,
        )
        state = next_state
        global_feat = next_global_feat
        if done:
            obs, _ = env.reset(seed=None if cfg.env.seed is None else cfg.env.seed + step_idx + 1)
            state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)

    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if device.type == "cuda":
        assert start is not None and end is not None
        start.record()
    else:
        import time

        t0 = time.perf_counter()

    global_step = warmup_steps
    for step_idx in range(benchmark_steps):
        action = agent.select_action(
            state,
            global_step=global_step,
            eval_mode=False,
            global_feat=global_feat,
        )
        next_obs, reward, done, _ = env.step(
            action,
            lightweight_info=cfg.lightweight_step_info,
        )
        next_state, next_global_feat = extract_model_inputs(env, next_obs, cfg, agent_input_size)
        replay.add(
            state,
            action,
            reward,
            next_state,
            done,
            global_feat=global_feat,
            next_global_feat=next_global_feat,
        )
        global_step += 1

        agent.update(
            replay_buffer=replay,
            global_step=global_step,
            batch_size=effective_batch_size,
            min_replay_size=effective_batch_size,
            train_frequency=cfg.train_frequency,
            target_update_interval=cfg.target_update_interval,
        )

        state = next_state
        global_feat = next_global_feat
        if done:
            obs, _ = env.reset(seed=None if cfg.env.seed is None else cfg.env.seed + warmup_steps + step_idx + 1)
            state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)

    if device.type == "cuda":
        assert start is not None and end is not None
        end.record()
        torch.cuda.synchronize(device)
        elapsed_s = start.elapsed_time(end) / 1000.0
    else:
        elapsed_s = time.perf_counter() - t0

    env.close()
    return benchmark_steps / max(elapsed_s, 1e-6)


def seconds_to_text(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} 分钟"
    hours = minutes / 60.0
    return f"{hours:.1f} 小时"


def main() -> None:
    args = parse_args()
    scales = parse_scales(args.step_scales)

    cfg = get_config()
    validate_config(cfg)
    set_global_seed(cfg.env.seed)
    device = torch.device(resolve_device(cfg.device))
    slices = build_estimate_slices(cfg)

    print("=== 当前配置耗时估算 ===")
    print(f"模型: {cfg.model_type}")
    print(f"设备: {device}")
    print(f"benchmark_steps: {args.benchmark_steps}")
    print(f"warmup_steps: {args.warmup_steps}")
    print()

    rows: list[tuple[EstimateSlice, float]] = []
    for item in slices:
        sps = benchmark_board(
            cfg=cfg,
            board_size=item.board_size,
            timeout=item.timeout,
            device=device,
            benchmark_steps=args.benchmark_steps,
            warmup_steps=args.warmup_steps,
        )
        rows.append((item, sps))
        print(
            f"{item.label:>14} | board={item.board_size:>2} | timeout={item.timeout:>4} "
            f"| episodes≈{item.episodes:>7.1f} | speed≈{sps:>7.1f} steps/s"
        )

    print()
    print("说明：下面三档估算使用的是“平均每局步数 = timeout × scale”的近似。")
    print("由于吃到食物后 timeout 会重置，强策略的实际步数可能高于这里的中档估算。")
    print()

    labels = ["保守", "中等", "偏长"]
    for idx, scale in enumerate(scales):
        total_steps = 0.0
        total_seconds = 0.0
        for item, sps in rows:
            expected_steps = min(cfg.max_steps_per_episode, max(1, int(round(item.timeout * scale))))
            total_steps += item.episodes * expected_steps
            total_seconds += (item.episodes * expected_steps) / max(sps, 1e-6)

        label = labels[idx] if idx < len(labels) else f"scale={scale:.2f}"
        print(
            f"{label:>4} | 假设平均步数≈timeout×{scale:.2f} "
            f"| 总步数≈{int(total_steps):,} | 预计耗时≈{seconds_to_text(total_seconds)}"
        )


if __name__ == "__main__":
    main()
