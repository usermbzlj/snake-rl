"""
估算当前训练方案的大致耗时：分离「环境+前向+写回放」与「反向更新」微基准，再按 train_frequency 合成，
比「每个地图尺寸各建一套 agent 全量预热」更快，且使用真实 batch_size（不再被 warmup 截断）。
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import dataclass
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .schemes import ACTIVE_SCHEME, get_config
from snake_rl.env import SnakeEnv, SnakeEnvConfig
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


@dataclass(slots=True)
class EstimateSlice:
    label: str
    board_size: int
    timeout: int
    episodes: float


@dataclass(slots=True)
class SliceBenchResult:
    item: EstimateSlice
    env_sps: float
    combined_sps: float
    env_frac: float


def build_estimate_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate training time for current config.", add_help=False)
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        help="覆盖训练方案 (custom/scheme1/2/3/4)，默认 custom，与 snake-rl train 一致。",
    )
    parser.add_argument(
        "--custom-config",
        type=Path,
        default=None,
        help=(
            "scheme=custom 时加载的 TrainConfig JSON 路径。"
            "若未指定，自动使用项目根目录的 custom_train_config.json"
        ),
    )
    parser.add_argument("--parallel", action="store_true", help="按并行配置进行估算")
    parser.add_argument("--parallel-workers", type=int, default=None, help="并行 worker 数")
    parser.add_argument("--parallel-sync-interval", type=int, default=None, help="策略同步间隔")
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=40,
        help="NN 更新微基准的步数；越大越稳但越慢。",
    )
    parser.add_argument(
        "--env-steps",
        type=int,
        default=20,
        help="每个地图尺寸的「环境+前向+写回放」微基准步数。",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：缩短微基准步数（精度换速度）。",
    )
    parser.add_argument(
        "--step-scales",
        type=str,
        default="0.6,1.2,2.0",
        help="按 timeout 乘这些倍数估算每局平均步数，例如 0.6,1.2,2.0。",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_estimate_arg_parser().parse_args(argv)


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


def _make_env(cfg, board_size: int, timeout: int) -> SnakeEnv:
    return SnakeEnv(
        config=SnakeEnvConfig(
            **build_env_options(
                cfg.env,
                board_size=board_size,
                max_steps_without_food=timeout,
            )
        ),
        seed=cfg.env.seed,
    )


def bench_env_forward_sps(
    *,
    cfg,
    device: torch.device,
    item: EstimateSlice,
    env_steps: int,
    warmup_steps: int,
) -> float:
    """每步：select_action + env.step + extract + replay.add（无 agent.update）。"""
    env = _make_env(cfg, item.board_size, item.timeout)
    agent_input_size = get_agent_input_size(cfg)
    obs, _ = env.reset(seed=cfg.env.seed)
    state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)
    agent = create_agent(cfg, device, state.shape)
    cap = max(int(cfg.batch_size) * 4, warmup_steps * 2)
    replay = create_replay(cfg, device, state.shape, capacity=cap)
    g = 0
    for step_idx in range(warmup_steps):
        action = agent.select_action(
            state,
            global_step=g,
            eval_mode=False,
            global_feat=global_feat,
        )
        next_obs, reward, done, _ = env.step(action, lightweight_info=cfg.lightweight_step_info)
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
        g += 1
        if done:
            obs, _ = env.reset(seed=None if cfg.env.seed is None else cfg.env.seed + step_idx + 1)
            state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for step_idx in range(env_steps):
        action = agent.select_action(
            state,
            global_step=g,
            eval_mode=False,
            global_feat=global_feat,
        )
        next_obs, reward, done, _ = env.step(action, lightweight_info=cfg.lightweight_step_info)
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
        g += 1
        if done:
            obs, _ = env.reset(seed=None if cfg.env.seed is None else cfg.env.seed + g + step_idx)
            state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - t0, 1e-9)
    env.close()
    return float(env_steps) / elapsed


def bench_update_sps(
    *,
    cfg,
    device: torch.device,
    board_size: int,
    timeout: int,
    benchmark_steps: int,
) -> float:
    """在代表尺寸上填满回放后，仅测 agent.update（train_frequency=1，避免 target 频繁同步）。"""
    env = _make_env(cfg, board_size, timeout)
    agent_input_size = get_agent_input_size(cfg)
    obs, _ = env.reset(seed=cfg.env.seed)
    state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)
    agent = create_agent(cfg, device, state.shape)
    bs = int(cfg.batch_size)
    warmup = max(bs + 8, int(cfg.min_replay_size))
    cap = max(warmup * 2, bs * 8)
    replay = create_replay(cfg, device, state.shape, capacity=cap)
    g = 0
    for step_idx in range(warmup):
        action = env.sample_action()
        next_obs, reward, done, _ = env.step(action, lightweight_info=cfg.lightweight_step_info)
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
        g += 1
        if done:
            obs, _ = env.reset(seed=None if cfg.env.seed is None else cfg.env.seed + step_idx + 1)
            state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)

    # 不计入时间的额外步，保证 update 路径被充分触发
    sync_iv = max(int(cfg.target_update_interval), 10**9)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(benchmark_steps):
        g += 1
        agent.update(
            replay_buffer=replay,
            global_step=g,
            batch_size=bs,
            min_replay_size=bs,
            train_frequency=1,
            target_update_interval=sync_iv,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - t0, 1e-9)
    env.close()
    return float(benchmark_steps) / elapsed


def seconds_to_text(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} 分钟"
    hours = minutes / 60.0
    return f"{hours:.1f} 小时"


def run_estimate(args: Namespace) -> None:
    scales = parse_scales(args.step_scales)

    effective_scheme = args.scheme or os.environ.get("SNAKE_TRAIN_SCHEME", ACTIVE_SCHEME)

    if args.custom_config and effective_scheme != "custom":
        print(
            f"警告：--custom-config 仅在 --scheme custom 时有效，"
            f"当前方案 {effective_scheme!r} 将忽略此参数。",
            file=sys.stderr,
        )

    os.environ["SNAKE_TRAIN_SCHEME"] = effective_scheme

    custom_path = args.custom_config if effective_scheme == "custom" else None
    try:
        cfg = get_config(scheme=effective_scheme, custom_config_path=custom_path)
    except FileNotFoundError as exc:
        print(f"错误：自定义配置文件不存在 —— {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"错误：自定义配置加载/解析失败 —— {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except Exception as exc:
        print(f"错误：自定义配置加载失败 —— {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if args.parallel:
        cfg.parallel.enabled = True
    if args.parallel_workers is not None:
        cfg.parallel.num_workers = max(1, int(args.parallel_workers))
    if args.parallel_sync_interval is not None:
        cfg.parallel.weight_sync_interval_steps = max(1, int(args.parallel_sync_interval))

    validate_config(cfg)
    set_global_seed(cfg.env.seed)
    device = torch.device(resolve_device(cfg.device))
    slices = build_estimate_slices(cfg)

    bench_steps = int(args.benchmark_steps)
    env_steps = int(args.env_steps)
    if args.quick:
        bench_steps = max(8, bench_steps // 2)
        env_steps = max(8, env_steps // 2)

    rep = max(slices, key=lambda s: s.board_size)
    warmup_steps = max(int(cfg.batch_size) + 8, 32)

    print("=== 当前配置耗时估算 ===", flush=True)
    print(f"模型: {cfg.model_type}", flush=True)
    print(f"设备: {device}", flush=True)
    print(
        "并行: "
        f"{'开启' if cfg.parallel.enabled else '关闭'}"
        + (f" (workers={cfg.parallel.num_workers})" if cfg.parallel.enabled else ""),
        flush=True,
    )
    print(
        f"微基准: env_steps={env_steps} / slice, nn_update_steps={bench_steps}, "
        f"代表尺寸={rep.board_size}×{rep.board_size}（NN 更新吞吐各 slice 复用）",
        flush=True,
    )
    print(flush=True)

    update_sps = bench_update_sps(
        cfg=cfg,
        device=device,
        board_size=rep.board_size,
        timeout=rep.timeout,
        benchmark_steps=bench_steps,
    )
    t_update = 1.0 / max(update_sps, 1e-9)
    t_update_per_env_step = t_update / max(int(cfg.train_frequency), 1)
    print(
        f"[NN 更新基准] board={rep.board_size} | batch={cfg.batch_size} | "
        f"{update_sps:.1f} updates/s（各 slice 复用）",
        flush=True,
    )
    print(flush=True)

    rows: list[tuple[EstimateSlice, float]] = []
    slice_results: list[SliceBenchResult] = []
    tf = max(int(cfg.train_frequency), 1)

    for item in slices:
        env_sps = bench_env_forward_sps(
            cfg=cfg,
            device=device,
            item=item,
            env_steps=env_steps,
            warmup_steps=warmup_steps,
        )
        t_env = 1.0 / max(env_sps, 1e-9)
        t_per_step = t_env + t_update_per_env_step
        combined_sps = 1.0 / max(t_per_step, 1e-9)
        env_frac = t_env / max(t_per_step, 1e-9)
        bottleneck = "compute-bound" if env_frac < 0.5 else "env-bound"
        slice_results.append(
            SliceBenchResult(
                item=item,
                env_sps=env_sps,
                combined_sps=combined_sps,
                env_frac=env_frac,
            )
        )
        rows.append((item, combined_sps))
        print(
            f"{item.label:>14} | board={item.board_size:>2} | timeout={item.timeout:>4} "
            f"| episodes≈{item.episodes:>7.1f} | env≈{env_sps:>7.1f}/s | "
            f"合成≈{combined_sps:>7.1f} steps/s | env {100 * env_frac:.0f}% | {bottleneck}",
            flush=True,
        )

    avg_env_frac = float(np.mean([r.env_frac for r in slice_results])) if slice_results else 0.5
    print(flush=True)
    print("性能分析（基于微基准近似）:", flush=True)
    if avg_env_frac > 0.55:
        print(
            "  环境与前向在单步中占比较高，可考虑启用「并行采样」把环境步放到多进程。",
            flush=True,
        )
    else:
        print(
            "  反向更新在单步中占比较高，可尝试减小 batch_size、或确认 GPU 已启用；"
            "并行采样对纯 learner 瓶颈帮助有限。",
            flush=True,
        )

    if cfg.parallel.enabled:
        scale = min(float(cfg.parallel.num_workers), 1.0 + 0.75 * max(0, cfg.parallel.num_workers - 1))
        rows = [(item, sps * scale) for item, sps in rows]
        print(flush=True)
        print(
            "[提示] 并行模式使用经验放大系数估算吞吐，"
            f"workers={cfg.parallel.num_workers} -> x{scale:.2f}，实际值请以实测为准。",
            flush=True,
        )

    print(flush=True)
    print("说明：下面三档估算使用「平均每局步数 ≈ timeout × scale」的近似。", flush=True)
    print("强策略存活更久时，实际步数可能高于中档估算。", flush=True)
    print(flush=True)

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
            f"| 总步数≈{int(total_steps):,} | 预计耗时≈{seconds_to_text(total_seconds)}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    run_estimate(parse_args(argv))


if __name__ == "__main__":
    main()
