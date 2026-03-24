from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .agent import AgentHyperParams, DDQNAgent
from .config import EnvPreset, TrainConfig, resolve_device
from .env import SnakeEnv, SnakeEnvConfig, TERMINAL_REASONS
from .replay_buffer import ReplayBuffer
from .viz import LivePlotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Double DQN snake agent (PyTorch).")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps-per-episode", type=int, default=2500)
    parser.add_argument("--difficulty", type=str, default="normal")
    parser.add_argument("--mode", type=str, default="classic", choices=["classic", "wrap"])
    parser.add_argument("--board-size", type=int, default=22)
    parser.add_argument("--enable-bonus-food", action="store_true")
    parser.add_argument("--enable-obstacles", action="store_true")
    parser.add_argument("--allow-leveling", action="store_true")
    parser.add_argument("--max-steps-without-food", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=20000)
    parser.add_argument("--min-replay-size", type=int, default=2000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100000)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--moving-avg-window", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--tensorboard-log-interval", type=int, default=5)
    parser.add_argument("--jsonl-flush-interval", type=int, default=20)
    parser.add_argument(
        "--model-type",
        type=str,
        default="small_cnn",
        choices=["small_cnn", "adaptive_cnn", "hybrid"],
    )
    parser.add_argument("--local-patch-size", type=int, default=11)
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-live-plot", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-jsonl", action="store_true")
    parser.add_argument("--full-step-info", action="store_true")
    return parser.parse_args()


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    env = EnvPreset(
        difficulty=args.difficulty,
        mode=args.mode,
        board_size=args.board_size,
        enable_bonus_food=args.enable_bonus_food,
        enable_obstacles=args.enable_obstacles,
        allow_leveling=args.allow_leveling,
        max_steps_without_food=args.max_steps_without_food,
        seed=args.seed,
    )
    cfg = TrainConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        train_frequency=args.train_frequency,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        moving_avg_window=args.moving_avg_window,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        tensorboard_log_interval=args.tensorboard_log_interval,
        jsonl_flush_interval=args.jsonl_flush_interval,
        model_type=args.model_type,
        local_patch_size=args.local_patch_size,
        run_name=args.run_name,
        output_root=args.output_root,
        device=args.device,
        live_plot=not args.no_live_plot,
        tensorboard=not args.no_tensorboard,
        save_csv=not args.no_csv,
        save_jsonl=not args.no_jsonl,
        lightweight_step_info=not args.full_step_info,
        env=env,
    )
    return cfg


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hwc_to_chw(obs_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(obs_hwc, (2, 0, 1)).astype(np.float32, copy=False)


def center_pad_chw(obs_chw: np.ndarray, target_size: int) -> np.ndarray:
    """把较小地图居中 padding 到固定尺寸，供可变尺寸训练共用 replay buffer。"""
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


def prepare_run_dir(cfg: TrainConfig) -> Path:
    run_name = cfg.run_name
    if run_name == "default":
        run_name = datetime.now().strftime("ddqn_%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    with (run_dir / "train_config.json").open("w", encoding="utf-8") as f:
        payload = asdict(cfg)
        payload["output_root"] = str(cfg.output_root)
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    return run_dir


def build_env_options(
    env: EnvPreset,
    *,
    board_size: int | None = None,
    max_steps_without_food: int | None = None,
) -> dict[str, Any]:
    return {
        "difficulty": env.difficulty,
        "mode": env.mode,
        "board_size": env.board_size if board_size is None else int(board_size),
        "enable_bonus_food": env.enable_bonus_food,
        "enable_obstacles": env.enable_obstacles,
        "allow_leveling": env.allow_leveling,
        "max_steps_without_food": (
            env.max_steps_without_food
            if max_steps_without_food is None
            else int(max_steps_without_food)
        ),
    }


def build_initial_env(cfg: TrainConfig) -> SnakeEnv:
    if cfg.curriculum is not None:
        first_stage = cfg.curriculum.stages[0]
        if first_stage.board_sizes:
            first_board = max(int(size) for size in first_stage.board_sizes)
            timeout = max(
                1,
                int(round(first_board * first_board * first_stage.max_steps_scale)),
            )
        else:
            first_board = first_stage.board_size
            timeout = first_stage.max_steps_without_food
            if cfg.curriculum.scale_timeout:
                timeout = first_board * first_board
        options = build_env_options(cfg.env, board_size=first_board, max_steps_without_food=timeout)
    elif cfg.random_board is not None:
        first_board = cfg.random_board.board_sizes[0]
        timeout = max(1, int(round(first_board * first_board * cfg.random_board.max_steps_scale)))
        options = build_env_options(cfg.env, board_size=first_board, max_steps_without_food=timeout)
    else:
        options = build_env_options(cfg.env)

    env = SnakeEnv(config=SnakeEnvConfig(**options), seed=cfg.env.seed, reward_weights=cfg.reward_weights)
    return env


def get_agent_input_size(cfg: TrainConfig) -> int:
    if cfg.model_type == "hybrid":
        return int(cfg.local_patch_size)
    if cfg.curriculum is not None:
        sizes: list[int] = []
        for stage in cfg.curriculum.stages:
            if stage.board_sizes:
                sizes.extend(int(size) for size in stage.board_sizes)
            else:
                sizes.append(int(stage.board_size))
        return max(sizes)
    if cfg.random_board is not None:
        return max(cfg.random_board.board_sizes)
    return int(cfg.env.board_size)


def extract_model_inputs(
    env: SnakeEnv,
    obs_hwc: np.ndarray,
    cfg: TrainConfig,
    agent_input_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if cfg.model_type == "hybrid":
        patch = env.get_local_patch(cfg.local_patch_size)
        return hwc_to_chw(patch), env.get_global_features()

    state = hwc_to_chw(obs_hwc)
    if cfg.model_type == "adaptive_cnn" and (cfg.curriculum is not None or cfg.random_board is not None):
        state = center_pad_chw(state, agent_input_size)
    return state, None


def create_agent(cfg: TrainConfig, device: torch.device, observation_shape: tuple[int, int, int]) -> DDQNAgent:
    return DDQNAgent(
        observation_shape=observation_shape,
        num_actions=3,
        device=device,
        hp=AgentHyperParams(
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            grad_clip_norm=cfg.grad_clip_norm,
            epsilon_start=cfg.epsilon_start,
            epsilon_end=cfg.epsilon_end,
            epsilon_decay_steps=cfg.epsilon_decay_steps,
        ),
        model_type=cfg.model_type,
    )


def create_replay(cfg: TrainConfig, device: torch.device, observation_shape: tuple[int, int, int], capacity: int | None = None) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=cfg.replay_capacity if capacity is None else int(capacity),
        observation_shape=observation_shape,
        device=device,
        hybrid=cfg.model_type == "hybrid",
    )


def validate_config(cfg: TrainConfig) -> None:
    if cfg.curriculum is not None and cfg.random_board is not None:
        raise ValueError("不能同时启用 curriculum 和 random_board，请二选一。")
    if cfg.model_type == "small_cnn" and (cfg.curriculum is not None or cfg.random_board is not None):
        raise ValueError("small_cnn 使用 Flatten+FC，无法支持可变尺寸，请改用 adaptive_cnn 或 hybrid。")
    if cfg.model_type == "hybrid" and (cfg.local_patch_size <= 0 or cfg.local_patch_size % 2 == 0):
        raise ValueError("hybrid 模型的 local_patch_size 必须是正奇数。")
    if cfg.curriculum is not None and not cfg.curriculum.stages:
        raise ValueError("curriculum.stages 不能为空。")
    if cfg.curriculum is not None:
        for idx, stage in enumerate(cfg.curriculum.stages, start=1):
            if stage.board_sizes:
                if len(stage.board_sizes) == 0:
                    raise ValueError(f"curriculum stage {idx} 的 board_sizes 不能为空。")
                if stage.weights is not None and len(stage.weights) != len(stage.board_sizes):
                    raise ValueError(f"curriculum stage {idx} 的 weights 长度必须等于 board_sizes。")
            elif int(stage.board_size) <= 0:
                raise ValueError(f"curriculum stage {idx} 的 board_size 必须大于 0。")
    if cfg.random_board is not None and not cfg.random_board.board_sizes:
        raise ValueError("random_board.board_sizes 不能为空。")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_random_board(cfg: TrainConfig) -> tuple[int, int]:
    if cfg.random_board is None:
        raise ValueError("random_board 配置不存在")
    board_size = random.choices(
        cfg.random_board.board_sizes,
        weights=cfg.random_board.weights,
        k=1,
    )[0]
    timeout = max(1, int(round(board_size * board_size * cfg.random_board.max_steps_scale)))
    return int(board_size), timeout


def sample_curriculum_stage_board(cfg: TrainConfig, stage: Any) -> tuple[int, int]:
    if stage.board_sizes:
        board_size = random.choices(stage.board_sizes, weights=stage.weights, k=1)[0]
        timeout = max(1, int(round(board_size * board_size * stage.max_steps_scale)))
        return int(board_size), timeout

    board_size = int(stage.board_size)
    timeout = board_size * board_size if cfg.curriculum and cfg.curriculum.scale_timeout else int(stage.max_steps_without_food)
    return board_size, timeout


def curriculum_stage_label(stage: Any) -> str:
    if stage.board_sizes:
        sizes = ", ".join(str(int(size)) for size in stage.board_sizes)
        return f"random[{sizes}]"
    return str(int(stage.board_size))


def maybe_write_episode(
    *,
    writer: SummaryWriter | None,
    plotter: LivePlotter,
    jsonl_file: Any,
    row: dict[str, Any],
    terminal_reason_counter: dict[str, int],
    tensorboard_log_interval: int,
    jsonl_flush_interval: int,
) -> None:
    episode = int(row["episode"])
    should_log_tensorboard = episode == 1 or episode % max(1, int(tensorboard_log_interval)) == 0
    should_flush_jsonl = episode % max(1, int(jsonl_flush_interval)) == 0

    if jsonl_file is not None:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if should_flush_jsonl:
            jsonl_file.flush()

    if writer is not None and should_log_tensorboard:
        writer.add_scalar("episode/reward", row["reward"], episode)
        writer.add_scalar("episode/avg_reward", row["avg_reward"], episode)
        writer.add_scalar("episode/steps", row["steps"], episode)
        writer.add_scalar("episode/avg_steps", row["avg_steps"], episode)
        writer.add_scalar("episode/foods", row["foods"], episode)
        writer.add_scalar("episode/score", row["score"], episode)
        writer.add_scalar("episode/win", row["win"], episode)
        writer.add_scalar("episode/board_size", row["board_size"], episode)
        writer.add_scalar("train/epsilon", row["epsilon"], episode)
        if row["loss"] is not None:
            writer.add_scalar("train/loss", row["loss"], episode)
        if row["q_mean"] is not None:
            writer.add_scalar("train/q_mean", row["q_mean"], episode)
        if row["target_q_mean"] is not None:
            writer.add_scalar("train/target_q_mean", row["target_q_mean"], episode)
        writer.add_scalar(
            f"terminal_reason/{row['terminal_reason']}",
            terminal_reason_counter[row["terminal_reason"]],
            episode,
        )
        if row["stage_index"] is not None:
            writer.add_scalar("curriculum/stage_index", row["stage_index"], episode)

    plotter.update(
        episode=int(row["episode"]),
        reward=float(row["reward"]),
        steps=int(row["steps"]),
        foods=int(row["foods"]),
        epsilon=float(row["epsilon"]),
        loss=row["loss"],
    )


def finalize_run(
    *,
    cfg: TrainConfig,
    run_dir: Path,
    episode_rows: list[dict[str, Any]],
    writer: SummaryWriter | None,
    plotter: LivePlotter,
    jsonl_file: Any,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if cfg.save_csv:
        write_csv(run_dir / "logs" / "episodes.csv", episode_rows)
    with (run_dir / "logs" / "summary.json").open("w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    if jsonl_file is not None:
        jsonl_file.close()
    if writer is not None:
        writer.close()
    plotter.close()
    return summary


def run_standard_training(cfg: TrainConfig, resume_path: Path | None = None) -> dict[str, Any]:
    device = torch.device(resolve_device(cfg.device))
    set_global_seed(cfg.env.seed)

    env = build_initial_env(cfg)
    obs, _ = env.reset(seed=cfg.env.seed)
    agent_input_size = get_agent_input_size(cfg)
    state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)
    if global_feat is not None:
        global_feat = global_feat.astype(np.float32, copy=False)

    agent = create_agent(cfg, device, state.shape)
    replay = create_replay(cfg, device, state.shape)

    run_dir = prepare_run_dir(cfg)
    writer = SummaryWriter(log_dir=str(run_dir / "logs" / "tensorboard")) if cfg.tensorboard else None
    plotter = LivePlotter(enabled=cfg.live_plot)
    jsonl_file = (run_dir / "logs" / "episodes.jsonl").open("a", encoding="utf-8") if cfg.save_jsonl else None

    episode_rows: list[dict[str, Any]] = []
    reward_window: list[float] = []
    steps_window: list[float] = []
    terminal_reason_counter: dict[str, int] = {}
    best_avg_reward = float("-inf")
    global_step = 0
    start_episode = 1

    if resume_path is not None:
        extra = agent.load_checkpoint(resume_path)
        global_step = int(extra.get("global_step", 0))
        start_episode = int(extra.get("episode", 0)) + 1
        best_avg_reward = float(extra.get("best_avg_reward", best_avg_reward))

    for episode in range(start_episode, cfg.episodes + 1):
        if cfg.random_board is not None:
            board_size, timeout = sample_random_board(cfg)
            obs, _ = env.reset(
                seed=None if cfg.env.seed is None else cfg.env.seed + episode,
                options=build_env_options(
                    cfg.env,
                    board_size=board_size,
                    max_steps_without_food=timeout,
                ),
            )
        else:
            board_size = cfg.env.board_size
            obs, _ = env.reset(seed=None if cfg.env.seed is None else cfg.env.seed + episode)

        state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)
        episode_reward = 0.0
        last_loss = None
        last_q_mean = None
        last_target_q_mean = None
        info: dict[str, Any] = {"terminal_reason": ""}

        for _ in range(cfg.max_steps_per_episode):
            action = agent.select_action(
                state,
                global_step=global_step,
                eval_mode=False,
                global_feat=global_feat,
            )
            next_obs, reward, done, info = env.step(
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
            episode_reward += float(reward)

            metrics = agent.update(
                replay_buffer=replay,
                global_step=global_step,
                batch_size=cfg.batch_size,
                min_replay_size=cfg.min_replay_size,
                train_frequency=cfg.train_frequency,
                target_update_interval=cfg.target_update_interval,
            )
            if metrics is not None:
                last_loss = metrics["loss"]
                last_q_mean = metrics["q_mean"]
                last_target_q_mean = metrics["target_q_mean"]

            state = next_state
            global_feat = next_global_feat
            if done:
                break

        stats = env.get_episode_stats()
        terminal_reason = str(info.get("terminal_reason", "")) or "running"
        terminal_reason_counter[terminal_reason] = terminal_reason_counter.get(terminal_reason, 0) + 1

        reward_window.append(episode_reward)
        if len(reward_window) > cfg.moving_avg_window:
            reward_window.pop(0)
        avg_reward = float(sum(reward_window) / len(reward_window))

        steps_window.append(float(stats["steps"]))
        if len(steps_window) > cfg.moving_avg_window:
            steps_window.pop(0)
        avg_steps = float(sum(steps_window) / len(steps_window))

        epsilon = agent.epsilon_by_step(global_step)
        win_flag = 1 if terminal_reason == TERMINAL_REASONS["BOARD_FULL"] else 0
        row = {
            "episode": episode,
            "global_step": global_step,
            "reward": episode_reward,
            "avg_reward": avg_reward,
            "steps": stats["steps"],
            "avg_steps": avg_steps,
            "foods": stats["foods"],
            "score": stats["score_end"],
            "epsilon": epsilon,
            "loss": last_loss,
            "q_mean": last_q_mean,
            "target_q_mean": last_target_q_mean,
            "terminal_reason": terminal_reason,
            "win": win_flag,
            "board_size": board_size,
            "stage_index": None,
        }
        episode_rows.append(row)
        maybe_write_episode(
            writer=writer,
            plotter=plotter,
            jsonl_file=jsonl_file,
            row=row,
            terminal_reason_counter=terminal_reason_counter,
            tensorboard_log_interval=cfg.tensorboard_log_interval,
            jsonl_flush_interval=cfg.jsonl_flush_interval,
        )

        if episode % cfg.log_interval == 0 or episode == start_episode:
            print(
                f"[Episode {episode:5d}] "
                f"board={board_size:2d} reward={episode_reward:8.3f} avg_reward={avg_reward:8.3f} "
                f"steps={stats['steps']:4d} foods={stats['foods']:3d} "
                f"score={stats['score_end']:5d} eps={epsilon:6.3f} "
                f"terminal={terminal_reason}"
            )

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_path = run_dir / "checkpoints" / "best.pt"
            agent.save_checkpoint(
                best_path,
                extra={
                    "episode": episode,
                    "global_step": global_step,
                    "best_avg_reward": best_avg_reward,
                    "run_dir": str(run_dir),
                },
            )

        if episode % cfg.checkpoint_interval == 0 or episode == cfg.episodes:
            latest_path = run_dir / "checkpoints" / "latest.pt"
            step_path = run_dir / "checkpoints" / f"ep_{episode:05d}.pt"
            extra = {
                "episode": episode,
                "global_step": global_step,
                "best_avg_reward": best_avg_reward,
                "run_dir": str(run_dir),
            }
            agent.save_checkpoint(latest_path, extra=extra)
            agent.save_checkpoint(step_path, extra=extra)

    env.close()
    summary = {
        "run_dir": str(run_dir),
        "mode": "random_board" if cfg.random_board is not None else "standard",
        "episodes": len(episode_rows),
        "best_avg_reward": best_avg_reward,
        "final_global_step": global_step,
        "last_episode": episode_rows[-1] if episode_rows else {},
        "model_type": cfg.model_type,
    }
    return finalize_run(
        cfg=cfg,
        run_dir=run_dir,
        episode_rows=episode_rows,
        writer=writer,
        plotter=plotter,
        jsonl_file=jsonl_file,
        summary=summary,
    )


def run_curriculum_training(cfg: TrainConfig) -> dict[str, Any]:
    device = torch.device(resolve_device(cfg.device))
    set_global_seed(cfg.env.seed)

    if cfg.curriculum is None:
        raise ValueError("curriculum 配置不存在")

    env = build_initial_env(cfg)
    agent_input_size = get_agent_input_size(cfg)
    obs, _ = env.reset(seed=cfg.env.seed)
    state, _ = extract_model_inputs(env, obs, cfg, agent_input_size)
    agent = create_agent(cfg, device, state.shape)

    run_dir = prepare_run_dir(cfg)
    writer = SummaryWriter(log_dir=str(run_dir / "logs" / "tensorboard")) if cfg.tensorboard else None
    plotter = LivePlotter(enabled=cfg.live_plot)
    jsonl_file = (run_dir / "logs" / "episodes.jsonl").open("a", encoding="utf-8") if cfg.save_jsonl else None

    episode_rows: list[dict[str, Any]] = []
    reward_window: list[float] = []
    steps_window: list[float] = []
    terminal_reason_counter: dict[str, int] = {}
    best_avg_reward = float("-inf")
    global_step = 0
    absolute_episode = 0
    stage_summaries: list[dict[str, Any]] = []

    replay = create_replay(cfg, device, state.shape, capacity=cfg.curriculum.stages[0].replay_capacity)

    for stage_index, stage in enumerate(cfg.curriculum.stages, start=1):
        if stage_index > 1:
            if cfg.curriculum.carry_replay:
                if replay.capacity != int(stage.replay_capacity):
                    old_size = len(replay)
                    replay = replay.resized_copy(stage.replay_capacity)
                    print(
                        f"[Curriculum] 迁移回放池: {old_size} 条经验, "
                        f"容量 {replay.capacity}"
                    )
            else:
                replay = create_replay(cfg, device, state.shape, capacity=stage.replay_capacity)

        stage_label = curriculum_stage_label(stage)
        stage_step = 0
        agent.reset_epsilon(
            epsilon_start=stage.epsilon_start,
            epsilon_end=stage.epsilon_end,
            epsilon_decay_steps=stage.epsilon_decay_steps,
        )

        print(
            f"\n=== Curriculum Stage {stage_index}/{len(cfg.curriculum.stages)} | "
            f"board={stage_label} | episodes={stage.episodes} ==="
        )

        stage_start_episode = absolute_episode + 1
        stage_end_episode = absolute_episode
        stage_foods_window: list[int] = []

        for stage_episode in range(1, stage.episodes + 1):
            absolute_episode += 1
            stage_end_episode = absolute_episode
            board_size, timeout = sample_curriculum_stage_board(cfg, stage)

            obs, _ = env.reset(
                seed=None if cfg.env.seed is None else cfg.env.seed + absolute_episode,
                options=build_env_options(
                    cfg.env,
                    board_size=board_size,
                    max_steps_without_food=timeout,
                ),
            )
            state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)
            episode_reward = 0.0
            last_loss = None
            last_q_mean = None
            last_target_q_mean = None
            info: dict[str, Any] = {"terminal_reason": ""}

            for _ in range(cfg.max_steps_per_episode):
                action = agent.select_action(
                    state,
                    global_step=stage_step,
                    eval_mode=False,
                    global_feat=global_feat,
                )
                next_obs, reward, done, info = env.step(
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
                stage_step += 1
                episode_reward += float(reward)

                metrics = agent.update(
                    replay_buffer=replay,
                    global_step=global_step,
                    batch_size=cfg.batch_size,
                    min_replay_size=stage.min_replay_size,
                    train_frequency=cfg.train_frequency,
                    target_update_interval=cfg.target_update_interval,
                )
                if metrics is not None:
                    last_loss = metrics["loss"]
                    last_q_mean = metrics["q_mean"]
                    last_target_q_mean = metrics["target_q_mean"]

                state = next_state
                global_feat = next_global_feat
                if done:
                    break

            stats = env.get_episode_stats()
            terminal_reason = str(info.get("terminal_reason", "")) or "running"
            terminal_reason_counter[terminal_reason] = terminal_reason_counter.get(terminal_reason, 0) + 1

            reward_window.append(episode_reward)
            if len(reward_window) > cfg.moving_avg_window:
                reward_window.pop(0)
            avg_reward = float(sum(reward_window) / len(reward_window))

            steps_window.append(float(stats["steps"]))
            if len(steps_window) > cfg.moving_avg_window:
                steps_window.pop(0)
            avg_steps = float(sum(steps_window) / len(steps_window))

            epsilon = agent.epsilon_by_step(stage_step)
            win_flag = 1 if terminal_reason == TERMINAL_REASONS["BOARD_FULL"] else 0
            row = {
                "episode": absolute_episode,
                "global_step": global_step,
                "reward": episode_reward,
                "avg_reward": avg_reward,
                "steps": stats["steps"],
                "avg_steps": avg_steps,
                "foods": stats["foods"],
                "score": stats["score_end"],
                "epsilon": epsilon,
                "loss": last_loss,
                "q_mean": last_q_mean,
                "target_q_mean": last_target_q_mean,
                "terminal_reason": terminal_reason,
                "win": win_flag,
                "board_size": board_size,
                "stage_index": stage_index,
            }
            episode_rows.append(row)
            maybe_write_episode(
                writer=writer,
                plotter=plotter,
                jsonl_file=jsonl_file,
                row=row,
                terminal_reason_counter=terminal_reason_counter,
                tensorboard_log_interval=cfg.tensorboard_log_interval,
                jsonl_flush_interval=cfg.jsonl_flush_interval,
            )

            if stage_episode % cfg.log_interval == 0 or stage_episode == 1:
                print(
                    f"[Stage {stage_index} | Ep {stage_episode:4d}/{stage.episodes}] "
                    f"board={board_size:2d} reward={episode_reward:8.3f} avg_reward={avg_reward:8.3f} "
                    f"steps={stats['steps']:4d} foods={stats['foods']:3d} "
                    f"score={stats['score_end']:5d} eps={epsilon:6.3f} terminal={terminal_reason}"
                )

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_path = run_dir / "checkpoints" / "best.pt"
                agent.save_checkpoint(
                    best_path,
                    extra={
                        "episode": absolute_episode,
                        "global_step": global_step,
                        "best_avg_reward": best_avg_reward,
                        "run_dir": str(run_dir),
                        "stage_index": stage_index,
                    },
                )

            if absolute_episode % cfg.checkpoint_interval == 0:
                latest_path = run_dir / "checkpoints" / "latest.pt"
                step_path = run_dir / "checkpoints" / f"ep_{absolute_episode:05d}.pt"
                extra = {
                    "episode": absolute_episode,
                    "global_step": global_step,
                    "best_avg_reward": best_avg_reward,
                    "run_dir": str(run_dir),
                    "stage_index": stage_index,
                }
                agent.save_checkpoint(latest_path, extra=extra)
                agent.save_checkpoint(step_path, extra=extra)

            stage_foods_window.append(int(stats["foods"]))
            if len(stage_foods_window) > stage.promotion_window:
                stage_foods_window.pop(0)
            if (
                stage.promotion_threshold_foods > 0
                and stage_episode >= stage.promotion_min_episodes
                and len(stage_foods_window) >= stage.promotion_window
            ):
                avg_foods = sum(stage_foods_window) / len(stage_foods_window)
                if avg_foods >= stage.promotion_threshold_foods:
                    print(
                        f"\n[Curriculum] 达到晋升条件！"
                        f"最近 {stage.promotion_window} 局平均食物: {avg_foods:.2f} "
                        f">= 门槛 {stage.promotion_threshold_foods:.1f} "
                        f"(阶段 {stage_index}, 第 {stage_episode}/{stage.episodes} 局)"
                    )
                    break

        stage_rows = [row for row in episode_rows if row["stage_index"] == stage_index]
        stage_summaries.append(
            {
                "stage_index": stage_index,
                "board_size": None if stage.board_sizes else stage.board_size,
                "stage_label": stage_label,
                "episodes": len(stage_rows),
                "episode_range": [stage_start_episode, stage_end_episode],
                "avg_reward_last": stage_rows[-1]["avg_reward"] if stage_rows else None,
            }
        )

    env.close()
    if episode_rows:
        latest_path = run_dir / "checkpoints" / "latest.pt"
        final_episode = episode_rows[-1]["episode"]
        agent.save_checkpoint(
            latest_path,
            extra={
                "episode": final_episode,
                "global_step": global_step,
                "best_avg_reward": best_avg_reward,
                "run_dir": str(run_dir),
                "stage_index": stage_summaries[-1]["stage_index"] if stage_summaries else None,
            },
        )

    summary = {
        "run_dir": str(run_dir),
        "mode": "curriculum",
        "episodes": len(episode_rows),
        "best_avg_reward": best_avg_reward,
        "final_global_step": global_step,
        "last_episode": episode_rows[-1] if episode_rows else {},
        "model_type": cfg.model_type,
        "stage_summaries": stage_summaries,
    }
    return finalize_run(
        cfg=cfg,
        run_dir=run_dir,
        episode_rows=episode_rows,
        writer=writer,
        plotter=plotter,
        jsonl_file=jsonl_file,
        summary=summary,
    )


def run_training(cfg: TrainConfig, resume_path: Path | None = None) -> dict[str, Any]:
    validate_config(cfg)
    if cfg.curriculum is not None:
        if resume_path is not None:
            raise NotImplementedError("curriculum 模式暂不支持 resume，请先用 latest.pt 手动 warm start。")
        return run_curriculum_training(cfg)
    return run_standard_training(cfg, resume_path=resume_path)


def main() -> None:
    args = parse_args()
    cfg = build_train_config(args)
    summary = run_training(cfg, resume_path=args.resume)
    print("Training finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
