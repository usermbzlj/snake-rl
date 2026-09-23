"""Unified DDQN training loop: serial/parallel × standard/curriculum."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import queue
import random
from typing import Any

import numpy as np
import torch

from .agent import DDQNAgent
from .config import TrainConfig, resolve_device
from .env import TERMINAL_REASONS, SnakeEnv
from .obs import extract_model_inputs, uses_spatial_padding
from .parallel_rollout import (
    ActorPoolHandle,
    EpisodeDoneMessage,
    PolicySnapshot,
    TransitionMessage,
    WorkerEpisodeConfig,
    broadcast_policy,
    broadcast_runtime,
    make_policy_snapshot,
    start_actor_pool,
    stop_actor_pool,
)
from .replay_buffer import NStepAccumulator, PendingTransition, ReplayBuffer
from .run_context import checkpoint_run_dir
from .train_events import emit_event
from .train_factory import (
    build_env_options,
    build_initial_env,
    create_agent,
    create_replay,
    get_agent_input_size,
    set_global_seed,
)
from .train_io import (
    append_episode_csv_incremental,
    create_scalar_writer,
    finalize_run,
    load_episode_history_snapshot,
    maybe_write_episode,
    persist_training_state,
    prepare_run_dir,
)
from .training_state import load_training_state
from .viz import LivePlotter


def infer_last_global_step_from_warm_checkpoint(checkpoint: Path) -> int | None:
    path = checkpoint.expanduser().resolve()
    try:
        ckpt_obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt_obj, dict):
            extra = ckpt_obj.get("extra")
            if isinstance(extra, dict) and "global_step" in extra:
                return max(0, int(extra["global_step"]))
    except Exception:
        pass
    run_dir = checkpoint_run_dir(path)
    if run_dir is None:
        return None
    jsonl_path = run_dir / "logs" / "episodes.jsonl"
    if not jsonl_path.is_file():
        return None
    last_gs: int | None = None
    try:
        import json

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and "global_step" in row:
                        last_gs = max(0, int(row["global_step"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
    except OSError:
        return last_gs
    return last_gs


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
    timeout = (
        board_size * board_size
        if cfg.curriculum and cfg.curriculum.scale_timeout
        else int(stage.max_steps_without_food)
    )
    return board_size, timeout


def curriculum_stage_label(stage: Any) -> str:
    if stage.board_sizes:
        sizes = ", ".join(str(int(size)) for size in stage.board_sizes)
        return f"random[{sizes}]"
    return str(int(stage.board_size))


def _per_beta(cfg: TrainConfig, global_step: int) -> float:
    if cfg.epsilon_decay_steps <= 0:
        return float(cfg.per_beta_end)
    ratio = min(1.0, max(0, int(global_step)) / float(cfg.epsilon_decay_steps))
    return float(cfg.per_beta_start + ratio * (cfg.per_beta_end - cfg.per_beta_start))


def _push_nstep(
    acc: NStepAccumulator,
    replay: ReplayBuffer,
    *args: Any,
    **kwargs: Any,
) -> None:
    for item in acc.push(*args, **kwargs):
        _add_pending(replay, item)


def _add_pending(replay: ReplayBuffer, item: PendingTransition) -> None:
    replay.add(
        item.state,
        item.action,
        item.reward,
        item.next_state,
        item.done,
        global_feat=item.global_feat,
        next_global_feat=item.next_global_feat,
    )


def _build_parallel_runtime_for_standard(cfg: TrainConfig) -> WorkerEpisodeConfig:
    if cfg.random_board is not None:
        return WorkerEpisodeConfig(
            mode="random",
            max_steps_per_episode=int(cfg.max_steps_per_episode),
            stage_index=None,
            board_sizes=[int(size) for size in cfg.random_board.board_sizes],
            weights=cfg.random_board.weights,
            timeout_scale=float(cfg.random_board.max_steps_scale),
        )
    return WorkerEpisodeConfig(
        mode="fixed",
        max_steps_per_episode=int(cfg.max_steps_per_episode),
        stage_index=None,
        fixed_board_size=int(cfg.env.board_size),
        fixed_timeout=int(cfg.env.max_steps_without_food),
    )


def _build_parallel_runtime_for_stage(
    cfg: TrainConfig,
    stage: Any,
    *,
    stage_index: int,
) -> WorkerEpisodeConfig:
    if stage.board_sizes:
        return WorkerEpisodeConfig(
            mode="random",
            max_steps_per_episode=int(cfg.max_steps_per_episode),
            stage_index=int(stage_index),
            board_sizes=[int(size) for size in stage.board_sizes],
            weights=stage.weights,
            timeout_scale=float(stage.max_steps_scale),
        )
    timeout = (
        int(stage.board_size) * int(stage.board_size)
        if cfg.curriculum is not None and cfg.curriculum.scale_timeout
        else int(stage.max_steps_without_food)
    )
    return WorkerEpisodeConfig(
        mode="fixed",
        max_steps_per_episode=int(cfg.max_steps_per_episode),
        stage_index=int(stage_index),
        fixed_board_size=int(stage.board_size),
        fixed_timeout=int(timeout),
    )


def _build_actor_pool(
    *,
    cfg: TrainConfig,
    agent: DDQNAgent,
    state_shape: tuple[int, ...],
    agent_input_size: int,
    runtime_cfg: WorkerEpisodeConfig,
    worker_episode_counter_starts: list[int] | None = None,
) -> ActorPoolHandle:
    return start_actor_pool(
        parallel_cfg=cfg.parallel,
        env_cfg=cfg.env,
        reward_weights=cfg.reward_weights,
        hp=agent.hp,
        model_type=cfg.model_type,
        observation_shape=state_shape,
        local_patch_size=cfg.local_patch_size,
        agent_input_size=agent_input_size,
        use_padding=uses_spatial_padding(cfg),
        lightweight_step_info=cfg.lightweight_step_info,
        runtime_cfg=runtime_cfg,
        num_actions=agent.num_actions,
        worker_episode_counter_starts=worker_episode_counter_starts,
        dueling=cfg.dueling,
        noisy=cfg.noisy,
    )


def _broadcast_parallel_policy(
    *,
    actor_pool: ActorPoolHandle,
    agent: DDQNAgent,
    epsilon: float,
    policy_version: int,
) -> int:
    snapshot: PolicySnapshot = make_policy_snapshot(
        agent,
        epsilon=float(epsilon),
        version=int(policy_version),
    )
    broadcast_policy(actor_pool, snapshot)
    return int(policy_version) + 1


def run_greedy_eval(
    agent: DDQNAgent,
    cfg: TrainConfig,
    *,
    agent_input_size: int,
    seed: int | None,
    episodes: int,
) -> float:
    if episodes <= 0:
        return float("nan")
    env = build_initial_env(cfg)
    rewards: list[float] = []
    try:
        for idx in range(episodes):
            obs, _ = env.reset(seed=None if seed is None else int(seed) + 10_000 + idx)
            state, global_feat = extract_model_inputs(env, obs, cfg, agent_input_size)
            episode_reward = 0.0
            for _ in range(cfg.max_steps_per_episode):
                action = agent.select_action(state, global_step=0, eval_mode=True, global_feat=global_feat)
                next_obs, reward, done, _ = env.step(action, lightweight_info=True)
                state, global_feat = extract_model_inputs(env, next_obs, cfg, agent_input_size)
                episode_reward += float(reward)
                if done:
                    break
            rewards.append(episode_reward)
    finally:
        env.close()
    return float(np.mean(rewards)) if rewards else 0.0


class _LoopState:
    def __init__(self) -> None:
        self.cfg: TrainConfig
        self.device: torch.device
        self.agent: DDQNAgent
        self.replay: ReplayBuffer
        self.env: SnakeEnv | None = None
        self.run_dir: Path
        self.agent_input_size: int
        self.state_shape: tuple[int, ...]
        self.global_step: int = 0
        self.best_avg_reward: float = float("-inf")
        self.best_eval_reward: float = float("-inf")
        self.episode_rows: list[dict[str, Any]] = []
        self.csv_committed: int = 0
        self.reward_window: deque[float] = deque()
        self.steps_window: deque[float] = deque()
        self.terminal_reason_counter: dict[str, int] = {}
        self.writer: Any = None
        self.plotter: LivePlotter
        self.jsonl_file: Any = None
        self.last_loss: float | None = None
        self.last_q_mean: float | None = None
        self.last_target_q_mean: float | None = None
        self.last_eval_reward: float | None = None
        self.previous_last_episode: dict[str, Any] | None = None
        self.existing_episode_count: int = 0
        self.resumed: bool = False
        self.nstep_serial: NStepAccumulator | None = None
        self.nstep_workers: dict[int, NStepAccumulator] = {}


def _init_windows(state: _LoopState, history: dict[str, Any] | None, window: int) -> None:
    maxlen = max(1, int(window))
    state.reward_window = deque(history["reward_window"] if history else [], maxlen=maxlen)
    state.steps_window = deque(history["steps_window"] if history else [], maxlen=maxlen)
    state.terminal_reason_counter = dict(history["terminal_reason_counter"]) if history else {}
    if history:
        state.previous_last_episode = history["last_row"]
        state.existing_episode_count = int(history["episodes_logged"])


def _bootstrap(
    cfg: TrainConfig,
    *,
    resume_state: Path | None,
    warm_start: Path | None,
    extra_episodes: int | None,
    warm_start_global_step: int | None,
    keep_env: bool,
) -> tuple[_LoopState, dict[str, Any]]:
    if resume_state is not None and warm_start is not None:
        raise ValueError("不能同时使用 --resume-state 与 --warm-start")

    state = _LoopState()
    state.device = torch.device(resolve_device(cfg.device))
    resume_meta: dict[str, Any] = {}

    if resume_state is not None:
        loaded = load_training_state(resume_state, state.device)
        cfg = loaded.cfg
        state.cfg = cfg
        state.resumed = True
        set_global_seed(cfg.env.seed)
        state.run_dir = resume_state.resolve().parent.parent
        (state.run_dir / "state").mkdir(parents=True, exist_ok=True)
        history = load_episode_history_snapshot(state.run_dir, cfg.moving_avg_window)
        _init_windows(state, history, cfg.moving_avg_window)
        env = build_initial_env(cfg)
        obs, _ = env.reset(seed=cfg.env.seed)
        state.agent_input_size = get_agent_input_size(cfg)
        probe, _ = extract_model_inputs(env, obs, cfg, state.agent_input_size)
        state.state_shape = tuple(probe.shape)
        if keep_env:
            state.env = env
        else:
            env.close()
        state.agent = create_agent(cfg, state.device, state.state_shape)
        state.agent.load_checkpoint_payload(loaded.agent_payload)
        state.replay = ReplayBuffer.from_state_dict(loaded.replay_state, state.device)
        resume_meta = loaded.meta
        state.global_step = int(resume_meta.get("global_step", 0))
        state.best_avg_reward = float(resume_meta.get("best_avg_reward", float("-inf")))
        state.best_eval_reward = float(resume_meta.get("best_eval_reward", float("-inf")))
        completed = int(resume_meta.get("completed_episodes", int(resume_meta.get("next_episode", 1)) - 1))
        completed = max(completed, state.existing_episode_count)
        resume_meta["completed_episodes"] = completed
        if extra_episodes is not None and extra_episodes > 0:
            if cfg.curriculum is None:
                cfg.episodes = completed + extra_episodes
                print(f"[恢复] 追加 {extra_episodes} 局，总局数上限: {cfg.episodes}")
            else:
                resume_meta["extra_episodes"] = int(extra_episodes)
                print(f"[恢复] 当前阶段追加 {extra_episodes} 局")
        elif cfg.curriculum is None and completed >= cfg.episodes:
            cfg.episodes = completed + max(cfg.episodes, 1000)
            print(f"[恢复] 已完成的局数 >= 原目标，自动追加，新总局数上限: {cfg.episodes}")
    else:
        state.cfg = cfg
        set_global_seed(cfg.env.seed)
        env = build_initial_env(cfg)
        obs, _ = env.reset(seed=cfg.env.seed)
        state.agent_input_size = get_agent_input_size(cfg)
        probe, _ = extract_model_inputs(env, obs, cfg, state.agent_input_size)
        state.state_shape = tuple(probe.shape)
        if keep_env:
            state.env = env
        else:
            env.close()
        state.agent = create_agent(cfg, state.device, state.state_shape)
        cap = None
        if cfg.curriculum is not None:
            cap = cfg.curriculum.stages[0].replay_capacity
        state.replay = create_replay(cfg, state.device, state.state_shape, capacity=cap)
        state.run_dir = prepare_run_dir(cfg)
        _init_windows(state, None, cfg.moving_avg_window)
        if warm_start is not None:
            state.agent.load_weights_only(warm_start)
            if warm_start_global_step is None:
                inferred = infer_last_global_step_from_warm_checkpoint(warm_start)
                state.global_step = int(inferred) if inferred is not None else 0
            else:
                state.global_step = max(0, int(warm_start_global_step))
            if state.global_step > 0:
                print(f"[热加载] 初始 global_step={state.global_step}（延续 ε 衰减进度）")

    purge = None
    if state.resumed:
        purge = int(resume_meta.get("completed_episodes", 0)) + 1
    state.writer = create_scalar_writer(cfg, state.run_dir, purge_step=purge)
    state.plotter = LivePlotter(enabled=cfg.live_plot)
    state.jsonl_file = (
        (state.run_dir / "logs" / "episodes.jsonl").open("a", encoding="utf-8") if cfg.save_jsonl else None
    )
    state.nstep_serial = NStepAccumulator(cfg.n_step, cfg.gamma)
    return state, resume_meta


def _agent_update(state: _LoopState, *, min_replay_size: int) -> None:
    metrics = state.agent.update(
        replay_buffer=state.replay,
        global_step=state.global_step,
        batch_size=state.cfg.batch_size,
        min_replay_size=min_replay_size,
        train_frequency=state.cfg.train_frequency,
        target_update_interval=state.cfg.target_update_interval,
        beta=_per_beta(state.cfg, state.global_step),
    )
    if metrics is not None:
        state.last_loss = metrics["loss"]
        state.last_q_mean = metrics["q_mean"]
        state.last_target_q_mean = metrics["target_q_mean"]


def _maybe_eval(state: _LoopState, episode: int) -> None:
    cfg = state.cfg
    if cfg.eval_episodes <= 0 or cfg.eval_interval <= 0:
        return
    if episode % int(cfg.eval_interval) != 0 and episode != 1:
        return
    eval_reward = run_greedy_eval(
        state.agent,
        cfg,
        agent_input_size=state.agent_input_size,
        seed=cfg.env.seed,
        episodes=cfg.eval_episodes,
    )
    state.last_eval_reward = eval_reward
    print(f"[Eval] episode={episode} avg_reward={eval_reward:.3f} over {cfg.eval_episodes} games")
    emit_event("eval", episode=episode, eval_reward=eval_reward, eval_episodes=cfg.eval_episodes)
    if eval_reward > state.best_eval_reward:
        state.best_eval_reward = eval_reward
        state.agent.save_checkpoint(
            state.run_dir / "checkpoints" / "best.pt",
            extra=_ckpt_extra(state, episode),
        )


def _ckpt_extra(state: _LoopState, episode: int, stage_index: int | None = None) -> dict[str, Any]:
    extra = {
        "episode": episode,
        "global_step": state.global_step,
        "best_avg_reward": state.best_avg_reward,
        "best_eval_reward": state.best_eval_reward,
        "run_dir": str(state.run_dir),
    }
    if stage_index is not None:
        extra["stage_index"] = stage_index
    return extra


def _log_episode(
    state: _LoopState,
    *,
    episode: int,
    reward: float,
    steps: int,
    foods: int,
    score: int,
    terminal_reason: str,
    board_size: int,
    epsilon: float,
    stage_index: int | None,
    episodes_total: int | None,
    log_prefix: str,
    force_log: bool,
) -> dict[str, Any]:
    state.terminal_reason_counter[terminal_reason] = state.terminal_reason_counter.get(terminal_reason, 0) + 1
    state.reward_window.append(float(reward))
    avg_reward = float(sum(state.reward_window) / len(state.reward_window))
    state.steps_window.append(float(steps))
    avg_steps = float(sum(state.steps_window) / len(state.steps_window))
    if avg_reward > state.best_avg_reward:
        state.best_avg_reward = avg_reward
        if state.cfg.eval_episodes <= 0:
            state.agent.save_checkpoint(
                state.run_dir / "checkpoints" / "best.pt",
                extra=_ckpt_extra(state, episode, stage_index),
            )
    row = {
        "episode": episode,
        "global_step": state.global_step,
        "reward": float(reward),
        "avg_reward": avg_reward,
        "best_avg_reward": state.best_avg_reward,
        "steps": int(steps),
        "avg_steps": avg_steps,
        "foods": int(foods),
        "score": int(score),
        "epsilon": float(epsilon),
        "loss": state.last_loss,
        "q_mean": state.last_q_mean,
        "target_q_mean": state.last_target_q_mean,
        "eval_reward": state.last_eval_reward,
        "terminal_reason": terminal_reason,
        "win": 1 if terminal_reason == TERMINAL_REASONS["BOARD_FULL"] else 0,
        "board_size": int(board_size),
        "stage_index": stage_index,
    }
    state.episode_rows.append(row)
    maybe_write_episode(
        writer=state.writer,
        plotter=state.plotter,
        jsonl_file=state.jsonl_file,
        row=row,
        terminal_reason_counter=state.terminal_reason_counter,
        tensorboard_log_interval=state.cfg.tensorboard_log_interval,
        jsonl_flush_interval=state.cfg.jsonl_flush_interval,
    )
    if force_log or episode % state.cfg.log_interval == 0:
        print(
            f"{log_prefix}"
            f"board={int(board_size):2d} reward={float(reward):8.3f} avg_reward={avg_reward:8.3f} "
            f"steps={int(steps):4d} foods={int(foods):3d} "
            f"score={int(score):5d} eps={epsilon:6.3f} terminal={terminal_reason}"
        )
    emit_event(
        "episode",
        episode=episode,
        episodes_total=episodes_total,
        global_step=state.global_step,
        reward=float(reward),
        avg_reward=avg_reward,
        foods=int(foods),
        epsilon=float(epsilon),
        stage_index=stage_index,
        terminal_reason=terminal_reason,
        board_size=int(board_size),
    )
    return row


def _persist(
    state: _LoopState,
    *,
    episode: int,
    stage_index: int | None = None,
    worker_episode_counters: list[int] | None = None,
    extra_meta: dict[str, Any] | None = None,
    save_named: bool = True,
) -> None:
    extra = _ckpt_extra(state, episode, stage_index)
    if save_named:
        state.agent.save_checkpoint(state.run_dir / "checkpoints" / "latest.pt", extra=extra)
        state.agent.save_checkpoint(state.run_dir / "checkpoints" / f"ep_{episode:05d}.pt", extra=extra)
    persist_training_state(
        state.run_dir,
        state.agent,
        state.replay,
        state.cfg,
        global_step=state.global_step,
        next_episode=episode + 1,
        best_avg_reward=state.best_avg_reward,
        completed_episodes=episode,
        worker_episode_counters=worker_episode_counters,
        extra_meta=extra_meta,
    )
    state.csv_committed = append_episode_csv_incremental(
        state.run_dir / "logs" / "episodes.csv", state.episode_rows, state.csv_committed
    )


def _run_serial_episode(
    state: _LoopState,
    *,
    episode: int,
    board_size: int,
    reset_options: dict[str, Any] | None,
    min_replay_size: int,
    epsilon_clock: list[int] | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    assert state.env is not None
    assert state.nstep_serial is not None
    env = state.env
    cfg = state.cfg
    obs, _ = env.reset(
        seed=None if cfg.env.seed is None else cfg.env.seed + episode,
        options=reset_options,
    )
    current, global_feat = extract_model_inputs(env, obs, cfg, state.agent_input_size)
    episode_reward = 0.0
    info: dict[str, Any] = {"terminal_reason": ""}
    acc = state.nstep_serial
    acc.reset()
    for _ in range(cfg.max_steps_per_episode):
        eps_step = epsilon_clock[0] if epsilon_clock is not None else state.global_step
        action = state.agent.select_action(
            current,
            global_step=eps_step,
            eval_mode=False,
            global_feat=global_feat,
        )
        next_obs, reward, done, info = env.step(action, lightweight_info=cfg.lightweight_step_info)
        nxt, next_global_feat = extract_model_inputs(env, next_obs, cfg, state.agent_input_size)
        _push_nstep(
            acc,
            state.replay,
            current,
            action,
            reward,
            nxt,
            done,
            global_feat=global_feat,
            next_global_feat=next_global_feat,
        )
        state.global_step += 1
        if epsilon_clock is not None:
            epsilon_clock[0] += 1
        episode_reward += float(reward)
        _agent_update(state, min_replay_size=min_replay_size)
        current = nxt
        global_feat = next_global_feat
        if done:
            break
    return episode_reward, info, env.get_episode_stats()


def _run_standard(state: _LoopState, resume_meta: dict[str, Any]) -> dict[str, Any]:
    cfg = state.cfg
    start_episode = int(resume_meta.get("completed_episodes", 0)) + 1 if state.resumed else 1
    if cfg.parallel.enabled:
        return _run_parallel_standard(state, resume_meta, start_episode)

    assert state.env is not None
    for episode in range(start_episode, cfg.episodes + 1):
        reset_options = None
        if cfg.random_board is not None:
            board_size, timeout = sample_random_board(cfg)
            reset_options = build_env_options(cfg.env, board_size=board_size, max_steps_without_food=timeout)
        else:
            board_size = cfg.env.board_size
        reward, info, stats = _run_serial_episode(
            state,
            episode=episode,
            board_size=board_size,
            reset_options=reset_options,
            min_replay_size=cfg.min_replay_size,
        )
        terminal = str(info.get("terminal_reason", "")) or "running"
        _log_episode(
            state,
            episode=episode,
            reward=reward,
            steps=int(stats["steps"]),
            foods=int(stats["foods"]),
            score=int(stats["score_end"]),
            terminal_reason=terminal,
            board_size=board_size,
            epsilon=state.agent.epsilon_by_step(state.global_step),
            stage_index=None,
            episodes_total=cfg.episodes,
            log_prefix=f"[Episode {episode:5d}] ",
            force_log=episode == start_episode,
        )
        _maybe_eval(state, episode)
        if episode % cfg.checkpoint_interval == 0 or episode == cfg.episodes:
            _persist(state, episode=episode)

    state.env.close()
    return _finish(state, "random_board" if cfg.random_board is not None else "standard", start_episode)


def _run_parallel_standard(state: _LoopState, resume_meta: dict[str, Any], start_episode: int) -> dict[str, Any]:
    cfg = state.cfg
    completed = start_episode - 1
    worker_starts: list[int] | None = None
    worker_done: dict[int, int] = defaultdict(int)
    raw_w = resume_meta.get("worker_episode_counters")
    nw = int(cfg.parallel.num_workers)
    if isinstance(raw_w, list) and len(raw_w) == nw:
        worker_starts = [int(x) for x in raw_w]
        for i, count in enumerate(worker_starts):
            worker_done[i] = count
    runtime_cfg = _build_parallel_runtime_for_standard(cfg)
    actor_pool = _build_actor_pool(
        cfg=cfg,
        agent=state.agent,
        state_shape=state.state_shape,
        agent_input_size=state.agent_input_size,
        runtime_cfg=runtime_cfg,
        worker_episode_counter_starts=worker_starts,
    )
    policy_version = _broadcast_parallel_policy(
        actor_pool=actor_pool,
        agent=state.agent,
        epsilon=state.agent.epsilon_by_step(state.global_step),
        policy_version=0,
    )
    for worker_id in range(nw):
        state.nstep_workers[worker_id] = NStepAccumulator(cfg.n_step, cfg.gamma)
    try:
        while completed < cfg.episodes:
            try:
                msg = actor_pool.out_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if isinstance(msg, TransitionMessage):
                acc = state.nstep_workers.setdefault(
                    int(msg.worker_id), NStepAccumulator(cfg.n_step, cfg.gamma)
                )
                _push_nstep(
                    acc,
                    state.replay,
                    msg.state,
                    msg.action,
                    msg.reward,
                    msg.next_state,
                    msg.done,
                    global_feat=msg.global_feat,
                    next_global_feat=msg.next_global_feat,
                )
                state.global_step += 1
                _agent_update(state, min_replay_size=cfg.min_replay_size)
                if state.global_step % cfg.parallel.weight_sync_interval_steps == 0:
                    policy_version = _broadcast_parallel_policy(
                        actor_pool=actor_pool,
                        agent=state.agent,
                        epsilon=state.agent.epsilon_by_step(state.global_step),
                        policy_version=policy_version,
                    )
                continue
            if not isinstance(msg, EpisodeDoneMessage):
                continue
            completed += 1
            worker_done[int(msg.worker_id)] += 1
            state.nstep_workers.get(int(msg.worker_id), NStepAccumulator(1, cfg.gamma)).reset()
            _log_episode(
                state,
                episode=completed,
                reward=float(msg.reward),
                steps=int(msg.steps),
                foods=int(msg.foods),
                score=int(msg.score),
                terminal_reason=str(msg.terminal_reason) or "running",
                board_size=int(msg.board_size),
                epsilon=state.agent.epsilon_by_step(state.global_step),
                stage_index=None,
                episodes_total=cfg.episodes,
                log_prefix=f"[Episode {completed:5d}] ",
                force_log=completed == start_episode,
            )
            _maybe_eval(state, completed)
            if completed % cfg.checkpoint_interval == 0 or completed == cfg.episodes:
                wlist = [worker_done[i] for i in range(nw)]
                _persist(state, episode=completed, worker_episode_counters=wlist)
    finally:
        stop_actor_pool(actor_pool)
    mode = "random_board_parallel" if cfg.random_board is not None else "standard_parallel"
    summary = _finish(state, mode, start_episode)
    summary["parallel_workers"] = cfg.parallel.num_workers
    return summary


def _apply_stage_replay(state: _LoopState, stage: Any, stage_index: int) -> None:
    cfg = state.cfg
    assert cfg.curriculum is not None
    if stage_index == 1 and not state.resumed:
        return
    if cfg.curriculum.carry_replay:
        if state.replay.capacity != int(stage.replay_capacity):
            old_size = len(state.replay)
            state.replay = state.replay.resized_copy(stage.replay_capacity)
            print(f"[Curriculum] 迁移回放池: {old_size} 条经验, 容量 {state.replay.capacity}")
    else:
        state.replay = create_replay(cfg, state.device, state.state_shape, capacity=stage.replay_capacity)


def _run_curriculum(state: _LoopState, resume_meta: dict[str, Any]) -> dict[str, Any]:
    cfg = state.cfg
    if cfg.curriculum is None:
        raise ValueError("curriculum 配置不存在")
    start_stage = int(resume_meta.get("stage_index", 1))
    start_stage = max(1, min(start_stage, len(cfg.curriculum.stages)))
    start_stage_episode = int(resume_meta.get("stage_episode", 0))
    stage_step = int(resume_meta.get("stage_step", 0))
    foods_saved = resume_meta.get("foods_window") or []
    absolute_episode = int(resume_meta.get("completed_episodes", 0)) if state.resumed else 0
    extra = int(resume_meta.get("extra_episodes", 0) or 0)
    stage_summaries: list[dict[str, Any]] = []
    actor_pool: ActorPoolHandle | None = None
    policy_version = 0
    worker_done: dict[int, int] = defaultdict(int)

    if cfg.parallel.enabled:
        runtime = _build_parallel_runtime_for_stage(
            cfg, cfg.curriculum.stages[start_stage - 1], stage_index=start_stage
        )
        raw_w = resume_meta.get("worker_episode_counters")
        starts = [int(x) for x in raw_w] if isinstance(raw_w, list) else None
        actor_pool = _build_actor_pool(
            cfg=cfg,
            agent=state.agent,
            state_shape=state.state_shape,
            agent_input_size=state.agent_input_size,
            runtime_cfg=runtime,
            worker_episode_counter_starts=starts,
        )
        policy_version = _broadcast_parallel_policy(
            actor_pool=actor_pool,
            agent=state.agent,
            epsilon=state.agent.epsilon_by_step(stage_step),
            policy_version=0,
        )

    try:
        for stage_index in range(start_stage, len(cfg.curriculum.stages) + 1):
            stage = cfg.curriculum.stages[stage_index - 1]
            if stage_index > start_stage:
                start_stage_episode = 0
                stage_step = 0
                foods_window: list[int] = []
                _apply_stage_replay(state, stage, stage_index)
            else:
                foods_window = [int(x) for x in foods_saved]
                if not state.resumed:
                    _apply_stage_replay(state, stage, stage_index)
            state.agent.reset_epsilon(stage.epsilon_start, stage.epsilon_end, stage.epsilon_decay_steps)
            label = curriculum_stage_label(stage)
            target_episodes = int(stage.episodes)
            if extra > 0 and stage_index == start_stage:
                target_episodes = start_stage_episode + extra
                extra = 0
            print(
                f"\n=== Curriculum Stage {stage_index}/{len(cfg.curriculum.stages)} | "
                f"board={label} | episodes={target_episodes} ==="
            )
            emit_event(
                "stage",
                stage_index=stage_index,
                stages_total=len(cfg.curriculum.stages),
                stage_label=label,
                episodes=target_episodes,
            )
            if actor_pool is not None:
                broadcast_runtime(actor_pool, _build_parallel_runtime_for_stage(cfg, stage, stage_index=stage_index))
                policy_version = _broadcast_parallel_policy(
                    actor_pool=actor_pool,
                    agent=state.agent,
                    epsilon=state.agent.epsilon_by_step(stage_step),
                    policy_version=policy_version,
                )
            stage_start = absolute_episode + 1
            promoted = False
            if cfg.parallel.enabled:
                assert actor_pool is not None
                stage_done = start_stage_episode
                while stage_done < target_episodes:
                    try:
                        msg = actor_pool.out_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    if isinstance(msg, TransitionMessage):
                        acc = state.nstep_workers.setdefault(
                            int(msg.worker_id), NStepAccumulator(cfg.n_step, cfg.gamma)
                        )
                        _push_nstep(
                            acc,
                            state.replay,
                            msg.state,
                            msg.action,
                            msg.reward,
                            msg.next_state,
                            msg.done,
                            global_feat=msg.global_feat,
                            next_global_feat=msg.next_global_feat,
                        )
                        state.global_step += 1
                        stage_step += 1
                        _agent_update(state, min_replay_size=stage.min_replay_size)
                        if state.global_step % cfg.parallel.weight_sync_interval_steps == 0:
                            policy_version = _broadcast_parallel_policy(
                                actor_pool=actor_pool,
                                agent=state.agent,
                                epsilon=state.agent.epsilon_by_step(stage_step),
                                policy_version=policy_version,
                            )
                        continue
                    if not isinstance(msg, EpisodeDoneMessage):
                        continue
                    stage_done += 1
                    absolute_episode += 1
                    worker_done[int(msg.worker_id)] += 1
                    _log_episode(
                        state,
                        episode=absolute_episode,
                        reward=float(msg.reward),
                        steps=int(msg.steps),
                        foods=int(msg.foods),
                        score=int(msg.score),
                        terminal_reason=str(msg.terminal_reason) or "running",
                        board_size=int(msg.board_size),
                        epsilon=state.agent.epsilon_by_step(stage_step),
                        stage_index=stage_index,
                        episodes_total=None,
                        log_prefix=f"[Stage {stage_index} | Ep {stage_done:4d}/{target_episodes}] ",
                        force_log=stage_done == 1,
                    )
                    foods_window.append(int(msg.foods))
                    if len(foods_window) > stage.promotion_window:
                        foods_window.pop(0)
                    _maybe_eval(state, absolute_episode)
                    if absolute_episode % cfg.checkpoint_interval == 0:
                        _persist(
                            state,
                            episode=absolute_episode,
                            stage_index=stage_index,
                            worker_episode_counters=[worker_done[i] for i in range(cfg.parallel.num_workers)],
                            extra_meta={
                                "stage_index": stage_index,
                                "stage_episode": stage_done,
                                "stage_step": stage_step,
                                "foods_window": foods_window,
                                "best_eval_reward": state.best_eval_reward,
                            },
                        )
                    if (
                        stage.promotion_threshold_foods > 0
                        and stage_done >= stage.promotion_min_episodes
                        and len(foods_window) >= stage.promotion_window
                    ):
                        avg_foods = sum(foods_window) / len(foods_window)
                        if avg_foods >= stage.promotion_threshold_foods:
                            print(
                                f"\n[Curriculum] 达到晋升条件！"
                                f"最近 {stage.promotion_window} 局平均食物: {avg_foods:.2f} "
                                f">= 门槛 {stage.promotion_threshold_foods:.1f} "
                                f"(阶段 {stage_index}, 第 {stage_done}/{target_episodes} 局)"
                            )
                            promoted = True
                            break
            else:
                assert state.env is not None
                for stage_episode in range(start_stage_episode + 1, target_episodes + 1):
                    absolute_episode += 1
                    board_size, timeout = sample_curriculum_stage_board(cfg, stage)
                    stage_clock = [stage_step]
                    reward, info, stats = _run_serial_episode(
                        state,
                        episode=absolute_episode,
                        board_size=board_size,
                        reset_options=build_env_options(
                            cfg.env, board_size=board_size, max_steps_without_food=timeout
                        ),
                        min_replay_size=stage.min_replay_size,
                        epsilon_clock=stage_clock,
                    )
                    stage_step = stage_clock[0]
                    terminal = str(info.get("terminal_reason", "")) or "running"
                    _log_episode(
                        state,
                        episode=absolute_episode,
                        reward=reward,
                        steps=int(stats["steps"]),
                        foods=int(stats["foods"]),
                        score=int(stats["score_end"]),
                        terminal_reason=terminal,
                        board_size=board_size,
                        epsilon=state.agent.epsilon_by_step(stage_step),
                        stage_index=stage_index,
                        episodes_total=None,
                        log_prefix=f"[Stage {stage_index} | Ep {stage_episode:4d}/{target_episodes}] ",
                        force_log=stage_episode == start_stage_episode + 1,
                    )
                    foods_window.append(int(stats["foods"]))
                    if len(foods_window) > stage.promotion_window:
                        foods_window.pop(0)
                    _maybe_eval(state, absolute_episode)
                    if absolute_episode % cfg.checkpoint_interval == 0:
                        _persist(
                            state,
                            episode=absolute_episode,
                            stage_index=stage_index,
                            extra_meta={
                                "stage_index": stage_index,
                                "stage_episode": stage_episode,
                                "stage_step": stage_step,
                                "foods_window": foods_window,
                                "best_eval_reward": state.best_eval_reward,
                            },
                        )
                    if (
                        stage.promotion_threshold_foods > 0
                        and stage_episode >= stage.promotion_min_episodes
                        and len(foods_window) >= stage.promotion_window
                    ):
                        avg_foods = sum(foods_window) / len(foods_window)
                        if avg_foods >= stage.promotion_threshold_foods:
                            print(
                                f"\n[Curriculum] 达到晋升条件！"
                                f"最近 {stage.promotion_window} 局平均食物: {avg_foods:.2f} "
                                f">= 门槛 {stage.promotion_threshold_foods:.1f} "
                                f"(阶段 {stage_index}, 第 {stage_episode}/{target_episodes} 局)"
                            )
                            promoted = True
                            break
            stage_rows = [row for row in state.episode_rows if row["stage_index"] == stage_index]
            stage_summaries.append(
                {
                    "stage_index": stage_index,
                    "board_size": None if stage.board_sizes else stage.board_size,
                    "stage_label": label,
                    "episodes": len(stage_rows),
                    "episode_range": [stage_start, absolute_episode],
                    "avg_reward_last": stage_rows[-1]["avg_reward"] if stage_rows else None,
                    "promoted": promoted,
                }
            )
            start_stage_episode = 0
    finally:
        if actor_pool is not None:
            stop_actor_pool(actor_pool)
        if state.env is not None:
            state.env.close()

    if state.episode_rows:
        last_ep = int(state.episode_rows[-1]["episode"])
        last_stage = stage_summaries[-1]["stage_index"] if stage_summaries else None
        _persist(state, episode=last_ep, stage_index=last_stage)
    summary = _finish(state, "curriculum_parallel" if cfg.parallel.enabled else "curriculum", 1)
    summary["stage_summaries"] = stage_summaries
    if cfg.parallel.enabled:
        summary["parallel_workers"] = cfg.parallel.num_workers
    return summary


def _finish(state: _LoopState, mode: str, start_episode: int) -> dict[str, Any]:
    emit_event("done", run_dir=str(state.run_dir), mode=mode, episodes=len(state.episode_rows))
    summary = {
        "run_dir": str(state.run_dir),
        "mode": mode,
        "episodes": state.episode_rows[-1]["episode"] if state.episode_rows else state.existing_episode_count,
        "best_avg_reward": state.best_avg_reward,
        "best_eval_reward": state.best_eval_reward,
        "final_global_step": state.global_step,
        "last_episode": state.episode_rows[-1] if state.episode_rows else (state.previous_last_episode or {}),
        "model_type": state.cfg.model_type,
    }
    if state.resumed:
        summary["resumed_from_episode"] = state.existing_episode_count
    return finalize_run(
        cfg=state.cfg,
        run_dir=state.run_dir,
        episode_rows=state.episode_rows,
        writer=state.writer,
        plotter=state.plotter,
        jsonl_file=state.jsonl_file,
        summary=summary,
        csv_committed=state.csv_committed,
    )


def run_unified_training(
    cfg: TrainConfig,
    *,
    resume_state: Path | None = None,
    warm_start: Path | None = None,
    extra_episodes: int | None = None,
    warm_start_global_step: int | None = None,
) -> dict[str, Any]:
    keep_env = not cfg.parallel.enabled
    state, resume_meta = _bootstrap(
        cfg,
        resume_state=resume_state,
        warm_start=warm_start,
        extra_episodes=extra_episodes,
        warm_start_global_step=warm_start_global_step,
        keep_env=keep_env,
    )
    if cfg.curriculum is not None:
        return _run_curriculum(state, resume_meta)
    return _run_standard(state, resume_meta)
