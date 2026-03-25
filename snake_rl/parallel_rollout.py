from __future__ import annotations

from dataclasses import asdict, dataclass
import multiprocessing as mp
import queue
import random
import time
from typing import Any, Literal

import numpy as np
import torch

from .agent import AgentHyperParams, DDQNAgent
from .config import EnvPreset, ParallelRolloutConfig, resolve_device
from .env import SnakeEnv, SnakeEnvConfig


WorkerMode = Literal["fixed", "random"]


@dataclass(slots=True)
class WorkerEpisodeConfig:
    mode: WorkerMode
    max_steps_per_episode: int
    stage_index: int | None = None
    fixed_board_size: int | None = None
    fixed_timeout: int | None = None
    board_sizes: list[int] | None = None
    weights: list[float] | None = None
    timeout_scale: float = 1.0


@dataclass(slots=True)
class PolicySnapshot:
    version: int
    epsilon: float
    online_state_dict_cpu: dict[str, torch.Tensor]


@dataclass(slots=True)
class TransitionMessage:
    worker_id: int
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    global_feat: np.ndarray | None
    next_global_feat: np.ndarray | None


@dataclass(slots=True)
class EpisodeDoneMessage:
    worker_id: int
    stage_index: int | None
    board_size: int
    reward: float
    steps: int
    foods: int
    score: int
    terminal_reason: str


@dataclass(slots=True)
class ActorPoolHandle:
    out_queue: mp.queues.Queue[Any]
    cmd_queues: list[mp.queues.Queue[Any]]
    processes: list[mp.Process]
    policy_version: int = 0


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


def extract_actor_inputs(
    env: SnakeEnv,
    obs_hwc: np.ndarray,
    *,
    model_type: str,
    local_patch_size: int,
    use_padding: bool,
    agent_input_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if model_type == "hybrid":
        patch = env.get_local_patch(local_patch_size)
        return hwc_to_chw(patch), env.get_global_features()
    state = hwc_to_chw(obs_hwc)
    if use_padding:
        state = center_pad_chw(state, agent_input_size)
    return state, None


def _sample_board_and_timeout(runtime: WorkerEpisodeConfig) -> tuple[int, int]:
    if runtime.mode == "fixed":
        if runtime.fixed_board_size is None:
            raise ValueError("fixed mode 需要 fixed_board_size")
        board_size = int(runtime.fixed_board_size)
        if runtime.fixed_timeout is None:
            timeout = board_size * board_size
        else:
            timeout = int(runtime.fixed_timeout)
        return board_size, max(1, timeout)

    if not runtime.board_sizes:
        raise ValueError("random mode 需要 board_sizes")
    board_size = int(random.choices(runtime.board_sizes, weights=runtime.weights, k=1)[0])
    timeout = max(1, int(round(board_size * board_size * float(runtime.timeout_scale))))
    return board_size, timeout


def _build_env_options(
    env: EnvPreset,
    *,
    board_size: int,
    max_steps_without_food: int,
) -> dict[str, Any]:
    return {
        "difficulty": env.difficulty,
        "mode": env.mode,
        "board_size": int(board_size),
        "enable_bonus_food": env.enable_bonus_food,
        "enable_obstacles": env.enable_obstacles,
        "allow_leveling": env.allow_leveling,
        "max_steps_without_food": int(max_steps_without_food),
    }


def actor_worker_main(
    worker_id: int,
    init_payload: dict[str, Any],
    out_queue: mp.queues.Queue[Any],
    cmd_queue: mp.queues.Queue[Any],
) -> None:
    rng = random.Random()
    np_rng = np.random.default_rng()

    seed = init_payload.get("seed")
    if seed is not None:
        rng.seed(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    env = SnakeEnv(
        config=SnakeEnvConfig(**init_payload["base_env_options"]),
        reward_weights=init_payload.get("reward_weights"),
        seed=seed,
    )

    hp = AgentHyperParams(**init_payload["hp"])
    actor_device = torch.device(resolve_device(init_payload["actor_device"]))
    agent = DDQNAgent(
        observation_shape=tuple(init_payload["observation_shape"]),
        num_actions=int(init_payload["num_actions"]),
        device=actor_device,
        hp=hp,
        model_type=init_payload["model_type"],
    )
    agent.online_net.eval()
    agent.target_net.eval()

    runtime = WorkerEpisodeConfig(**init_payload["runtime"])
    lightweight_info = bool(init_payload["lightweight_step_info"])
    model_type = str(init_payload["model_type"])
    local_patch_size = int(init_payload["local_patch_size"])
    use_padding = bool(init_payload["use_padding"])
    agent_input_size = int(init_payload["agent_input_size"])

    current_epsilon = 1.0
    current_version = -1
    episode_counter = int(init_payload.get("episode_counter_start", 0))
    active = True
    state: np.ndarray | None = None
    global_feat: np.ndarray | None = None
    episode_reward = 0.0
    episode_steps = 0
    board_size = int(init_payload["base_env_options"]["board_size"])

    while active:
        try:
            while True:
                cmd = cmd_queue.get_nowait()
                cmd_type = cmd.get("type")
                if cmd_type == "stop":
                    active = False
                    break
                if cmd_type == "runtime":
                    runtime = WorkerEpisodeConfig(**cmd["payload"])
                    state = None
                    global_feat = None
                elif cmd_type == "policy":
                    snapshot = PolicySnapshot(**cmd["payload"])
                    if snapshot.version > current_version:
                        agent.online_net.load_state_dict(snapshot.online_state_dict_cpu)
                        agent.target_net.load_state_dict(snapshot.online_state_dict_cpu)
                        current_epsilon = float(snapshot.epsilon)
                        current_version = int(snapshot.version)
        except queue.Empty:
            pass

        if not active:
            break

        if state is None:
            board_size, timeout = _sample_board_and_timeout(runtime)
            episode_counter += 1
            episode_seed = seed + episode_counter if seed is not None else None
            obs, _ = env.reset(
                seed=episode_seed,
                options=_build_env_options(
                    env.config,
                    board_size=board_size,
                    max_steps_without_food=timeout,
                ),
            )
            state, global_feat = extract_actor_inputs(
                env,
                obs,
                model_type=model_type,
                local_patch_size=local_patch_size,
                use_padding=use_padding,
                agent_input_size=agent_input_size,
            )
            episode_reward = 0.0
            episode_steps = 0

        if current_version < 0:
            time.sleep(0.002)
            continue

        if np_rng.random() < current_epsilon:
            action = int(np_rng.integers(0, init_payload["num_actions"]))
        else:
            action = int(
                agent.select_action(
                    state,
                    global_step=0,
                    eval_mode=True,
                    global_feat=global_feat,
                )
            )

        next_obs, reward, done, info = env.step(action, lightweight_info=lightweight_info)
        next_state, next_global_feat = extract_actor_inputs(
            env,
            next_obs,
            model_type=model_type,
            local_patch_size=local_patch_size,
            use_padding=use_padding,
            agent_input_size=agent_input_size,
        )

        transition = TransitionMessage(
            worker_id=worker_id,
            state=np.asarray(state > 0.5, dtype=np.uint8),
            action=action,
            reward=float(reward),
            next_state=np.asarray(next_state > 0.5, dtype=np.uint8),
            done=bool(done),
            global_feat=None if global_feat is None else global_feat.astype(np.float32, copy=False),
            next_global_feat=None
            if next_global_feat is None
            else next_global_feat.astype(np.float32, copy=False),
        )
        out_queue.put(transition)

        episode_steps += 1
        episode_reward += float(reward)
        state = next_state
        global_feat = next_global_feat

        if done or episode_steps >= int(runtime.max_steps_per_episode):
            stats = env.get_episode_stats()
            terminal_reason = str(info.get("terminal_reason", "")) or "running"
            out_queue.put(
                EpisodeDoneMessage(
                    worker_id=worker_id,
                    stage_index=runtime.stage_index,
                    board_size=board_size,
                    reward=float(episode_reward),
                    steps=int(stats["steps"]),
                    foods=int(stats["foods"]),
                    score=int(stats["score_end"]),
                    terminal_reason=terminal_reason,
                )
            )
            state = None
            global_feat = None

        sleep_ms = int(init_payload.get("actor_loop_sleep_ms", 0))
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    env.close()


def make_policy_snapshot(agent: DDQNAgent, *, epsilon: float, version: int) -> PolicySnapshot:
    cpu_weights = {
        name: tensor.detach().cpu()
        for name, tensor in agent.online_net.state_dict().items()
    }
    return PolicySnapshot(
        version=int(version),
        epsilon=float(epsilon),
        online_state_dict_cpu=cpu_weights,
    )


def start_actor_pool(
    *,
    parallel_cfg: ParallelRolloutConfig,
    env_cfg: EnvPreset,
    reward_weights: dict[str, float] | None,
    hp: AgentHyperParams,
    model_type: str,
    observation_shape: tuple[int, int, int],
    local_patch_size: int,
    agent_input_size: int,
    use_padding: bool,
    lightweight_step_info: bool,
    runtime_cfg: WorkerEpisodeConfig,
    num_actions: int = 3,
    worker_episode_counter_starts: list[int] | None = None,
) -> ActorPoolHandle:
    ctx = mp.get_context("spawn")
    out_queue: mp.queues.Queue[Any] = ctx.Queue(maxsize=max(128, int(parallel_cfg.queue_capacity)))

    processes: list[mp.Process] = []
    cmd_queues: list[mp.queues.Queue[Any]] = []
    for worker_id in range(int(parallel_cfg.num_workers)):
        cmd_q: mp.queues.Queue[Any] = ctx.Queue(maxsize=8)
        cmd_queues.append(cmd_q)

        if env_cfg.seed is None:
            worker_seed = None
        else:
            worker_seed = int(env_cfg.seed) + worker_id * int(parallel_cfg.actor_seed_stride)

        ep_start = 0
        if worker_episode_counter_starts is not None and worker_id < len(worker_episode_counter_starts):
            ep_start = int(worker_episode_counter_starts[worker_id])

        init_payload = {
            "seed": worker_seed,
            "episode_counter_start": ep_start,
            "hp": asdict(hp),
            "num_actions": int(num_actions),
            "model_type": model_type,
            "actor_device": parallel_cfg.actor_device,
            "observation_shape": tuple(int(v) for v in observation_shape),
            "local_patch_size": int(local_patch_size),
            "agent_input_size": int(agent_input_size),
            "use_padding": bool(use_padding),
            "lightweight_step_info": bool(lightweight_step_info),
            "reward_weights": reward_weights,
            "base_env_options": {
                "difficulty": env_cfg.difficulty,
                "mode": env_cfg.mode,
                "board_size": int(env_cfg.board_size),
                "enable_bonus_food": env_cfg.enable_bonus_food,
                "enable_obstacles": env_cfg.enable_obstacles,
                "allow_leveling": env_cfg.allow_leveling,
                "max_steps_without_food": int(env_cfg.max_steps_without_food),
            },
            "runtime": asdict(runtime_cfg),
            "actor_loop_sleep_ms": int(parallel_cfg.actor_loop_sleep_ms),
        }
        proc = ctx.Process(
            target=actor_worker_main,
            args=(worker_id, init_payload, out_queue, cmd_q),
            daemon=True,
        )
        proc.start()
        processes.append(proc)

    return ActorPoolHandle(
        out_queue=out_queue,
        cmd_queues=cmd_queues,
        processes=processes,
        policy_version=0,
    )


def broadcast_policy(handle: ActorPoolHandle, snapshot: PolicySnapshot) -> None:
    payload = {"type": "policy", "payload": asdict(snapshot)}
    for cmd_q in handle.cmd_queues:
        cmd_q.put(payload)


def broadcast_runtime(handle: ActorPoolHandle, runtime_cfg: WorkerEpisodeConfig) -> None:
    payload = {"type": "runtime", "payload": asdict(runtime_cfg)}
    for cmd_q in handle.cmd_queues:
        cmd_q.put(payload)


def stop_actor_pool(handle: ActorPoolHandle, timeout_s: float = 5.0) -> None:
    for cmd_q in handle.cmd_queues:
        try:
            cmd_q.put({"type": "stop"})
        except Exception:
            pass

    deadline = time.time() + float(timeout_s)
    for proc in handle.processes:
        remaining = max(0.0, deadline - time.time())
        proc.join(timeout=remaining)
        if proc.is_alive():
            proc.terminate()
    for proc in handle.processes:
        proc.join(timeout=0.2)
