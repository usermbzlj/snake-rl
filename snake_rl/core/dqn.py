"""Double Dueling DQN trainer with GPU component replay."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from snake_rl.core.config import ExperimentConfig, live_field_keys
from snake_rl.core.env import N_COMPONENTS, BatchedSnakeEnv, Obs
from snake_rl.core.network import SnakeNet
from snake_rl.core.replay import ComponentReplayBuffer
from snake_rl.core.trainer import EpisodeStats, reward_weights_tensor, set_seed


class DQNTrainer:
    def __init__(self, config: ExperimentConfig, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        set_seed(config.run.seed)

        dqn = config.dqn
        self.env = BatchedSnakeEnv(
            num_envs=dqn.num_envs,
            min_size=config.env.min_size,
            max_size=config.env.max_size,
            hunger_factor=config.env.hunger_factor,
            device=self.device,
            seed=config.run.seed,
        )
        self.online = SnakeNet(mode="dqn", width=config.model.width).to(self.device)
        self.target = SnakeNet(mode="dqn", width=config.model.width).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        if config.run.compile and self.device.type == "cuda":
            with contextlib.suppress(Exception):
                self.online = torch.compile(self.online)  # type: ignore[assignment]

        self.opt = torch.optim.Adam(self.online.parameters(), lr=dqn.lr, eps=1e-5)
        self.weights = reward_weights_tensor(config, self.device)

        w = 2 * self.env.grid - 1
        self.replay = ComponentReplayBuffer(
            capacity=dqn.replay_size,
            grid_shape=(4, w, w),
            scalar_dim=4,
            n_components=N_COMPONENTS,
            device=self.device,
        )

        # Per-env n-step ring buffers
        ns = dqn.n_step
        ne = dqn.num_envs
        self._ns = ns
        self._buf_len = torch.zeros(ne, dtype=torch.int32, device=self.device)
        self._buf_grid = torch.zeros(ne, ns, 4, w, w, dtype=torch.float32, device=self.device)
        self._buf_scal = torch.zeros(ne, ns, 4, dtype=torch.float32, device=self.device)
        self._buf_act = torch.zeros(ne, ns, dtype=torch.int64, device=self.device)
        self._buf_comp = torch.zeros(ne, ns, N_COMPONENTS, dtype=torch.float32, device=self.device)

        self.env_steps = 0
        self.iteration = 0
        self._epsilon_override: float | None = None
        self._last_eval_t = 0.0
        self._best_eval = float("-inf")
        self._amp = self.device.type == "cuda"
        self._stats = EpisodeStats()

        self._eval_env = BatchedSnakeEnv(
            num_envs=128,
            min_size=config.env.max_size,
            max_size=config.env.max_size,
            hunger_factor=config.env.hunger_factor,
            device=self.device,
            seed=(config.run.seed + 7) if config.run.seed is not None else None,
        )

    def policy_net(self) -> SnakeNet:
        return self.online  # type: ignore[return-value]

    def epsilon(self) -> float:
        if self._epsilon_override is not None:
            return max(self._epsilon_override, self.config.dqn.epsilon_end)
        dqn = self.config.dqn
        if dqn.epsilon_decay_steps <= 0:
            return dqn.epsilon_end
        frac = min(1.0, self.env_steps / dqn.epsilon_decay_steps)
        return dqn.epsilon_start + frac * (dqn.epsilon_end - dqn.epsilon_start)

    def apply_live(self, patch: dict[str, float]) -> None:
        allowed = live_field_keys("dqn")
        for key, val in patch.items():
            if key not in allowed:
                continue
            if key == "dqn.epsilon_end":
                # Live epsilon: set floor / fixed override
                self._epsilon_override = float(val)
                self.config.dqn.epsilon_end = float(val)
                continue
            parts = key.split(".")
            obj: Any = self.config
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], type(getattr(obj, parts[-1]))(val))
        self.weights = reward_weights_tensor(self.config, self.device)
        for g in self.opt.param_groups:
            g["lr"] = self.config.dqn.lr

    def act(self, obs: Obs, greedy: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        self.online.eval()
        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self._amp),
        ):
            q, value = self.online(obs.grid, obs.scalars)
            q = q.float()
            value = value.float()
            probs = F.softmax(q, dim=-1)
            if greedy:
                actions = q.argmax(dim=-1)
            else:
                eps = self.epsilon()
                greedy_a = q.argmax(dim=-1)
                rand_a = torch.randint(0, 3, (obs.grid.shape[0],), device=self.device)
                mask = torch.rand(obs.grid.shape[0], device=self.device) < eps
                actions = torch.where(mask, rand_a, greedy_a)
        return actions, probs, value

    def train_iteration(self) -> dict[str, float]:
        t0 = time.perf_counter()
        self._stats.reset()
        dqn = self.config.dqn
        steps = dqn.steps_per_iteration
        n = dqn.num_envs
        loss_acc = 0.0
        q_acc = 0.0
        n_updates = 0
        steps_since_update = 0

        obs = self.env.observe()
        for _ in range(steps):
            actions, _, _ = self.act(obs, greedy=False)
            step = self.env.step(actions)
            next_obs = self.env.observe()
            ep_ret = step.ep_return_components @ self.weights
            self._stats.add(step.done, step.cause, step.ep_score, step.ep_length, step.ep_steps, ep_ret)
            self._push_transitions(obs, actions, step.components, next_obs, step.done)
            obs = next_obs
            self.env_steps += n
            steps_since_update += n

            # At most a few updates per env-step to keep wall-clock healthy
            updates_this = 0
            while (
                self.env_steps >= dqn.learning_starts
                and len(self.replay) >= dqn.batch_size
                and steps_since_update >= dqn.train_freq
                and updates_this < 4
            ):
                loss, qmean = self._update()
                loss_acc += loss
                q_acc += qmean
                n_updates += 1
                updates_this += 1
                steps_since_update -= dqn.train_freq

        self.iteration += 1
        elapsed = time.perf_counter() - t0
        collected = steps * n
        metrics: dict[str, float] = {
            "iter": float(self.iteration),
            "env_steps": float(self.env_steps),
            "time_s": elapsed,
            "sps": collected / max(elapsed, 1e-9),
            "lr": float(self.config.dqn.lr),
            "epsilon": self.epsilon(),
        }
        if n_updates > 0:
            metrics["loss_q"] = loss_acc / n_updates
            metrics["q_mean"] = q_acc / n_updates
        metrics.update(self._stats.as_metrics())

        now = time.perf_counter()
        if now - self._last_eval_t >= self.config.run.eval_every_s or self.iteration == 1:
            self._last_eval_t = now
            metrics.update(self._evaluate())

        return metrics

    def _push_transitions(
        self,
        obs: Obs,
        actions: Tensor,
        components: Tensor,
        next_obs: Obs,
        done: Tensor,
    ) -> None:
        """Append to per-env n-step buffers and flush completed n-step transitions."""
        ns = self._ns
        n = self.env.num_envs
        env_ix = torch.arange(n, device=self.device)

        pos = self._buf_len.long().clamp(max=ns - 1)
        self._buf_grid[env_ix, pos] = obs.grid
        self._buf_scal[env_ix, pos] = obs.scalars
        self._buf_act[env_ix, pos] = actions
        self._buf_comp[env_ix, pos] = components
        self._buf_len += 1

        # Terminals first (next_obs is post-reset — never bootstrap).
        if done.any():
            self._flush_terminal(next_obs, done)
            self._buf_len[done] = 0

        # Continuing envs with a full n-step window.
        cont = (~done) & (self._buf_len >= ns)
        if cont.any():
            self._flush_nstep(next_obs, cont)

    def _flush_nstep(self, next_obs: Obs, mask: Tensor) -> None:
        ns = self._ns
        gamma = self.config.dqn.gamma
        full = mask.nonzero(as_tuple=False).view(-1)
        comps = self._buf_comp[full, :ns]
        disc = gamma ** torch.arange(ns, device=self.device, dtype=torch.float32)
        comp_sum = (comps * disc[None, :, None]).sum(dim=1)
        discount = torch.full((full.shape[0],), gamma**ns, device=self.device)
        self.replay.add_batch(
            obs_grid=self._buf_grid[full, 0],
            obs_scalars=self._buf_scal[full, 0],
            actions=self._buf_act[full, 0],
            comp_sum=comp_sum,
            discount=discount,
            next_grid=next_obs.grid[full],
            next_scalars=next_obs.scalars[full],
            done=torch.zeros(full.shape[0], dtype=torch.bool, device=self.device),
        )
        if ns > 1:
            self._buf_grid[full, :-1] = self._buf_grid[full, 1:].clone()
            self._buf_scal[full, :-1] = self._buf_scal[full, 1:].clone()
            self._buf_act[full, :-1] = self._buf_act[full, 1:].clone()
            self._buf_comp[full, :-1] = self._buf_comp[full, 1:].clone()
        self._buf_len[full] = ns - 1

    def _flush_terminal(self, next_obs: Obs, done: Tensor) -> None:
        """Emit one truncated n-step return from buffer start for each terminal env."""
        gamma = self.config.dqn.gamma
        ns = self._ns
        env_ix = done.nonzero(as_tuple=False).view(-1)
        lengths = self._buf_len[env_ix].long().clamp(min=1, max=ns)
        b = env_ix.shape[0]
        comps = self._buf_comp[env_ix, :ns]
        steps = torch.arange(ns, device=self.device)[None, :] < lengths[:, None]
        disc = (gamma ** torch.arange(ns, device=self.device, dtype=torch.float32))[None, :]
        comp_sum = (comps * disc[:, :, None] * steps.unsqueeze(-1).float()).sum(dim=1)
        self.replay.add_batch(
            obs_grid=self._buf_grid[env_ix, 0],
            obs_scalars=self._buf_scal[env_ix, 0],
            actions=self._buf_act[env_ix, 0],
            comp_sum=comp_sum,
            discount=torch.zeros(b, device=self.device),
            next_grid=next_obs.grid[env_ix],
            next_scalars=next_obs.scalars[env_ix],
            done=torch.ones(b, dtype=torch.bool, device=self.device),
        )

    def _update(self) -> tuple[float, float]:
        dqn = self.config.dqn
        batch = self.replay.sample(dqn.batch_size)
        reward = batch["comp_sum"] @ self.weights  # live weights

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=False):
            q_all, _ = self.online(batch["obs_grid"], batch["obs_scalars"])
            q_all = q_all.float()
            q_sa = q_all.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_online, _ = self.online(batch["next_grid"], batch["next_scalars"])
                next_q_online = next_q_online.float()
                next_act = next_q_online.argmax(dim=1)
                next_q_target, _ = self.target(batch["next_grid"], batch["next_scalars"])
                next_q_target = next_q_target.float()
                next_q = next_q_target.gather(1, next_act.unsqueeze(1)).squeeze(1)
                target = reward + batch["discount"] * next_q

            loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        # Soft update
        with torch.no_grad():
            tau = dqn.tau
            for p, tp in zip(self.online.parameters(), self.target.parameters(), strict=True):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

        return float(loss.item()), float(q_sa.mean().item())

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        env = self._eval_env
        env.reset()
        scores: list[float] = []
        wins = 0
        finished = 0
        games = 128
        steps = 0
        while finished < games and steps < 5000:
            obs = env.observe()
            actions, _, _ = self.act(obs, greedy=True)
            step = env.step(actions)
            if step.done.any():
                idx = step.done.nonzero(as_tuple=False).view(-1)
                for i in idx.tolist():
                    if finished >= games:
                        break
                    scores.append(float(step.ep_score[i].item()))
                    if int(step.cause[i].item()) == 4:
                        wins += 1
                    finished += 1
            steps += 1
        if not scores:
            return {}
        mean = sum(scores) / len(scores)
        if mean > self._best_eval:
            self._best_eval = mean
        return {
            "eval_score_mean": mean,
            "eval_score_max": float(max(scores)),
            "eval_win_rate": wins / len(scores),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "env_steps": self.env_steps,
            "iteration": self.iteration,
            "best_eval": self._best_eval,
            "epsilon_override": self._epsilon_override,
            "config": self.config.model_dump(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.online.load_state_dict(d["online"])
        self.target.load_state_dict(d.get("target", d["online"]))
        if "opt" in d:
            self.opt.load_state_dict(d["opt"])
        self.env_steps = int(d.get("env_steps", 0))
        self.iteration = int(d.get("iteration", 0))
        self._best_eval = float(d.get("best_eval", float("-inf")))
        self._epsilon_override = d.get("epsilon_override")
        if "config" in d:
            self.config = ExperimentConfig.model_validate(d["config"])
            self.weights = reward_weights_tensor(self.config, self.device)
