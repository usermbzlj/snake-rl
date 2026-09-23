"""Proximal Policy Optimization trainer."""

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
from snake_rl.core.trainer import EpisodeStats, reward_weights_tensor, set_seed


class PPOTrainer:
    def __init__(self, config: ExperimentConfig, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        set_seed(config.run.seed)

        ppo = config.ppo
        self.env = BatchedSnakeEnv(
            num_envs=ppo.num_envs,
            min_size=config.env.min_size,
            max_size=config.env.max_size,
            hunger_factor=config.env.hunger_factor,
            device=self.device,
            seed=config.run.seed,
        )
        self.net = SnakeNet(mode="ppo", width=config.model.width).to(self.device)
        if config.run.compile and self.device.type == "cuda":
            with contextlib.suppress(Exception):
                self.net = torch.compile(self.net)  # type: ignore[assignment]
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo.lr, eps=1e-5)
        self.weights = reward_weights_tensor(config, self.device)

        self.env_steps = 0
        self.iteration = 0
        self._last_eval_t = 0.0
        self._best_eval = float("-inf")
        self._amp = self.device.type == "cuda"
        self._stats = EpisodeStats()

        # Eval env (greedy)
        self._eval_env = BatchedSnakeEnv(
            num_envs=128,
            min_size=config.env.max_size,
            max_size=config.env.max_size,
            hunger_factor=config.env.hunger_factor,
            device=self.device,
            seed=(config.run.seed + 1) if config.run.seed is not None else None,
        )

    def policy_net(self) -> SnakeNet:
        return self.net  # type: ignore[return-value]

    def apply_live(self, patch: dict[str, float]) -> None:
        allowed = live_field_keys("ppo")
        for key, val in patch.items():
            if key not in allowed:
                continue
            parts = key.split(".")
            obj: Any = self.config
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], type(getattr(obj, parts[-1]))(val))
        self.weights = reward_weights_tensor(self.config, self.device)
        for g in self.opt.param_groups:
            g["lr"] = self.config.ppo.lr

    def act(self, obs: Obs, greedy: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        self.net.eval()
        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self._amp),
        ):
            logits, value = self.net(obs.grid, obs.scalars)
            logits = logits.float()
            value = value.float()
            probs = F.softmax(logits, dim=-1)
            actions = probs.argmax(dim=-1) if greedy else torch.multinomial(probs, 1).squeeze(-1)
            return actions, probs, value

    def train_iteration(self) -> dict[str, float]:
        t0 = time.perf_counter()
        self._stats.reset()
        ppo = self.config.ppo
        n, t_len = ppo.num_envs, ppo.rollout

        obs = self.env.observe()
        grids = torch.empty(t_len, n, 4, obs.grid.shape[-1], obs.grid.shape[-1], device=self.device)
        scalars = torch.empty(t_len, n, 4, device=self.device)
        actions = torch.empty(t_len, n, dtype=torch.int64, device=self.device)
        logprobs = torch.empty(t_len, n, device=self.device)
        rewards = torch.empty(t_len, n, device=self.device)
        dones = torch.empty(t_len, n, dtype=torch.bool, device=self.device)
        values = torch.empty(t_len, n, device=self.device)
        comps = torch.empty(t_len, n, N_COMPONENTS, device=self.device)

        self.net.train()
        for t in range(t_len):
            grids[t] = obs.grid
            scalars[t] = obs.scalars
            with (
                torch.no_grad(),
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self._amp),
            ):
                logits, value = self.net(obs.grid, obs.scalars)
                logits = logits.float()
                value = value.float()
                dist_probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=dist_probs)
                act = dist.sample()
                lp = dist.log_prob(act)
            actions[t] = act
            logprobs[t] = lp
            values[t] = value

            step = self.env.step(act)
            r = step.components @ self.weights
            rewards[t] = r
            dones[t] = step.done
            comps[t] = step.components
            ep_ret = step.ep_return_components @ self.weights
            self._stats.add(step.done, step.cause, step.ep_score, step.ep_length, step.ep_steps, ep_ret)
            obs = self.env.observe()

        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self._amp),
        ):
            _, last_val = self.net(obs.grid, obs.scalars)
            last_val = last_val.float()

        advantages, returns = self._gae(rewards, values, dones, last_val, ppo.gamma, ppo.gae_lambda)

        # Flatten
        b_grid = grids.reshape(t_len * n, *grids.shape[2:])
        b_scal = scalars.reshape(t_len * n, 4)
        b_act = actions.reshape(t_len * n)
        b_logp = logprobs.reshape(t_len * n)
        b_adv = advantages.reshape(t_len * n)
        b_ret = returns.reshape(t_len * n)

        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = t_len * n
        mb_size = batch_size // ppo.minibatches
        idx = torch.arange(batch_size, device=self.device)

        loss_pi_acc = 0.0
        loss_v_acc = 0.0
        ent_acc = 0.0
        clipfrac_acc = 0.0
        kl_acc = 0.0
        updates = 0

        for _ in range(ppo.epochs):
            perm = idx[torch.randperm(batch_size, device=self.device)]
            for start in range(0, batch_size, mb_size):
                mb = perm[start : start + mb_size]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self._amp):
                    logits, value = self.net(b_grid[mb], b_scal[mb])
                    logits = logits.float()
                    value = value.float()
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(b_act[mb])
                    entropy = dist.entropy().mean()
                    ratio = (new_logp - b_logp[mb]).exp()
                    adv = b_adv[mb]
                    pg1 = ratio * adv
                    pg2 = torch.clamp(ratio, 1.0 - ppo.clip, 1.0 + ppo.clip) * adv
                    loss_pi = -torch.min(pg1, pg2).mean()
                    # value clip optional — use MSE to returns
                    loss_v = F.mse_loss(value, b_ret[mb])
                    loss = loss_pi + ppo.vf_coef * loss_v - ppo.ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), ppo.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    clipfrac_acc += float(((ratio - 1.0).abs() > ppo.clip).float().mean().item())
                    kl_acc += float((b_logp[mb] - new_logp).mean().item())
                loss_pi_acc += float(loss_pi.item())
                loss_v_acc += float(loss_v.item())
                ent_acc += float(entropy.item())
                updates += 1

        self.env_steps += t_len * n
        self.iteration += 1
        elapsed = time.perf_counter() - t0
        metrics = {
            "iter": float(self.iteration),
            "env_steps": float(self.env_steps),
            "time_s": elapsed,
            "sps": (t_len * n) / max(elapsed, 1e-9),
            "lr": float(self.config.ppo.lr),
            "entropy": ent_acc / max(updates, 1),
            "loss_policy": loss_pi_acc / max(updates, 1),
            "loss_value": loss_v_acc / max(updates, 1),
            "kl": kl_acc / max(updates, 1),
            "clipfrac": clipfrac_acc / max(updates, 1),
        }
        metrics.update(self._stats.as_metrics())

        now = time.perf_counter()
        if now - self._last_eval_t >= self.config.run.eval_every_s or self.iteration == 1:
            self._last_eval_t = now
            ev = self._evaluate()
            metrics.update(ev)

        return metrics

    def _gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        last_value: Tensor,
        gamma: float,
        lam: float,
    ) -> tuple[Tensor, Tensor]:
        t_len, n = rewards.shape
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros(n, device=self.device)
        next_val = last_value
        for t in reversed(range(t_len)):
            next_nonterminal = (~dones[t]).float()
            delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[t] = last_gae
            next_val = values[t]
        returns = adv + values
        return adv, returns

    @torch.no_grad()
    def _evaluate(self, games: int = 128) -> dict[str, float]:
        env = self._eval_env
        env.reset()
        scores: list[float] = []
        wins = 0
        finished = 0
        # Cap steps to avoid infinite loops
        max_steps = 5000
        steps = 0
        while finished < games and steps < max_steps:
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
        out = {
            "eval_score_mean": mean,
            "eval_score_max": float(max(scores)),
            "eval_win_rate": wins / len(scores),
        }
        if mean > self._best_eval:
            self._best_eval = mean
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "env_steps": self.env_steps,
            "iteration": self.iteration,
            "best_eval": self._best_eval,
            "config": self.config.model_dump(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.net.load_state_dict(d["net"])
        if "opt" in d:
            self.opt.load_state_dict(d["opt"])
        self.env_steps = int(d.get("env_steps", 0))
        self.iteration = int(d.get("iteration", 0))
        self._best_eval = float(d.get("best_eval", float("-inf")))
        if "config" in d:
            self.config = ExperimentConfig.model_validate(d["config"])
            self.weights = reward_weights_tensor(self.config, self.device)
