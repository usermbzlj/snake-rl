"""Live game viewer streaming latest weights (server process, CPU)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import torch

from snake_rl.core.env import BatchedSnakeEnv
from snake_rl.core.network import SnakeNet

if TYPE_CHECKING:
    from snake_rl.lab.manager import ExperimentManager

log = logging.getLogger(__name__)

SendJson = Callable[[dict[str, Any]], Awaitable[None]]


def _apply_relative(direction: int, action: int) -> int:
    if action == 1:
        return (direction - 1) % 4
    if action == 2:
        return (direction + 1) % 4
    return direction


class WatchSession:
    """One WebSocket watch connection."""

    def __init__(self, manager: ExperimentManager, exp_id: str, *, send_json: SendJson) -> None:
        self.manager = manager
        self.exp_id = exp_id
        self.send_json = send_json
        self.games = 4
        self.board_size = 8
        self.speed = 10.0
        self.greedy = True
        self._env: BatchedSnakeEnv | None = None
        self._net: SnakeNet | None = None
        self._algo = "ppo"
        self._width = 1.0
        self._hunger = 1.0
        self._model_version = -1
        self._env_steps = 0
        # idx -> (until_time, display_snap, cause, probs, value, action)
        self._holding: dict[int, tuple[float, dict[str, Any], int, list[float], float, int]] = {}
        self._running = False
        self._cfg_event = asyncio.Event()
        self._info_sent = False

    def apply_config(self, msg: dict[str, Any]) -> None:
        games = int(msg.get("games", self.games))
        if games not in (1, 4, 9):
            games = 4
        board_size = max(5, min(32, int(msg.get("board_size", self.board_size))))
        speed = max(1.0, min(60.0, float(msg.get("speed", self.speed))))
        greedy = bool(msg.get("greedy", self.greedy))
        rebuild = games != self.games or board_size != self.board_size
        self.games = games
        self.board_size = board_size
        self.speed = speed
        self.greedy = greedy
        if rebuild:
            self._env = None
            self._holding.clear()
        self._cfg_event.set()

    async def run(self) -> None:
        self._running = True
        try:
            meta = self.manager.store.read_meta(self.exp_id)
            self._algo = meta.get("algo", "ppo")
            cfg = meta.get("config") or {}
            self._width = float((cfg.get("model") or {}).get("width", 1.0))
            env_cfg = cfg.get("env") or {}
            self.board_size = int(env_cfg.get("max_size", self.board_size))
            self._hunger = float(env_cfg.get("hunger_factor", 1.0))
        except Exception:
            log.warning("WatchSession: failed reading meta for %s", self.exp_id, exc_info=True)

        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.exception("WatchSession tick failed for %s", self.exp_id)
                await self.send_json({"type": "info", "message": f"观看出错: {exc}"})
                await asyncio.sleep(0.5)
                continue
            delay = 1.0 / max(self.speed, 1.0)
            try:
                await asyncio.wait_for(self._cfg_event.wait(), timeout=delay)
                self._cfg_event.clear()
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._running = False
        self._cfg_event.set()

    async def _tick(self) -> None:
        weights = self.manager.latest_weights(self.exp_id)
        if weights is None:
            if not self._info_sent:
                await self.send_json({"type": "info", "message": "尚无可用权重，等待训练开始…"})
                self._info_sent = True
            return
        self._info_sent = False
        version, state_dict, env_steps = weights
        self._env_steps = env_steps

        if self._env is None:
            self._env = BatchedSnakeEnv(
                num_envs=self.games,
                min_size=self.board_size,
                max_size=self.board_size,
                hunger_factor=self._hunger,
                device=torch.device("cpu"),
            )
            self._env.reset()
            self._holding.clear()

        if self._net is None or version != self._model_version:
            await asyncio.to_thread(self._reload_net, state_dict)
            self._model_version = version

        frame = await asyncio.to_thread(self._step_frame)
        await self.send_json(frame)

    def _reload_net(self, state_dict: dict[str, Any]) -> None:
        net = SnakeNet(mode=self._algo, width=self._width)  # type: ignore[arg-type]
        net.load_state_dict(state_dict)
        net.eval()
        self._net = net

    def _fresh_start_snap(self) -> dict[str, Any]:
        s = self.board_size
        mid = s // 2
        body = [[mid, mid], [mid, mid - 1], [mid, mid - 2]]
        occupied = {(r, c) for r, c in body}
        food = [0, 0]
        for r in range(s):
            for c in range(s):
                if (r, c) not in occupied:
                    food = [r, c]
                    break
            else:
                continue
            break
        return {
            "size": s,
            "body": body,
            "food": food,
            "dir": 1,
            "score": 0,
            "steps": 0,
            "steps_since_food": 0,
            "length": 3,
        }

    def _step_frame(self) -> dict[str, Any]:
        assert self._env is not None and self._net is not None
        env = self._env
        net = self._net
        n = self.games
        now = time.perf_counter()

        # Release holds whose pause elapsed → restart
        for i in list(self._holding):
            until = self._holding[i][0]
            if now >= until:
                env.load_snapshot(i, self._fresh_start_snap())
                del self._holding[i]

        # Snapshot before step (for death final frames)
        pre = {i: env.snapshot(i) for i in range(n) if i not in self._holding}

        obs = env.observe()
        with torch.no_grad():
            logits_or_q, value = net(obs.grid, obs.scalars)
            if self._algo == "ppo":
                probs = torch.softmax(logits_or_q.float(), dim=-1)
                actions = probs.argmax(dim=-1) if self.greedy else torch.multinomial(probs, 1).squeeze(-1)
            else:
                q = logits_or_q.float()
                probs = torch.softmax(q, dim=-1)
                actions = q.argmax(dim=-1) if self.greedy else torch.multinomial(probs, 1).squeeze(-1)
            value = value.float()

        # Restore holding envs after observe mutated nothing; freeze them across step
        hold_snaps = {i: self._holding[i][1] for i in self._holding}
        for i, snap in hold_snaps.items():
            env.load_snapshot(i, snap)

        act = actions.clone()
        step = env.step(act)

        # Restore holds again (step may have moved them)
        for i, snap in hold_snaps.items():
            env.load_snapshot(i, snap)

        games_out: list[dict[str, Any]] = []
        for i in range(n):
            p = [float(x) for x in probs[i].tolist()]
            v = float(value[i].item())
            a = int(actions[i].item())

            if i in self._holding:
                _, snap, cause, hp, hv, ha = self._holding[i]
                games_out.append(
                    {
                        "snake": snap["body"],
                        "food": snap["food"],
                        "dir": snap["dir"],
                        "score": snap["score"],
                        "steps": snap["steps"],
                        "probs": hp,
                        "value": hv,
                        "action": ha,
                        "dead": True,
                        "cause": cause,
                    }
                )
                continue

            done = bool(step.done[i].item())
            cause = int(step.cause[i].item())
            if done:
                # Build final frame from pre-step + action (env already auto-reset)
                final = _death_frame(pre[i], a, cause, int(step.ep_score[i].item()))
                self._holding[i] = (now + 0.6, final, cause, p, v, a)
                # Keep env at a fresh start but hold display
                env.load_snapshot(i, self._fresh_start_snap())
                games_out.append(
                    {
                        "snake": final["body"],
                        "food": final["food"],
                        "dir": final["dir"],
                        "score": final["score"],
                        "steps": final["steps"],
                        "probs": p,
                        "value": v,
                        "action": a,
                        "dead": True,
                        "cause": cause,
                    }
                )
            else:
                snap = env.snapshot(i)
                games_out.append(
                    {
                        "snake": snap["body"],
                        "food": snap["food"],
                        "dir": snap["dir"],
                        "score": snap["score"],
                        "steps": snap["steps"],
                        "probs": p,
                        "value": v,
                        "action": a,
                        "dead": False,
                        "cause": 0,
                    }
                )

        return {
            "type": "frame",
            "model_version": self._model_version,
            "env_steps": self._env_steps,
            "games": games_out,
            "board_size": self.board_size,
        }


def _death_frame(pre: dict[str, Any], action: int, cause: int, score: int) -> dict[str, Any]:
    """Approximate the fatal board state from the pre-step snapshot."""
    body = [list(c) for c in pre["body"]]
    direction = _apply_relative(int(pre["dir"]), action)
    dr = [-1, 0, 1, 0][direction]
    dc = [0, 1, 0, -1][direction]
    hr, hc = body[0][0] + dr, body[0][1] + dc
    # Show head at collision cell when wall/self; keep food
    if cause in (1, 2):
        body = [[hr, hc], *body]
    elif cause == 4:
        pass
    else:
        # starve / other: advance one step if legal-ish
        body = [[hr, hc], *body[:-1]] if body else body
    return {
        "size": pre["size"],
        "body": body,
        "food": list(pre["food"]),
        "dir": direction,
        "score": score,
        "steps": int(pre["steps"]) + 1,
        "steps_since_food": int(pre.get("steps_since_food", 0)) + 1,
        "length": len(body),
    }
