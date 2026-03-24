from __future__ import annotations

from collections import deque
from typing import Any


class LivePlotter:
    """Matplotlib-based live training dashboard."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.history: dict[str, list[float]] = {
            "episode": [],
            "reward": [],
            "avg_reward": [],
            "steps": [],
            "foods": [],
            "loss": [],
            "epsilon": [],
        }
        self._recent_rewards = deque(maxlen=100)
        self._plt = None
        self._fig = None
        self._axes = None
        self._lines: dict[str, Any] = {}
        if self.enabled:
            self._init_figure()

    def _init_figure(self) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Snake DDQN Training Monitor")
        self._fig = fig
        self._axes = axes

        (line_reward,) = axes[0][0].plot([], [], label="reward", color="tab:blue")
        (line_avg_reward,) = axes[0][0].plot([], [], label="avg_reward(100)", color="tab:orange")
        axes[0][0].set_title("Episode Reward")
        axes[0][0].set_xlabel("Episode")
        axes[0][0].legend(loc="best")

        (line_steps,) = axes[0][1].plot([], [], label="steps", color="tab:green")
        (line_foods,) = axes[0][1].plot([], [], label="foods", color="tab:red")
        axes[0][1].set_title("Episode Steps / Foods")
        axes[0][1].set_xlabel("Episode")
        axes[0][1].legend(loc="best")

        (line_loss,) = axes[1][0].plot([], [], label="loss", color="tab:purple")
        axes[1][0].set_title("Training Loss")
        axes[1][0].set_xlabel("Episode")
        axes[1][0].legend(loc="best")

        (line_epsilon,) = axes[1][1].plot([], [], label="epsilon", color="tab:brown")
        axes[1][1].set_title("Exploration Epsilon")
        axes[1][1].set_xlabel("Episode")
        axes[1][1].legend(loc="best")

        self._lines = {
            "reward": line_reward,
            "avg_reward": line_avg_reward,
            "steps": line_steps,
            "foods": line_foods,
            "loss": line_loss,
            "epsilon": line_epsilon,
        }

        fig.tight_layout()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def update(
        self,
        episode: int,
        reward: float,
        steps: int,
        foods: int,
        epsilon: float,
        loss: float | None,
    ) -> None:
        self.history["episode"].append(float(episode))
        self.history["reward"].append(float(reward))
        self.history["steps"].append(float(steps))
        self.history["foods"].append(float(foods))
        self.history["epsilon"].append(float(epsilon))
        self.history["loss"].append(float(loss) if loss is not None else float("nan"))
        self._recent_rewards.append(float(reward))
        avg_reward = sum(self._recent_rewards) / len(self._recent_rewards)
        self.history["avg_reward"].append(avg_reward)

        if not self.enabled or self._plt is None or self._axes is None:
            return

        episodes = self.history["episode"]
        self._lines["reward"].set_data(episodes, self.history["reward"])
        self._lines["avg_reward"].set_data(episodes, self.history["avg_reward"])
        self._lines["steps"].set_data(episodes, self.history["steps"])
        self._lines["foods"].set_data(episodes, self.history["foods"])
        self._lines["loss"].set_data(episodes, self.history["loss"])
        self._lines["epsilon"].set_data(episodes, self.history["epsilon"])

        for row in self._axes:
            for ax in row:
                ax.relim()
                ax.autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close(self) -> None:
        if self.enabled and self._plt is not None and self._fig is not None:
            self._plt.ioff()
            self._plt.close(self._fig)
