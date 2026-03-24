"""Snake RL training package."""

from .env import (
    ACTIONS,
    OBSERVATION_CHANNELS,
    TERMINAL_REASONS,
    SnakeEnv,
    SnakeEnvConfig,
)

__all__ = [
    "ACTIONS",
    "OBSERVATION_CHANNELS",
    "TERMINAL_REASONS",
    "SnakeEnv",
    "SnakeEnvConfig",
]
