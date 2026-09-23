"""Shared request models and helpers for API routes."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from snake_rl.core.config import ExperimentConfig
from snake_rl.lab.manager import ExperimentManager


class CreateBody(BaseModel):
    config: ExperimentConfig
    start: bool = True


class LiveBody(BaseModel):
    patch: dict[str, float] = Field(default_factory=dict)


class CloneBody(BaseModel):
    name: str
    with_weights: bool = False


class InspectBody(BaseModel):
    checkpoint: Literal["latest", "best"] = "latest"
    board_size: int = 8
    seed: int | None = None
    greedy: bool = True


class CompareEntry(BaseModel):
    experiment_id: str
    checkpoint: Literal["latest", "best"] = "latest"


class CompareBody(BaseModel):
    entries: list[CompareEntry]
    board_size: int = 8
    seed: int | None = None


def get_manager(request: Request) -> ExperimentManager:
    return request.app.state.manager


def get_manager_from_ws(ws: Any) -> ExperimentManager:
    return ws.app.state.manager


def require_experiment(mgr: ExperimentManager, exp_id: str) -> None:
    if not mgr.store.exists(exp_id):
        raise HTTPException(404, f"实验不存在: {exp_id}")
