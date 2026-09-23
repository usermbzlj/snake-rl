"""Experiment REST endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import ValidationError

from snake_rl.lab.inspect import compare as compare_episodes
from snake_rl.lab.inspect import inspect_episode
from snake_rl.server.schemas import (
    CloneBody,
    CompareBody,
    CreateBody,
    InspectBody,
    LiveBody,
    get_manager,
    require_experiment,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["experiments"])


@router.get("/api/experiments")
def api_list(request: Request) -> list[dict[str, Any]]:
    return get_manager(request).list()


@router.post("/api/experiments")
def api_create(request: Request, body: CreateBody) -> dict[str, Any]:
    try:
        return get_manager(request).create(body.config, start=body.start)
    except ValidationError as e:
        raise HTTPException(400, f"配置无效: {e}") from e
    except Exception as e:
        log.exception("create experiment failed")
        raise HTTPException(400, f"创建实验失败: {e}") from e


@router.get("/api/experiments/{exp_id}")
def api_get(request: Request, exp_id: str) -> dict[str, Any]:
    mgr = get_manager(request)
    require_experiment(mgr, exp_id)
    return mgr.get(exp_id)


@router.patch("/api/experiments/{exp_id}/live")
def api_live(request: Request, exp_id: str, body: LiveBody) -> dict[str, Any]:
    mgr = get_manager(request)
    require_experiment(mgr, exp_id)
    try:
        return mgr.live_patch(exp_id, body.patch)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        log.exception("live_patch failed for %s", exp_id)
        raise HTTPException(400, f"实时调参失败: {e}") from e


@router.post("/api/experiments/{exp_id}/clone")
def api_clone(request: Request, exp_id: str, body: CloneBody) -> dict[str, Any]:
    mgr = get_manager(request)
    require_experiment(mgr, exp_id)
    try:
        return mgr.clone(exp_id, name=body.name, with_weights=body.with_weights)
    except Exception as e:
        log.exception("clone failed for %s", exp_id)
        raise HTTPException(400, f"克隆失败: {e}") from e


@router.delete("/api/experiments/{exp_id}")
def api_delete(request: Request, exp_id: str) -> Response:
    mgr = get_manager(request)
    require_experiment(mgr, exp_id)
    try:
        mgr.delete(exp_id)
    except Exception as e:
        log.exception("delete failed for %s", exp_id)
        raise HTTPException(400, f"删除失败: {e}") from e
    return Response(status_code=204)


@router.get("/api/experiments/{exp_id}/checkpoints")
def api_ckpts(request: Request, exp_id: str) -> list[dict[str, Any]]:
    mgr = get_manager(request)
    require_experiment(mgr, exp_id)
    return mgr.store.checkpoint_infos(exp_id)


@router.post("/api/experiments/{exp_id}/inspect")
async def api_inspect(request: Request, exp_id: str, body: InspectBody) -> dict[str, Any]:
    mgr = get_manager(request)
    require_experiment(mgr, exp_id)
    path = mgr.store.ckpt_path(exp_id, body.checkpoint)
    if not path.is_file():
        raise HTTPException(400, f"检查点不存在: {body.checkpoint}")
    try:
        traj = await asyncio.to_thread(
            inspect_episode,
            path,
            body.board_size,
            body.seed,
            body.greedy,
            5000,
            True,
        )
        traj["experiment_id"] = exp_id
        meta = mgr.store.read_meta(exp_id)
        traj["name"] = meta.get("name")
        return traj
    except Exception as e:
        log.exception("inspect failed for %s", exp_id)
        raise HTTPException(400, f"分析失败: {e}") from e


@router.post("/api/experiments/{exp_id}/{action}")
def api_action(request: Request, exp_id: str, action: str) -> dict[str, Any]:
    mgr = get_manager(request)
    require_experiment(mgr, exp_id)
    if action not in ("start", "pause", "resume", "stop"):
        raise HTTPException(404, f"未知操作: {action}")
    try:
        if action == "start":
            return mgr.start(exp_id)
        if action == "pause":
            return mgr.pause(exp_id)
        if action == "resume":
            return mgr.resume(exp_id)
        return mgr.stop(exp_id)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        log.exception("action %s failed for %s", action, exp_id)
        raise HTTPException(400, f"操作失败: {e}") from e


@router.post("/api/compare")
async def api_compare(request: Request, body: CompareBody) -> dict[str, Any]:
    mgr = get_manager(request)
    if len(body.entries) < 2 or len(body.entries) > 4:
        raise HTTPException(400, "请选择 2–4 个实验进行对比")
    pairs: list[tuple[str, str]] = []
    labels: list[tuple[str, str]] = []
    for ent in body.entries:
        if not mgr.store.exists(ent.experiment_id):
            raise HTTPException(404, f"实验不存在: {ent.experiment_id}")
        path = mgr.store.ckpt_path(ent.experiment_id, ent.checkpoint)
        if not path.is_file():
            raise HTTPException(400, f"检查点不存在: {ent.experiment_id}/{ent.checkpoint}")
        meta = mgr.store.read_meta(ent.experiment_id)
        pairs.append((str(path), meta.get("name", ent.experiment_id)))
        labels.append((ent.experiment_id, meta.get("name", ent.experiment_id)))
    seed = body.seed if body.seed is not None else 0
    try:
        trajs = await asyncio.to_thread(compare_episodes, pairs, body.board_size, seed)
        for traj, (eid, name) in zip(trajs, labels, strict=True):
            traj["experiment_id"] = eid
            traj["name"] = name
        return {"seed": seed, "trajectories": trajs}
    except Exception as e:
        log.exception("compare failed")
        raise HTTPException(400, f"对比失败: {e}") from e
