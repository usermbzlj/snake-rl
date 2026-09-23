from __future__ import annotations

from pathlib import Path

from snake_rl.config import CurriculumConfig, CurriculumStage, EnvPreset, TrainConfig
from snake_rl.train import run_training


def _tiny_cfg(tmp_path: Path, **kwargs) -> TrainConfig:
    base = dict(
        episodes=2,
        max_steps_per_episode=16,
        model_type="tiny",
        n_step=1,
        per_enabled=False,
        dueling=True,
        noisy=False,
        eval_episodes=0,
        eval_interval=0,
        replay_capacity=64,
        min_replay_size=8,
        batch_size=8,
        train_frequency=1,
        checkpoint_interval=2,
        log_interval=1,
        tensorboard=False,
        save_csv=False,
        save_jsonl=True,
        live_plot=False,
        output_root=tmp_path,
        run_name="smoke",
        env=EnvPreset(board_size=8, seed=1, max_steps_without_food=16),
    )
    base.update(kwargs)
    return TrainConfig(**base)


def test_standard_training_writes_state(tmp_path: Path) -> None:
    summary = run_training(_tiny_cfg(tmp_path))
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "state" / "training.pt").is_file()
    assert (run_dir / "checkpoints" / "latest.pt").is_file()
    assert (run_dir / "run_config.json").is_file()
    assert not (run_dir / "train_config.json").exists()


def test_curriculum_resume_continues(tmp_path: Path) -> None:
    cfg = _tiny_cfg(
        tmp_path,
        run_name="curr",
        episodes=1,
        curriculum=CurriculumConfig(
            stages=[
                CurriculumStage(
                    board_size=8,
                    episodes=2,
                    replay_capacity=64,
                    min_replay_size=8,
                    epsilon_decay_steps=50,
                )
            ]
        ),
    )
    first = run_training(cfg)
    state_path = Path(first["run_dir"]) / "state" / "training.pt"
    assert state_path.is_file()
    second = run_training(
        _tiny_cfg(tmp_path, run_name="curr"),
        resume_state=state_path,
        extra_episodes=1,
    )
    assert second.get("resumed_from_episode") is not None
    assert Path(second["run_dir"]) == Path(first["run_dir"])
