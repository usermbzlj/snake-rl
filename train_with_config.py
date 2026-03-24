"""
使用 train_config.py 中的参数启动训练。

运行方式：
    uv run python train_with_config.py
    uv run python train_with_config.py --scheme scheme4
"""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="使用配置文件启动训练")
    parser.add_argument("--scheme", default=None, help="覆盖训练方案 (scheme1/2/3/4)")
    args = parser.parse_args()

    if args.scheme:
        os.environ["SNAKE_TRAIN_SCHEME"] = args.scheme

    from train_config import get_config
    from snake_rl.train import run_training

    cfg = get_config()
    total_episodes = cfg.episodes
    if cfg.curriculum is not None:
        total_episodes = sum(stage.episodes for stage in cfg.curriculum.stages)

    print("=== 训练配置 ===")
    print(f"  模型：{cfg.model_type}")
    if cfg.model_type == "hybrid":
        print(f"  局部 patch：{cfg.local_patch_size}x{cfg.local_patch_size}")

    if cfg.curriculum is not None:
        def stage_desc(stage) -> str:
            if stage.board_sizes:
                return "[" + ", ".join(str(size) for size in stage.board_sizes) + "]"
            return str(stage.board_size)

        stage_text = " -> ".join(stage_desc(stage) for stage in cfg.curriculum.stages)
        print(f"  模式：课程学习（{stage_text}）")

        has_promotion = any(s.promotion_threshold_foods > 0 for s in cfg.curriculum.stages)
        if has_promotion:
            print("  晋升：基于表现门槛（平均食物数达标后自动晋升）")
        else:
            print("  晋升：固定局数")
    elif cfg.random_board is not None:
        board_text = ", ".join(str(size) for size in cfg.random_board.board_sizes)
        print(f"  模式：随机地图（{board_text}）")
    else:
        print(f"  环境：{cfg.env.difficulty} 难度 / {cfg.env.mode} 模式 / {cfg.env.board_size}×{cfg.env.board_size} 地图")

    if cfg.reward_weights:
        approach = cfg.reward_weights.get("approachFood", 0)
        retreat = cfg.reward_weights.get("retreatFood", 0)
        if approach or retreat:
            print(f"  奖励塑形：approachFood={approach}, retreatFood={retreat}")

    print(f"  总局数上限：{total_episodes}  batch_size：{cfg.batch_size}  lr：{cfg.learning_rate}")
    print(f"  ε: {cfg.epsilon_start} → {cfg.epsilon_end}（{cfg.epsilon_decay_steps} 步衰减）")
    print(f"  设备：{cfg.device}  输出目录：{cfg.output_root}/{cfg.run_name}")
    print("================\n")

    summary = run_training(cfg)
    print("训练完成。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()

    run_dir = Path(summary.get("run_dir", ""))
    if run_dir:
        best_ckpt = run_dir / "checkpoints" / "best.pt"
        latest_ckpt = run_dir / "checkpoints" / "latest.pt"
        print("后续建议：")
        print(f"  评估 best：uv run python eval_pytorch.py --checkpoint \"{best_ckpt}\"")
        print(f"  评估 latest：uv run python eval_pytorch.py --checkpoint \"{latest_ckpt}\"")


if __name__ == "__main__":
    main()
