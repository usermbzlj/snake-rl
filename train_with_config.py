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
    parser.add_argument("--parallel", action="store_true", help="启用多进程并行采样")
    parser.add_argument("--parallel-workers", type=int, default=None, help="并行 worker 数")
    parser.add_argument("--parallel-queue-capacity", type=int, default=None, help="并行队列容量")
    parser.add_argument("--parallel-sync-interval", type=int, default=None, help="策略同步间隔（步）")
    parser.add_argument("--parallel-actor-sleep-ms", type=int, default=None, help="actor 循环 sleep 毫秒")
    parser.add_argument("--parallel-actor-seed-stride", type=int, default=None, help="actor seed 跨度")
    parser.add_argument("--parallel-actor-device", type=str, default=None, help="actor 设备（默认 cpu）")
    args = parser.parse_args()

    if args.scheme:
        os.environ["SNAKE_TRAIN_SCHEME"] = args.scheme

    from train_config import get_config
    from snake_rl.train import run_training

    cfg = get_config()
    if args.parallel:
        cfg.parallel.enabled = True
    if args.parallel_workers is not None:
        cfg.parallel.num_workers = max(1, int(args.parallel_workers))
    if args.parallel_queue_capacity is not None:
        cfg.parallel.queue_capacity = max(128, int(args.parallel_queue_capacity))
    if args.parallel_sync_interval is not None:
        cfg.parallel.weight_sync_interval_steps = max(1, int(args.parallel_sync_interval))
    if args.parallel_actor_sleep_ms is not None:
        cfg.parallel.actor_loop_sleep_ms = max(0, int(args.parallel_actor_sleep_ms))
    if args.parallel_actor_seed_stride is not None:
        cfg.parallel.actor_seed_stride = max(1, int(args.parallel_actor_seed_stride))
    if args.parallel_actor_device is not None:
        cfg.parallel.actor_device = str(args.parallel_actor_device)
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
    if cfg.parallel.enabled:
        print(
            "  并行采样：开启 "
            f"(workers={cfg.parallel.num_workers}, queue={cfg.parallel.queue_capacity}, "
            f"sync={cfg.parallel.weight_sync_interval_steps} steps, actor_device={cfg.parallel.actor_device})"
        )
    else:
        print("  并行采样：关闭（串行训练）")
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
