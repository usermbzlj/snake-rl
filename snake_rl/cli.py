"""统一 CLI：train / eval / estimate / monitor / serve-model。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .estimate_time import build_estimate_arg_parser, run_estimate
from .evaluate import build_eval_arg_parser, run_eval
from .inference_server import build_inference_arg_parser, serve_inference_http
from .monitor_server import build_monitor_arg_parser, run_web_monitor


def _configure_console_encoding() -> None:
    """Best-effort console UTF-8 setup for Windows terminals."""
    if os.name != "nt":
        return

    stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    stderr_is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    if not (stdout_is_tty or stderr_is_tty):
        return

    # Keep this best-effort to avoid breaking non-console environments.
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

    for stream, is_tty in ((sys.stdout, stdout_is_tty), (sys.stderr, stderr_is_tty)):
        if not is_tty:
            continue
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _print_train_banner(cfg, *, scheme: str = "", custom_config_path: str = "") -> None:
    total_episodes = cfg.episodes
    if cfg.curriculum is not None:
        total_episodes = sum(stage.episodes for stage in cfg.curriculum.stages)

    print("=== 训练配置 ===")
    if scheme == "custom" and custom_config_path:
        print(f"  方案：custom（配置文件: {custom_config_path}）")
    elif scheme:
        print(f"  方案：{scheme}")
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
        food_distance_k = cfg.reward_weights.get("foodDistanceK", 0)
        if food_distance_k:
            print(f"  奖励塑形：foodDistanceK={food_distance_k}")

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


def cmd_train(args: argparse.Namespace) -> None:
    if args.resume_state is not None and args.warm_start is not None:
        print("错误：不能同时指定 --resume-state 与 --warm-start", file=sys.stderr)
        raise SystemExit(2)

    from .schemes import ACTIVE_SCHEME, get_config
    from .train import run_training, validate_config

    # 在修改环境变量之前先确定 effective_scheme，避免残留污染
    effective_scheme = args.scheme or os.environ.get("SNAKE_TRAIN_SCHEME", ACTIVE_SCHEME)

    # --custom-config 与非 custom scheme 组合：给出明确警告而非静默忽略
    if args.custom_config and effective_scheme != "custom":
        print(
            f"警告：--custom-config 仅在 --scheme custom 时有效，"
            f"当前方案 {effective_scheme!r} 将忽略此参数。",
            file=sys.stderr,
        )

    # 始终将 env var 更新为已解析的方案（防止上一次运行残留）
    os.environ["SNAKE_TRAIN_SCHEME"] = effective_scheme

    custom_path = args.custom_config if effective_scheme == "custom" else None
    try:
        cfg = get_config(scheme=effective_scheme, custom_config_path=custom_path)
    except FileNotFoundError as exc:
        print(f"错误：自定义配置文件不存在 —— {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"错误：自定义配置解析/校验失败 —— {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    # 对所有方案（含 custom）统一做配置校验，尽早报错而非等训练启动后才失败
    try:
        validate_config(cfg)
    except (ValueError, TypeError) as exc:
        print(f"错误：配置校验失败 —— {exc}", file=sys.stderr)
        raise SystemExit(2) from None

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

    _print_train_banner(cfg, scheme=effective_scheme, custom_config_path=str(custom_path) if custom_path else "")
    summary = run_training(
        cfg,
        resume_state=args.resume_state,
        warm_start=args.warm_start,
        extra_episodes=args.extra_episodes,
        warm_start_global_step=getattr(args, "warm_start_global_step", None),
    )
    print("训练完成。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()

    run_dir = Path(summary.get("run_dir", ""))
    if run_dir:
        best_ckpt = run_dir / "checkpoints" / "best.pt"
        latest_ckpt = run_dir / "checkpoints" / "latest.pt"
        print("后续建议：")
        print(f"  评估 best：snake-rl eval --checkpoint \"{best_ckpt}\"")
        print(f"  评估 latest：snake-rl eval --checkpoint \"{latest_ckpt}\"")


def cmd_eval(args: argparse.Namespace) -> None:
    result = run_eval(args)
    print("\nEvaluation summary:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def cmd_estimate(args: argparse.Namespace) -> None:
    run_estimate(args)


def cmd_monitor(args: argparse.Namespace) -> None:
    run_web_monitor(args)


def cmd_serve_model(args: argparse.Namespace) -> None:
    serve_inference_http(args)


def _build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="按方案启动训练（与 GUI 默认一致）。", add_help=False)
    p.add_argument("--scheme", default=None, help="训练方案 (custom/scheme1/2/3/4)，默认 custom")
    p.add_argument(
        "--custom-config",
        type=Path,
        default=None,
        help=(
            "scheme=custom 时加载的 TrainConfig JSON 路径（与 run_config.json 结构一致）。"
            "若未指定，自动使用项目根目录的 custom_train_config.json"
        ),
    )
    p.add_argument("--parallel", action="store_true", help="启用多进程并行采样")
    p.add_argument("--parallel-workers", type=int, default=None, help="并行 worker 数")
    p.add_argument("--parallel-queue-capacity", type=int, default=None, help="并行队列容量")
    p.add_argument("--parallel-sync-interval", type=int, default=None, help="策略同步间隔（步）")
    p.add_argument("--parallel-actor-sleep-ms", type=int, default=None, help="actor 循环 sleep 毫秒")
    p.add_argument("--parallel-actor-seed-stride", type=int, default=None, help="actor seed 跨度")
    p.add_argument("--parallel-actor-device", type=str, default=None, help="actor 设备（默认 cpu）")
    p.add_argument(
        "--resume-state",
        type=Path,
        default=None,
        help="从 state/training.pt 完整恢复训练",
    )
    p.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="仅从 checkpoint 加载权重，清空回放",
    )
    p.add_argument(
        "--extra-episodes",
        type=int,
        default=None,
        help="与 --resume-state 配合：在已完成基础上追加训练的局数",
    )
    p.add_argument(
        "--warm-start-global-step",
        type=int,
        default=None,
        metavar="N",
        help="与 --warm-start 配合：初始 global_step；省略则从源 checkpoint/run 推断；0=不继承",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    _configure_console_encoding()
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="snake-rl", description="Snake RL 统一命令行入口。")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser(
        "train",
        parents=[_build_train_parser()],
        help="按 snake_rl.schemes 中的方案训练",
    )
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser(
        "eval",
        parents=[build_eval_arg_parser()],
        help="评估 checkpoint",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_est = sub.add_parser(
        "estimate",
        parents=[build_estimate_arg_parser()],
        help="估算当前方案训练耗时",
    )
    p_est.set_defaults(func=cmd_estimate)

    p_mon = sub.add_parser(
        "monitor",
        parents=[build_monitor_arg_parser()],
        help="启动轻量 Web 训练监控后台（默认读取 runs）",
    )
    p_mon.set_defaults(func=cmd_monitor)

    p_srv = sub.add_parser(
        "serve-model",
        parents=[build_inference_arg_parser()],
        help="启动浏览器可用的模型推理 HTTP 服务",
    )
    p_srv.set_defaults(func=cmd_serve_model)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
