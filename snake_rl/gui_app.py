"""
贪吃蛇 AI 训练管理器 —— 一体化 GUI（包内入口）。

启动：
    uv run snake-gui
    或 uv run python -m snake_rl.gui_app
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk, messagebox

from .process_supervisor import terminate_process
from .schemes import SCHEME_INFO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
RUNS_DIR = PROJECT_ROOT / "runs"


class TrainingManager:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("贪吃蛇 AI 训练管理器")
        self.root.geometry("960x820")
        self.root.minsize(800, 600)

        self.training_proc: subprocess.Popen | None = None
        self.tb_proc: subprocess.Popen | None = None
        self.inference_proc: subprocess.Popen | None = None
        self._user_requested_training_stop = False

        self._build_ui()
        self.refresh_runs()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("", 15, "bold"))
        style.configure("Status.TLabel", font=("", 10))

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="贪吃蛇 AI 训练管理器", style="Title.TLabel").pack(pady=(0, 8))

        self._build_training_section(main)
        self._build_runs_section(main)
        self._build_tools_section(main)
        self._build_log_section(main)

    def _build_training_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="训练控制", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="训练方案:").pack(side=tk.LEFT)
        self.scheme_var = tk.StringVar(value="scheme1")
        cb = ttk.Combobox(
            row1,
            textvariable=self.scheme_var,
            values=list(SCHEME_INFO.keys()),
            state="readonly",
            width=12,
        )
        cb.pack(side=tk.LEFT, padx=5)
        cb.bind("<<ComboboxSelected>>", self._on_scheme_changed)

        self.scheme_desc_var = tk.StringVar(value=SCHEME_INFO["scheme1"])
        ttk.Label(frame, textvariable=self.scheme_desc_var, wraplength=880).pack(
            fill=tk.X, pady=4
        )

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=(4, 0))
        self.start_btn = ttk.Button(row2, text="▶  开始训练", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.stop_btn = ttk.Button(
            row2, text="■  停止训练", command=self.stop_training, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 12))

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(row2, textvariable=self.status_var, style="Status.TLabel").pack(
            side=tk.LEFT
        )

        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=(6, 0))
        self.parallel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row3,
            text="启用并行采样",
            variable=self.parallel_var,
        ).pack(side=tk.LEFT)
        ttk.Label(row3, text="Workers:").pack(side=tk.LEFT, padx=(12, 4))
        self.parallel_workers_var = tk.IntVar(value=4)
        ttk.Spinbox(
            row3,
            from_=1,
            to=64,
            width=6,
            textvariable=self.parallel_workers_var,
        ).pack(side=tk.LEFT)
        ttk.Label(row3, text="同步步长:").pack(side=tk.LEFT, padx=(12, 4))
        self.parallel_sync_var = tk.IntVar(value=512)
        ttk.Spinbox(
            row3,
            from_=16,
            to=100000,
            increment=16,
            width=8,
            textvariable=self.parallel_sync_var,
        ).pack(side=tk.LEFT)

    def _build_runs_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="训练记录", padding=8)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        columns = ("name", "model", "episodes", "best_reward", "status")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=6)
        self.tree.heading("name", text="运行名称")
        self.tree.heading("model", text="模型")
        self.tree.heading("episodes", text="局数")
        self.tree.heading("best_reward", text="最佳平均奖励")
        self.tree.heading("status", text="状态")
        self.tree.column("name", width=280)
        self.tree.column("model", width=100)
        self.tree.column("episodes", width=80, anchor=tk.CENTER)
        self.tree.column("best_reward", width=120, anchor=tk.CENTER)
        self.tree.column("status", width=80, anchor=tk.CENTER)

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(btn_frame, text="刷新列表", command=self.refresh_runs).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="删除选中", command=self.delete_selected).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="清空所有记录", command=self.delete_all).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8
        )
        ttk.Button(btn_frame, text="用此模型演示", command=self.launch_demo).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="停止演示", command=self.stop_inference).pack(
            side=tk.LEFT, padx=2
        )

    def _build_tools_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="工具", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        row = ttk.Frame(frame)
        row.pack(fill=tk.X)
        ttk.Button(row, text="打开 TensorBoard", command=self.open_tensorboard).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(row, text="关闭 TensorBoard", command=self.stop_tensorboard).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(row, text="打开游戏", command=self.open_game).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(row, text="估算训练时间", command=self.estimate_time).pack(
            side=tk.LEFT, padx=2
        )

    def _build_log_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="输出日志", padding=4)
        frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            selectbackground="#264f78",
            font=("Consolas", 9),
        )
        log_sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_sb.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_scheme_changed(self, _event: Any = None) -> None:
        self.scheme_desc_var.set(SCHEME_INFO.get(self.scheme_var.get(), ""))

    def log(self, text: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _win_flags(self) -> int:
        return subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

    def _popen_training(self, args: list[str]) -> subprocess.Popen:
        return subprocess.Popen(
            args,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            creationflags=self._win_flags(),
        )

    def _popen_detach(self, args: list[str]) -> subprocess.Popen:
        return subprocess.Popen(
            args,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            creationflags=self._win_flags(),
        )

    def _selected_run_dir(self) -> Path | None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("提示", "请先在列表中选择一个训练记录")
            return None
        name = self.tree.item(sel[0], "values")[0]
        return RUNS_DIR / name

    def start_training(self) -> None:
        if self.training_proc is not None and self.training_proc.poll() is None:
            messagebox.showwarning("提示", "训练已在进行中")
            return

        scheme = self.scheme_var.get()
        use_parallel = bool(self.parallel_var.get())
        workers = max(1, int(self.parallel_workers_var.get()))
        sync_interval = max(1, int(self.parallel_sync_var.get()))
        cmd = [sys.executable, "-m", "snake_rl.cli", "train", "--scheme", scheme]
        if use_parallel:
            cmd.extend(
                [
                    "--parallel",
                    "--parallel-workers",
                    str(workers),
                    "--parallel-sync-interval",
                    str(sync_interval),
                ]
            )
        try:
            self._user_requested_training_stop = False
            self.training_proc = self._popen_training(cmd)
        except Exception as exc:
            messagebox.showerror("错误", f"启动训练失败:\n{exc}")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        mode_text = f"{scheme} | {'并行' if use_parallel else '串行'}"
        if use_parallel:
            mode_text += f" | workers={workers}"
        self.status_var.set(f"训练中 ({mode_text}) ...")
        self.log(f"[{self._ts()}] 开始训练: {mode_text}")

        threading.Thread(target=self._read_proc_output, args=(self.training_proc,), daemon=True).start()

    def _read_proc_output(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                stripped = line.rstrip("\n\r")
                if stripped:
                    self.root.after(0, self.log, stripped)
        except Exception:
            pass
        finally:
            code = proc.wait()
            self.root.after(0, self._on_training_done, code)

    def _on_training_done(self, code: int) -> None:
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self._user_requested_training_stop:
            msg = "训练已中断（用户停止）"
        elif code == 0:
            msg = "训练完成"
        else:
            msg = f"训练结束 (exit {code})"
        self._user_requested_training_stop = False
        self.status_var.set(msg)
        self.log(f"[{self._ts()}] {msg}")
        self.refresh_runs()

    def stop_training(self) -> None:
        if self.training_proc is None or self.training_proc.poll() is not None:
            return
        if not messagebox.askyesno("确认", "确定要停止训练吗？\n已保存的检查点不会丢失。"):
            return
        self._user_requested_training_stop = True
        terminate_process(self.training_proc)
        self.status_var.set("正在停止...")
        self.log(f"[{self._ts()}] 正在停止训练...")

    def refresh_runs(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        if not RUNS_DIR.exists():
            return

        for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            info = self._read_run_info(run_dir)
            self.tree.insert(
                "",
                tk.END,
                values=(
                    info["name"],
                    info["model"],
                    info["episodes"],
                    info["best_reward"],
                    info["status"],
                ),
            )

    @staticmethod
    def _read_run_info(run_dir: Path) -> dict[str, str]:
        info: dict[str, str] = {
            "name": run_dir.name,
            "model": "-",
            "episodes": "-",
            "best_reward": "-",
            "status": "未知",
        }
        for cfg_name in ("run_config.json", "train_config.json"):
            cfg_path = run_dir / cfg_name
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    info["model"] = str(cfg.get("model_type", "-"))
                except Exception:
                    pass
                break

        summary_path = run_dir / "logs" / "summary.json"
        has_checkpoint = (run_dir / "checkpoints" / "latest.pt").exists() or (
            run_dir / "state" / "training.pt"
        ).exists()

        if summary_path.exists():
            try:
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                info["episodes"] = str(s.get("episodes", "-"))
                bar = s.get("best_avg_reward")
                if bar is not None:
                    info["best_reward"] = f"{bar:.3f}"
                info["status"] = "完成"
            except Exception:
                info["status"] = "中断" if has_checkpoint else "空"
        elif has_checkpoint:
            info["status"] = "中断"
        else:
            info["status"] = "空"

        return info

    def delete_selected(self) -> None:
        run_dir = self._selected_run_dir()
        if run_dir is None:
            return
        if not messagebox.askyesno("确认", f"删除 {run_dir.name}？\n此操作不可恢复。"):
            return
        try:
            shutil.rmtree(run_dir)
            self.log(f"已删除: {run_dir.name}")
        except Exception as exc:
            messagebox.showerror("错误", f"删除失败:\n{exc}")
        self.refresh_runs()

    def delete_all(self) -> None:
        if not RUNS_DIR.exists() or not any(RUNS_DIR.iterdir()):
            messagebox.showinfo("提示", "没有训练记录")
            return
        if not messagebox.askyesno("确认", "确定清空所有训练记录？\n此操作不可恢复！"):
            return
        try:
            shutil.rmtree(RUNS_DIR)
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            self.log("已清空所有训练记录")
        except Exception as exc:
            messagebox.showerror("错误", f"清理失败:\n{exc}")
        self.refresh_runs()

    def launch_demo(self) -> None:
        run_dir = self._selected_run_dir()
        if run_dir is None:
            return

        ckpt = run_dir / "checkpoints" / "best.pt"
        if not ckpt.exists():
            ckpt = run_dir / "checkpoints" / "latest.pt"
        if not ckpt.exists():
            messagebox.showwarning("提示", f"未找到模型文件:\n{run_dir / 'checkpoints'}")
            return

        self.stop_inference()
        try:
            self.inference_proc = self._popen_detach(
                [
                    sys.executable,
                    "-m",
                    "snake_rl.cli",
                    "serve-model",
                    "--port",
                    "8765",
                    "--checkpoint",
                    str(ckpt),
                ]
            )
            self.log(f"[{self._ts()}] 推理服务已启动 (模型: {ckpt.name})")
            self.log("  请在游戏页面 AI 接管面板连接 http://127.0.0.1:8765")
        except Exception as exc:
            messagebox.showerror("错误", f"启动推理服务失败:\n{exc}")
            return

        game = WEB_DIR / "index.html"
        if game.exists():
            webbrowser.open(game.as_uri())

    def stop_inference(self) -> None:
        if self.inference_proc is not None and self.inference_proc.poll() is None:
            terminate_process(self.inference_proc)
            self.inference_proc = None
            self.log(f"[{self._ts()}] 推理服务已停止")

    def open_tensorboard(self) -> None:
        if self.tb_proc is not None and self.tb_proc.poll() is None:
            self.log("TensorBoard 已在运行，正在打开浏览器...")
            webbrowser.open("http://127.0.0.1:6006")
            return
        try:
            self.tb_proc = self._popen_detach(
                [sys.executable, "-m", "snake_rl.cli", "monitor", "--port", "6006"]
            )
            self.log(f"[{self._ts()}] TensorBoard 已启动: http://127.0.0.1:6006")
            self.root.after(2500, lambda: webbrowser.open("http://127.0.0.1:6006"))
        except Exception as exc:
            messagebox.showerror("错误", f"启动 TensorBoard 失败:\n{exc}")

    def stop_tensorboard(self) -> None:
        if self.tb_proc is not None and self.tb_proc.poll() is None:
            terminate_process(self.tb_proc)
            self.tb_proc = None
            self.log(f"[{self._ts()}] TensorBoard 已停止")

    def open_game(self) -> None:
        game = WEB_DIR / "index.html"
        if game.exists():
            webbrowser.open(game.as_uri())
        else:
            messagebox.showerror("错误", "找不到 web/index.html")

    def estimate_time(self) -> None:
        self.log(f"[{self._ts()}] 正在估算训练时间...")
        scheme = self.scheme_var.get()
        use_parallel = bool(self.parallel_var.get())
        workers = max(1, int(self.parallel_workers_var.get()))
        sync_interval = max(1, int(self.parallel_sync_var.get()))

        def _run() -> None:
            try:
                cmd = [sys.executable, "-m", "snake_rl.cli", "estimate", "--scheme", scheme]
                if use_parallel:
                    cmd.extend(
                        [
                            "--parallel",
                            "--parallel-workers",
                            str(workers),
                            "--parallel-sync-interval",
                            str(sync_interval),
                        ]
                    )
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=180,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                out = (result.stdout or "") + (result.stderr or "")
                self.root.after(0, self.log, f"--- 训练时间估算 ---\n{out.strip()}")
            except subprocess.TimeoutExpired:
                self.root.after(0, self.log, "估算超时（超过 3 分钟）")
            except Exception as exc:
                self.root.after(0, self.log, f"估算失败: {exc}")

        threading.Thread(target=_run, daemon=True).start()

    def on_close(self) -> None:
        for proc in (self.training_proc, self.tb_proc, self.inference_proc):
            if proc is not None and proc.poll() is None:
                terminate_process(proc, timeout_s=2.0)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = TrainingManager(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
