"""
贪吃蛇 AI 训练管理器 —— 一体化 GUI（包内入口）。

启动：
    uv run snake-gui
    或 uv run python -m snake_rl.gui_app
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import tkinter as tk
from tkinter import messagebox

try:
    import ttkbootstrap as ttk
    HAS_TTKBOOTSTRAP = True
except ImportError:
    from tkinter import ttk
    HAS_TTKBOOTSTRAP = False

from .process_supervisor import terminate_process
from .run_meta import list_run_metas_sorted, run_meta_to_gui_row
from .schemes import SCHEME_INFO, get_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
RUNS_DIR = PROJECT_ROOT / "runs"
GUI_STATE_PATH = PROJECT_ROOT / ".snake_gui_state.json"
DEFAULT_THEME = "flatly"
THEME_OPTIONS = (
    "flatly",
    "litera",
    "cosmo",
    "minty",
    "darkly",
    "superhero",
    "cyborg",
)
EPISODE_LINE_RE = re.compile(r"\[Episode\s+(\d+)\]")
STAGE_LINE_RE = re.compile(r"\[Stage\s+(\d+)\s+\|\s+Ep\s+(\d+)/(\d+)\]")
STAGE_HEADER_RE = re.compile(r"Curriculum Stage\s+(\d+)/(\d+)")
TOTAL_EPISODES_RE = re.compile(r"总局数上限[：:]\s*(\d+)")
AVG_REWARD_RE = re.compile(r"avg_reward=\s*([-+]?\d+(?:\.\d+)?)")
EPSILON_RE = re.compile(r"\beps=\s*([-+]?\d+(?:\.\d+)?)")

LOG_MAX_LINES = 5000


def _load_gui_state_file() -> dict[str, Any]:
    try:
        if GUI_STATE_PATH.exists():
            data = json.loads(GUI_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_gui_state_file(data: dict[str, Any]) -> None:
    try:
        GUI_STATE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_saved_theme() -> str:
    state = _load_gui_state_file()
    theme = str(state.get("theme", DEFAULT_THEME))
    return theme if theme in THEME_OPTIONS else DEFAULT_THEME


class TrainingManager:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("贪吃蛇 AI 训练管理器")
        self.root.geometry("1024x880")
        self.root.minsize(880, 640)
        self.root.option_add("*tearOff", False)

        self.training_proc: subprocess.Popen | None = None
        self.monitor_proc: subprocess.Popen | None = None
        self.inference_proc: subprocess.Popen | None = None
        self._user_requested_training_stop = False
        self._gui_state = _load_gui_state_file()
        self._runs_all: list[dict[str, str]] = []
        self._sort_column = "name"
        self._sort_reverse = True
        self._runs_refresh_token = 0
        self._progress_total_episodes = 0
        self._progress_current_episode = 0
        self._progress_stage_prefix: dict[int, int] = {}
        self._progress_stage_text = "-"
        self._progress_last_avg_reward: float | None = None
        self._progress_last_epsilon: float | None = None
        self.monitor_port_var = tk.IntVar(
            value=self._as_int(self._gui_state.get("monitor_port"), default=6006, min_v=1024, max_v=65535)
        )
        self.inference_port_var = tk.IntVar(
            value=self._as_int(self._gui_state.get("inference_port"), default=8765, min_v=1024, max_v=65535)
        )
        self.run_detail_name_var = tk.StringVar(value="-")
        self.run_detail_status_var = tk.StringVar(value="-")
        self.run_detail_model_var = tk.StringVar(value="-")
        self.run_detail_episodes_var = tk.StringVar(value="-")
        self.run_detail_best_var = tk.StringVar(value="-")
        self.run_detail_updated_var = tk.StringVar(value="-")
        self.run_detail_ckpt_var = tk.StringVar(value="-")
        self.run_detail_badges_var = tk.StringVar(value="-")
        self._last_selected_run = ""

        self._build_ui()
        self.refresh_runs()
        self._update_service_status()
        self._bind_shortcuts()
        self.root.after(1200, self._poll_service_status)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("", 15, "bold"))
        style.configure("Status.TLabel", font=("", 10))

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="贪吃蛇 AI 训练管理器", style="Title.TLabel").pack(pady=(0, 6))

        nb = ttk.Notebook(main)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_train = ttk.Frame(nb, padding=6)
        tab_runs = ttk.Frame(nb, padding=6)
        tab_settings = ttk.Frame(nb, padding=6)
        nb.add(tab_train, text="训练中心")
        nb.add(tab_runs, text="运行记录")
        nb.add(tab_settings, text="服务与设置")

        self._build_training_section(tab_train)
        self._build_log_section(tab_train)
        self._build_runs_tab(tab_runs)
        self._build_settings_section(tab_settings)

    def _build_training_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="训练控制", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        default_scheme = str(self._gui_state.get("scheme", "scheme1"))
        if default_scheme not in SCHEME_INFO:
            default_scheme = "scheme1"

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="训练方案:").pack(side=tk.LEFT)
        self.scheme_var = tk.StringVar(value=default_scheme)
        cb = ttk.Combobox(
            row1,
            textvariable=self.scheme_var,
            values=list(SCHEME_INFO.keys()),
            state="readonly",
            width=12,
        )
        cb.pack(side=tk.LEFT, padx=5)
        cb.bind("<<ComboboxSelected>>", self._on_scheme_changed)

        self.scheme_desc_var = tk.StringVar(value=SCHEME_INFO[default_scheme])
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
        self.parallel_var = tk.BooleanVar(value=bool(self._gui_state.get("parallel", False)))
        ttk.Checkbutton(
            row3,
            text="启用并行采样",
            variable=self.parallel_var,
        ).pack(side=tk.LEFT)
        ttk.Label(row3, text="Workers:").pack(side=tk.LEFT, padx=(12, 4))
        self.parallel_workers_var = tk.IntVar(
            value=self._as_int(self._gui_state.get("parallel_workers"), default=4, min_v=1, max_v=64)
        )
        ttk.Spinbox(
            row3,
            from_=1,
            to=64,
            width=6,
            textvariable=self.parallel_workers_var,
        ).pack(side=tk.LEFT)
        ttk.Label(row3, text="同步步长:").pack(side=tk.LEFT, padx=(12, 4))
        self.parallel_sync_var = tk.IntVar(
            value=self._as_int(
                self._gui_state.get("parallel_sync_interval"),
                default=512,
                min_v=16,
                max_v=100000,
            )
        )
        ttk.Spinbox(
            row3,
            from_=16,
            to=100000,
            increment=16,
            width=8,
            textvariable=self.parallel_sync_var,
        ).pack(side=tk.LEFT)

        row4 = ttk.Frame(frame)
        row4.pack(fill=tk.X, pady=(8, 0))
        self.progress_text_var = tk.StringVar(value="训练进度：0 / ?")
        ttk.Label(row4, textvariable=self.progress_text_var).pack(side=tk.LEFT)
        self.progress_detail_var = tk.StringVar(value="最近指标：avg_reward=- | eps=- | stage=-")
        ttk.Label(row4, textvariable=self.progress_detail_var).pack(side=tk.RIGHT)

        self.progress_value_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            frame,
            maximum=100.0,
            variable=self.progress_value_var,
            mode="determinate",
        )
        self.progress_bar.pack(fill=tk.X, pady=(4, 0))

    def _build_runs_tab(self, parent: ttk.Frame) -> None:
        paned = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, padding=(0, 0, 6, 0))
        right = ttk.Frame(paned, padding=(6, 0, 0, 0))
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        frame = ttk.LabelFrame(left, text="训练记录", padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        filter_row = ttk.Frame(frame)
        filter_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(filter_row, text="筛选:").pack(side=tk.LEFT)
        self.run_keyword_var = tk.StringVar(value=str(self._gui_state.get("run_keyword", "")))
        run_entry = ttk.Entry(filter_row, textvariable=self.run_keyword_var, width=22)
        run_entry.pack(side=tk.LEFT, padx=(4, 8))
        run_entry.bind("<KeyRelease>", self._apply_run_filters)

        ttk.Label(filter_row, text="状态:").pack(side=tk.LEFT)
        status_options = ("全部", "完成", "训练中", "中断", "空", "未知")
        initial_status = str(self._gui_state.get("run_status", "全部"))
        if initial_status not in status_options:
            initial_status = "全部"
        self.run_status_var = tk.StringVar(value=initial_status)
        status_cb = ttk.Combobox(
            filter_row,
            textvariable=self.run_status_var,
            values=status_options,
            state="readonly",
            width=8,
        )
        status_cb.pack(side=tk.LEFT, padx=(4, 8))
        status_cb.bind("<<ComboboxSelected>>", self._apply_run_filters)
        ttk.Button(filter_row, text="清除筛选", command=self._clear_run_filters).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.runs_stats_var = tk.StringVar(value="显示 0 / 共 0 条")
        ttk.Label(filter_row, textvariable=self.runs_stats_var).pack(side=tk.RIGHT)

        columns = ("name", "model", "episodes", "best_reward", "status")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=16)
        self.tree.heading("name", text="运行名称", command=lambda: self._sort_by("name"))
        self.tree.heading("model", text="模型", command=lambda: self._sort_by("model"))
        self.tree.heading("episodes", text="局数", command=lambda: self._sort_by("episodes"))
        self.tree.heading(
            "best_reward",
            text="最佳平均奖励",
            command=lambda: self._sort_by("best_reward"),
        )
        self.tree.heading("status", text="状态", command=lambda: self._sort_by("status"))
        self.tree.column("name", width=200)
        self.tree.column("model", width=80)
        self.tree.column("episodes", width=64, anchor=tk.CENTER)
        self.tree.column("best_reward", width=100, anchor=tk.CENTER)
        self.tree.column("status", width=72, anchor=tk.CENTER)
        self.tree.bind("<Double-1>", self._on_run_double_click)
        self.tree.bind("<<TreeviewSelect>>", self._on_run_select)
        self._update_sort_headings()

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_frame, text="刷新列表", command=self.refresh_runs).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="删除选中", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="清空所有记录", command=self.delete_all).pack(side=tk.LEFT, padx=2)

        detail = ttk.LabelFrame(right, text="选中运行", padding=10)
        detail.pack(fill=tk.BOTH, expand=True)

        def row(lbl: str, var: tk.StringVar) -> None:
            r = ttk.Frame(detail)
            r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=lbl, width=12).pack(side=tk.LEFT)
            ttk.Label(r, textvariable=var, wraplength=320).pack(side=tk.LEFT, fill=tk.X, expand=True)

        row("名称", self.run_detail_name_var)
        row("状态", self.run_detail_status_var)
        row("模型", self.run_detail_model_var)
        row("局数", self.run_detail_episodes_var)
        row("最佳 avg", self.run_detail_best_var)
        row("最近活动", self.run_detail_updated_var)
        row("检查点", self.run_detail_badges_var)

        ck = ttk.Frame(detail)
        ck.pack(fill=tk.X, pady=(6, 4))
        ttk.Label(ck, text="演示将加载", wraplength=360).pack(anchor=tk.W)
        ttk.Label(ck, textvariable=self.run_detail_ckpt_var, wraplength=360).pack(anchor=tk.W)

        act = ttk.Frame(detail)
        act.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(act, text="用此模型演示", command=self.launch_demo).pack(fill=tk.X, pady=2)
        ttk.Button(act, text="在监控中打开", command=self.open_monitor_for_selected).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(act, text="打开运行目录", command=self.open_selected_dir).pack(fill=tk.X, pady=2)
        ttk.Button(act, text="停止推理服务", command=self.stop_inference).pack(fill=tk.X, pady=2)

    def _build_settings_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="端口与服务", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))

        prow = ttk.Frame(frame)
        prow.pack(fill=tk.X)
        ttk.Label(prow, text="监控端口:").pack(side=tk.LEFT)
        ttk.Spinbox(
            prow,
            from_=1024,
            to=65535,
            width=8,
            textvariable=self.monitor_port_var,
        ).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(prow, text="推理端口:").pack(side=tk.LEFT)
        ttk.Spinbox(
            prow,
            from_=1024,
            to=65535,
            width=8,
            textvariable=self.inference_port_var,
        ).pack(side=tk.LEFT, padx=(6, 0))

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(row, text="打开监控后台", command=self.open_monitor).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="关闭监控后台", command=self.stop_monitor).pack(side=tk.LEFT, padx=2)
        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(row, text="打开游戏", command=self.open_game).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="估算训练时间", command=self.estimate_time).pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=(10, 0))
        self.service_status_var = tk.StringVar(value="")
        ttk.Label(row2, textvariable=self.service_status_var, style="Status.TLabel").pack(
            side=tk.LEFT
        )

        if HAS_TTKBOOTSTRAP:
            row3 = ttk.Frame(frame)
            row3.pack(fill=tk.X, pady=(10, 0))
            ttk.Label(row3, text="主题:").pack(side=tk.LEFT)
            self.theme_var = tk.StringVar(value=self._current_theme_name())
            theme_cb = ttk.Combobox(
                row3,
                textvariable=self.theme_var,
                values=THEME_OPTIONS,
                state="readonly",
                width=10,
            )
            theme_cb.pack(side=tk.LEFT, padx=(6, 6))
            ttk.Button(row3, text="应用主题", command=self.apply_theme).pack(side=tk.LEFT)

    def _build_log_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="输出日志", padding=4)
        frame.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(toolbar, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="复制日志", command=self.copy_log).pack(side=tk.LEFT, padx=2)
        self.log_autoscroll_var = tk.BooleanVar(
            value=bool(self._gui_state.get("log_autoscroll", True))
        )
        ttk.Checkbutton(
            toolbar,
            text="自动滚动",
            variable=self.log_autoscroll_var,
        ).pack(side=tk.RIGHT, padx=2)

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

    @staticmethod
    def _as_int(value: Any, default: int, min_v: int, max_v: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(min_v, min(max_v, parsed))

    def _current_theme_name(self) -> str:
        try:
            current = str(ttk.Style().theme_use())
            return current or DEFAULT_THEME
        except Exception:
            return DEFAULT_THEME

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-r>", lambda _e: self.refresh_runs())
        self.root.bind("<Control-l>", lambda _e: self.clear_log())
        self.root.bind("<F5>", lambda _e: self.refresh_runs())

    def _is_alive(self, proc: subprocess.Popen | None) -> bool:
        return proc is not None and proc.poll() is None

    def _update_service_status(self) -> None:
        train = "运行中" if self._is_alive(self.training_proc) else "空闲"
        monitor = "运行中" if self._is_alive(self.monitor_proc) else "关闭"
        infer = "运行中" if self._is_alive(self.inference_proc) else "关闭"
        mp = int(self.monitor_port_var.get())
        ip = int(self.inference_port_var.get())
        self.service_status_var.set(
            f"服务状态 训练:{train} | 监控:{monitor} @:{mp} | 推理:{infer} @:{ip}"
        )

    def _poll_service_status(self) -> None:
        try:
            if not self.root.winfo_exists():
                return
            self._update_service_status()
            self.root.after(1200, self._poll_service_status)
        except tk.TclError:
            return

    def _clear_run_filters(self) -> None:
        self.run_keyword_var.set("")
        self.run_status_var.set("全部")
        self._apply_run_filters()

    def _sort_key(self, info: dict[str, str], column: str) -> tuple[int, Any]:
        raw = str(info.get(column, ""))
        if column in {"episodes", "best_reward"}:
            try:
                return (0, float(raw))
            except Exception:
                return (1, 0.0)
        return (0, raw.lower())

    def _update_sort_headings(self) -> None:
        headings = {
            "name": "运行名称",
            "model": "模型",
            "episodes": "局数",
            "best_reward": "最佳平均奖励",
            "status": "状态",
        }
        for key, title in headings.items():
            arrow = ""
            if key == self._sort_column:
                arrow = " ↓" if self._sort_reverse else " ↑"
            self.tree.heading(key, text=title + arrow)

    def _sort_by(self, column: str) -> None:
        if self._sort_column == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = column in {"name", "episodes", "best_reward"}
        self._apply_run_filters()

    def _on_run_double_click(self, _event: Any = None) -> None:
        if self.tree.selection():
            self.launch_demo()

    def _apply_run_filters(self, _event: Any = None) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        keyword = self.run_keyword_var.get().strip().lower()
        status = self.run_status_var.get()

        filtered: list[dict[str, str]] = []
        for info in self._runs_all:
            if keyword:
                search_area = f"{info['name']} {info['model']}".lower()
                if keyword not in search_area:
                    continue
            if status != "全部" and info["status"] != status:
                continue
            filtered.append(info)

        sorted_infos = sorted(
            filtered,
            key=lambda item: self._sort_key(item, self._sort_column),
            reverse=self._sort_reverse,
        )
        select_iid: str | None = None
        for info in sorted_infos:
            iid = self.tree.insert(
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
            if info.get("name") == self._last_selected_run:
                select_iid = iid
        children = self.tree.get_children()
        if select_iid:
            self.tree.selection_set(select_iid)
            self.tree.focus(select_iid)
            self.tree.see(select_iid)
        elif children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
            self.tree.see(children[0])
        self.runs_stats_var.set(f"显示 {len(sorted_infos)} / 共 {len(self._runs_all)} 条")
        self._update_sort_headings()
        self._update_run_detail_panel()

    def _collect_gui_state(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "scheme": self.scheme_var.get(),
            "parallel": bool(self.parallel_var.get()),
            "parallel_workers": int(self.parallel_workers_var.get()),
            "parallel_sync_interval": int(self.parallel_sync_var.get()),
            "run_keyword": self.run_keyword_var.get().strip(),
            "run_status": self.run_status_var.get(),
            "log_autoscroll": bool(self.log_autoscroll_var.get()),
            "theme": self._current_theme_name(),
            "monitor_port": int(self.monitor_port_var.get()),
            "inference_port": int(self.inference_port_var.get()),
        }
        return data

    def _save_gui_state(self) -> None:
        _save_gui_state_file(self._collect_gui_state())

    def clear_log(self) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    def copy_log(self) -> None:
        content = self.log_text.get("1.0", tk.END).strip()
        if not content:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.log(f"[{self._ts()}] 日志已复制到剪贴板")

    def apply_theme(self) -> None:
        if not HAS_TTKBOOTSTRAP:
            return
        selected = self.theme_var.get().strip()
        if selected not in THEME_OPTIONS:
            messagebox.showwarning("提示", f"不支持的主题: {selected}")
            return
        try:
            ttk.Style().theme_use(selected)
            self.log(f"[{self._ts()}] 已切换主题: {selected}")
            self._save_gui_state()
        except Exception as exc:
            messagebox.showerror("错误", f"主题切换失败:\n{exc}")

    def _prepare_progress_context(self, scheme: str) -> None:
        self._progress_total_episodes = 0
        self._progress_stage_prefix.clear()
        self._progress_current_episode = 0
        self._progress_stage_text = "-"
        self._progress_last_avg_reward = None
        self._progress_last_epsilon = None
        try:
            cfg = get_config(scheme=scheme)
            if cfg.curriculum is not None and cfg.curriculum.stages:
                total = 0
                for idx, stage in enumerate(cfg.curriculum.stages, start=1):
                    self._progress_stage_prefix[idx] = total
                    total += int(stage.episodes)
                self._progress_total_episodes = max(0, total)
            else:
                self._progress_total_episodes = max(0, int(cfg.episodes))
        except Exception:
            self._progress_total_episodes = 0
        self._refresh_progress_widgets()

    def _refresh_progress_widgets(self) -> None:
        total = self._progress_total_episodes
        current = self._progress_current_episode
        if total > 0:
            progress_pct = max(0.0, min(100.0, 100.0 * current / total))
            self.progress_text_var.set(f"训练进度：{current} / {total} ({progress_pct:.1f}%)")
        else:
            progress_pct = 0.0
            self.progress_text_var.set(f"训练进度：{current} / ?")
        self.progress_value_var.set(progress_pct)
        avg_text = "-" if self._progress_last_avg_reward is None else f"{self._progress_last_avg_reward:.3f}"
        eps_text = "-" if self._progress_last_epsilon is None else f"{self._progress_last_epsilon:.3f}"
        self.progress_detail_var.set(
            f"最近指标：avg_reward={avg_text} | eps={eps_text} | stage={self._progress_stage_text}"
        )

    def _handle_training_output_line(self, line: str) -> None:
        self.log(line)
        self._update_progress_from_line(line)

    def _update_progress_from_line(self, line: str) -> None:
        total_match = TOTAL_EPISODES_RE.search(line)
        if total_match:
            try:
                self._progress_total_episodes = max(
                    self._progress_total_episodes,
                    int(total_match.group(1)),
                )
            except Exception:
                pass

        episode_match = EPISODE_LINE_RE.search(line)
        if episode_match:
            try:
                self._progress_current_episode = max(
                    self._progress_current_episode,
                    int(episode_match.group(1)),
                )
            except Exception:
                pass

        stage_match = STAGE_LINE_RE.search(line)
        if stage_match:
            try:
                stage_index = int(stage_match.group(1))
                stage_episode = int(stage_match.group(2))
                stage_total = int(stage_match.group(3))
                prefix = self._progress_stage_prefix.get(stage_index)
                if prefix is not None:
                    self._progress_current_episode = max(
                        self._progress_current_episode,
                        prefix + stage_episode,
                    )
                self._progress_stage_text = f"{stage_index} ({stage_episode}/{stage_total})"
            except Exception:
                pass
        else:
            stage_header_match = STAGE_HEADER_RE.search(line)
            if stage_header_match:
                self._progress_stage_text = f"{stage_header_match.group(1)} (准备中)"

        avg_match = AVG_REWARD_RE.search(line)
        if avg_match:
            try:
                self._progress_last_avg_reward = float(avg_match.group(1))
            except Exception:
                pass

        eps_match = EPSILON_RE.search(line)
        if eps_match:
            try:
                self._progress_last_epsilon = float(eps_match.group(1))
            except Exception:
                pass

        self._refresh_progress_widgets()

    def log(self, text: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        try:
            end_line = int(str(self.log_text.index("end-1c")).split(".", maxsplit=1)[0])
            if end_line > LOG_MAX_LINES:
                drop = end_line - LOG_MAX_LINES
                self.log_text.delete("1.0", f"{drop + 1}.0")
        except (tk.TclError, ValueError):
            pass
        if self.log_autoscroll_var.get():
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

    def _popen_service(self, args: list[str], log_tag: str) -> subprocess.Popen:
        proc = subprocess.Popen(
            args,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            creationflags=self._win_flags(),
        )

        def _read_stderr() -> None:
            try:
                if proc.stderr is not None:
                    for line in proc.stderr:
                        stripped = line.rstrip("\n\r")
                        if stripped:
                            self.root.after(0, self.log, f"[{log_tag}] {stripped}")
            except Exception as exc:
                self.root.after(0, self.log, f"[{log_tag}] stderr 读取异常: {exc}")

        threading.Thread(target=_read_stderr, daemon=True).start()
        return proc

    @staticmethod
    def _tcp_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    @staticmethod
    def _http_ok(url: str, timeout: float = 1.0) -> bool:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return int(getattr(resp, "status", 200)) == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _selected_run_name(self) -> str | None:
        sel = self.tree.selection()
        if not sel:
            return None
        return str(self.tree.item(sel[0], "values")[0])

    def _selected_run_dir(self) -> Path | None:
        name = self._selected_run_name()
        if name is None:
            messagebox.showinfo("提示", "请先在列表中选择一个训练记录")
            return None
        return RUNS_DIR / name

    def _on_run_select(self, _event: Any = None) -> None:
        name = self._selected_run_name()
        if name:
            self._last_selected_run = name
        self._update_run_detail_panel()

    def _update_run_detail_panel(self) -> None:
        name = self._selected_run_name()
        if not name:
            self.run_detail_name_var.set("-")
            self.run_detail_status_var.set("-")
            self.run_detail_model_var.set("-")
            self.run_detail_episodes_var.set("-")
            self.run_detail_best_var.set("-")
            self.run_detail_updated_var.set("-")
            self.run_detail_badges_var.set("-")
            self.run_detail_ckpt_var.set("-")
            return
        info = next((r for r in self._runs_all if r.get("name") == name), None)
        if not info:
            self.run_detail_name_var.set(name)
            self.run_detail_status_var.set("（加载中或已删除）")
            return
        self.run_detail_name_var.set(info.get("name", "-"))
        self.run_detail_status_var.set(info.get("status", "-"))
        self.run_detail_model_var.set(info.get("model", "-"))
        self.run_detail_episodes_var.set(info.get("episodes", "-"))
        self.run_detail_best_var.set(info.get("best_reward", "-"))
        self.run_detail_updated_var.set(info.get("updated", "-"))
        self.run_detail_badges_var.set(info.get("badges", "-"))
        run_dir = RUNS_DIR / name
        ckpt = run_dir / "checkpoints" / "best.pt"
        if not ckpt.exists():
            ckpt = run_dir / "checkpoints" / "latest.pt"
        self.run_detail_ckpt_var.set(str(ckpt) if ckpt.exists() else "（无可用 .pt）")

    def start_training(self) -> None:
        if self.training_proc is not None and self.training_proc.poll() is None:
            messagebox.showwarning("提示", "训练已在进行中")
            return

        scheme = self.scheme_var.get()
        use_parallel = bool(self.parallel_var.get())
        workers = max(1, int(self.parallel_workers_var.get()))
        sync_interval = max(1, int(self.parallel_sync_var.get()))
        self._prepare_progress_context(scheme)
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
            self._save_gui_state()
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
        self._update_service_status()

        threading.Thread(target=self._read_proc_output, args=(self.training_proc,), daemon=True).start()

    def _read_proc_output(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                stripped = line.rstrip("\n\r")
                if stripped:
                    self.root.after(0, self._handle_training_output_line, stripped)
        except Exception as exc:
            self.root.after(0, self.log, f"[训练] 读取子进程输出失败: {exc}")
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
        self.training_proc = None
        if code == 0 and self._progress_total_episodes > 0:
            self._progress_current_episode = self._progress_total_episodes
            self._refresh_progress_widgets()
        self.status_var.set(msg)
        self.log(f"[{self._ts()}] {msg}")
        self.refresh_runs()
        self._update_service_status()

    def stop_training(self) -> None:
        if self.training_proc is None or self.training_proc.poll() is not None:
            return
        if not messagebox.askyesno("确认", "确定要停止训练吗？\n已保存的检查点不会丢失。"):
            return
        self._user_requested_training_stop = True
        terminate_process(self.training_proc)
        self.status_var.set("正在停止...")
        self.log(f"[{self._ts()}] 正在停止训练...")
        self._update_service_status()

    def refresh_runs(self) -> None:
        self._runs_refresh_token += 1
        token = self._runs_refresh_token
        self.runs_stats_var.set("训练记录加载中...")
        threading.Thread(
            target=self._load_runs_worker,
            args=(token,),
            daemon=True,
        ).start()

    def _load_runs_worker(self, token: int) -> None:
        infos: list[dict[str, str]] = []
        err_msg = ""
        try:
            if RUNS_DIR.exists():
                for meta in list_run_metas_sorted(RUNS_DIR):
                    infos.append(run_meta_to_gui_row(meta))
        except Exception as exc:
            err_msg = str(exc)
        try:
            self.root.after(0, self._on_runs_loaded, token, infos, err_msg)
        except Exception:
            return

    def _on_runs_loaded(self, token: int, infos: list[dict[str, str]], err_msg: str) -> None:
        if token != self._runs_refresh_token:
            return
        self._runs_all = infos
        self._apply_run_filters()
        if err_msg:
            self.log(f"[{self._ts()}] 训练记录加载异常: {err_msg}")

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

    def open_selected_dir(self) -> None:
        run_dir = self._selected_run_dir()
        if run_dir is None:
            return
        try:
            if os.name == "nt":
                os.startfile(str(run_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(run_dir)])
            else:
                subprocess.Popen(["xdg-open", str(run_dir)])
        except Exception as exc:
            messagebox.showerror("错误", f"打开目录失败:\n{exc}")

    def _wait_infer_then_open_game(self, port: int, attempt: int = 0) -> None:
        health_url = f"http://127.0.0.1:{port}/health"
        if self._http_ok(health_url):
            self.log(f"[{self._ts()}] 推理服务就绪")
            game = WEB_DIR / "index.html"
            if game.exists():
                webbrowser.open(game.as_uri())
            return
        if self.inference_proc is None or self.inference_proc.poll() is not None:
            return
        if attempt >= 50:
            msg = "等待推理服务 /health 超时，未自动打开游戏页；请查看 [infer] 日志后重试。"
            self.log(f"[{self._ts()}] {msg}")
            messagebox.showwarning("提示", msg)
            return
        self.root.after(250, lambda: self._wait_infer_then_open_game(port, attempt + 1))

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

        port = max(1024, min(65535, int(self.inference_port_var.get())))
        self._save_gui_state()
        health_url = f"http://127.0.0.1:{port}/health"

        if self._tcp_port_open("127.0.0.1", port):
            if self._http_ok(health_url):
                if not messagebox.askyesno(
                    "确认",
                    f"端口 {port} 上已有推理服务在运行。\n是否打开游戏页面（使用已有服务）？",
                ):
                    return
                game = WEB_DIR / "index.html"
                if game.exists():
                    webbrowser.open(game.as_uri())
                return
            messagebox.showwarning(
                "提示",
                f"端口 {port} 已被占用，但未能识别为本项目的推理服务。\n"
                "请在「服务与设置」中更换推理端口，或关闭占用该端口的程序。",
            )
            return

        self.stop_inference()
        try:
            self.inference_proc = self._popen_service(
                [
                    sys.executable,
                    "-m",
                    "snake_rl.cli",
                    "serve-model",
                    "--port",
                    str(port),
                    "--checkpoint",
                    str(ckpt),
                ],
                "infer",
            )
        except Exception as exc:
            messagebox.showerror("错误", f"启动推理服务失败:\n{exc}")
            return

        self.log(f"[{self._ts()}] 推理服务启动中: {ckpt.name} (端口 {port})")
        self.log(f"  游戏页 AI 接管面板连接 http://127.0.0.1:{port}")
        self._update_service_status()
        if self.inference_proc.poll() is not None:
            self.inference_proc = None
            self._update_service_status()
            return
        self._wait_infer_then_open_game(port)

    def stop_inference(self) -> None:
        if self.inference_proc is not None and self.inference_proc.poll() is None:
            terminate_process(self.inference_proc)
            self.inference_proc = None
            self.log(f"[{self._ts()}] 推理服务已停止")
            self._update_service_status()

    def _monitor_base_url(self) -> tuple[str, int]:
        port = max(1024, min(65535, int(self.monitor_port_var.get())))
        return f"http://127.0.0.1:{port}", port

    def _dashboard_url(self, run_name: str | None) -> str:
        base, _port = self._monitor_base_url()
        path = "/dashboard"
        if run_name:
            path += f"?run={quote(run_name, safe='')}"
        return f"{base}{path}"

    def _wait_monitor_then_open(self, run_name: str | None, attempt: int = 0) -> None:
        base, _port = self._monitor_base_url()
        health = f"{base}/health"
        dashboard_url = self._dashboard_url(run_name)
        if self._http_ok(health):
            self.log(f"[{self._ts()}] 监控后台已就绪")
            webbrowser.open(dashboard_url)
            return
        if self.monitor_proc is not None and self.monitor_proc.poll() is not None:
            self.log(f"[{self._ts()}] 监控进程已退出，请查看 [monitor] 日志")
            return
        if attempt >= 50:
            msg = "等待监控后台 /health 超时，未自动打开浏览器；请查看 [monitor] 日志后重试。"
            self.log(f"[{self._ts()}] {msg}")
            messagebox.showwarning("提示", msg)
            return
        self.root.after(250, lambda: self._wait_monitor_then_open(run_name, attempt + 1))

    def open_monitor(self) -> None:
        self._open_monitor_dashboard(run_name=None)

    def open_monitor_for_selected(self) -> None:
        name = self._selected_run_name()
        if not name:
            messagebox.showinfo("提示", "请先在列表中选择一个训练记录")
            return
        self._open_monitor_dashboard(run_name=name)

    def _open_monitor_dashboard(self, run_name: str | None) -> None:
        self._save_gui_state()
        base, port = self._monitor_base_url()
        health = f"{base}/health"
        dashboard_url = self._dashboard_url(run_name)

        if self.monitor_proc is not None and self.monitor_proc.poll() is None:
            self.log("监控后台（由本 GUI 启动）已在运行，打开浏览器...")
            if self._http_ok(health):
                webbrowser.open(dashboard_url)
            else:
                self._wait_monitor_then_open(run_name)
            return

        if self._tcp_port_open("127.0.0.1", port) and self._http_ok(health):
            self.log(f"检测到端口 {port} 已有监控服务，直接打开浏览器")
            webbrowser.open(dashboard_url)
            return

        if self._tcp_port_open("127.0.0.1", port) and not self._http_ok(health):
            messagebox.showwarning(
                "提示",
                f"端口 {port} 已被占用，但未能识别为 Snake 监控服务。\n"
                "请在「服务与设置」中更换监控端口，或关闭占用该端口的程序。",
            )
            return

        try:
            self.monitor_proc = self._popen_service(
                [
                    sys.executable,
                    "-m",
                    "snake_rl.cli",
                    "monitor",
                    "--port",
                    str(port),
                ],
                "monitor",
            )
        except Exception as exc:
            messagebox.showerror("错误", f"启动监控后台失败:\n{exc}")
            return

        self.log(f"[{self._ts()}] 已启动监控进程，等待 /health ...")
        self._update_service_status()
        if self.monitor_proc.poll() is not None:
            self.monitor_proc = None
            self._update_service_status()
            return
        self._wait_monitor_then_open(run_name)

    def stop_monitor(self) -> None:
        if self.monitor_proc is not None and self.monitor_proc.poll() is None:
            terminate_process(self.monitor_proc)
            self.monitor_proc = None
            self.log(f"[{self._ts()}] 监控后台已停止")
            self._update_service_status()

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
        self._save_gui_state()
        for proc in (self.training_proc, self.monitor_proc, self.inference_proc):
            if proc is not None and proc.poll() is None:
                terminate_process(proc, timeout_s=2.0)
        self.root.destroy()


def main() -> None:
    root = ttk.Window(themename=_load_saved_theme()) if HAS_TTKBOOTSTRAP else tk.Tk()
    app = TrainingManager(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
