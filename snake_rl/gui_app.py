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
from dataclasses import asdict
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

try:
    import ttkbootstrap as tb
    HAS_TTKBOOTSTRAP = True
except ImportError:
    tb = None
    HAS_TTKBOOTSTRAP = False

from .config import TrainConfig, train_config_from_dict
from .process_supervisor import terminate_process
from .run_meta import list_run_metas_sorted, run_meta_to_gui_row
from .schemes import SCHEME_INFO, default_custom_train_config, get_config
from .train import validate_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
RUNS_DIR = PROJECT_ROOT / "runs"
GUI_STATE_PATH = PROJECT_ROOT / ".snake_gui_state.json"
DEFAULT_THEME = "flatly"
EPISODE_LINE_RE = re.compile(r"\[Episode\s+(\d+)\]")
STAGE_LINE_RE = re.compile(r"\[Stage\s+(\d+)\s+\|\s+Ep\s+(\d+)/(\d+)\]")
STAGE_HEADER_RE = re.compile(r"Curriculum Stage\s+(\d+)/(\d+)")
TOTAL_EPISODES_RE = re.compile(r"总局数上限[：:]\s*(\d+)")
AVG_REWARD_RE = re.compile(r"avg_reward=\s*([-+]?\d+(?:\.\d+)?)")
EPSILON_RE = re.compile(r"\beps=\s*([-+]?\d+(?:\.\d+)?)")

LOG_MAX_LINES = 5000


def _train_config_to_json_text(cfg: TrainConfig) -> str:
    payload = asdict(cfg)
    payload["output_root"] = str(cfg.output_root)
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# ConfigFormEditor — 表单式参数编辑器
# 将 TrainConfig JSON 渲染为分 Tab 页的带说明表单，取代纯文本编辑框。
# ---------------------------------------------------------------------------

# 每个字段描述：(中文标签, 简短说明, 控件类型, 额外选项)
# 控件类型: "entry" | "spinbox" | "check" | "combo" | "label"
_FIELD_TIPS: dict[str, tuple[str, str]] = {
    "run_name":               ("实验名称",     "输出目录 runs/<名称>/，每次实验建议用不同名字方便 TensorBoard 对比"),
    "output_root":            ("输出根目录",    "训练产物存储位置（相对项目根），通常不必修改"),
    "device":                 ("计算设备",      "auto = 有 GPU 自动用，否则用 CPU；也可填 cuda / cpu"),
    "episodes":               ("训练总局数",    "训练跑多少局；14×14 地图建议 20000~50000"),
    "max_steps_per_episode":  ("每局最大步数",  "硬上限，防止蛇无限绕圈；建议 board_size² × 10 左右"),
    "model_type":             ("网络类型",      "small_cnn=入门首选；adaptive_cnn/hybrid 支持可变地图"),
    "local_patch_size":       ("局部感受野",    "仅 hybrid 模型有效，蛇头周围观察范围（奇数）"),
    "learning_rate":          ("学习率",        "最重要超参；推荐 1e-4，不稳定时降低 10 倍"),
    "weight_decay":           ("L2 正则化",     "防过拟合，1e-5 即可；0 = 关闭"),
    "gamma":                  ("折扣因子",      "越接近 1 越看重长远奖励；贪吃蛇推荐 0.99，一般不改"),
    "grad_clip_norm":         ("梯度裁剪",      "防梯度爆炸；loss 暴增时可降低到 1.0"),
    "batch_size":             ("批大小",        "每次更新采样多少条经验；128 是入门平衡点"),
    "replay_capacity":        ("回放池容量",    "最多保存多少条历史经验；14×14 建议 50000~100000"),
    "min_replay_size":        ("最少经验数",    "收集够这么多条经验后才开始更新；建议 batch_size × 20"),
    "train_frequency":        ("更新频率",      "每采集 N 步做一次反向传播；通常保持 4"),
    "target_update_interval": ("目标网更新步",  "每隔多少全局步同步目标网络；建议 1000~3000"),
    "epsilon_start":          ("初始探索率",    "ε-greedy 初始值；1.0 = 完全随机探索"),
    "epsilon_end":            ("最低探索率",    "训练末期保留的最小随机性；推荐 0.01~0.05"),
    "epsilon_decay_steps":    ("探索衰减步数",  "ε 从初始值线性降到最低所需的全局步数（不是局数！）"),
    "moving_avg_window":      ("平均窗口",      "计算平均奖励使用的最近 N 局窗口大小"),
    "eval_episodes":          ("评估局数",      "训练完成后自动评估的局数；0 = 不评估"),
    "checkpoint_interval":    ("检查点频率",    "每隔多少局保存一次模型并增量写入 CSV；建议 200~1000"),
    "log_interval":           ("日志打印频率",  "每隔多少局在终端/GUI 打印一行进度"),
    "tensorboard_log_interval":("TensorBoard频率","每隔多少局写一次 TFEvents；5 即可"),
    "jsonl_flush_interval":   ("JSONL 刷新频率","每隔多少局强制 flush 一次详细日志"),
    "tensorboard":            ("TensorBoard",   "写 TFEvents 供实时监控；强烈建议保持开启"),
    "save_csv":               ("保存 CSV",      "训练完成后存 episodes.csv 供数据分析"),
    "save_jsonl":             ("保存 JSONL",    "实时写每局详细日志，中断也不丢数据"),
    "live_plot":              ("实时曲线",       "弹 Matplotlib 窗口；服务器/无头环境必须关闭"),
    "lightweight_step_info":  ("轻量统计",       "true = 只在局结束时收集统计（更快），false = 每步都收集"),
    # env 子字段
    "env.board_size":         ("地图尺寸",       "格子数（宽=高）；越大越难；入门推荐 10~14"),
    "env.difficulty":         ("难度",           "easy/normal/hard，影响蛇的初始长度"),
    "env.mode":               ("边界模式",        "classic = 碰墙即死；wrap = 穿越到对侧"),
    "env.max_steps_without_food":("无食超时步",  "连续多少步没吃食物则超时；建议 board_size²"),
    "env.enable_bonus_food":  ("奖励食物",        "短暂出现的高奖励食物；入门建议关闭"),
    "env.enable_obstacles":   ("障碍物",          "随机障碍物；入门建议关闭"),
    "env.allow_leveling":     ("升级系统",        "进阶功能；入门建议关闭"),
    "env.seed":               ("随机种子",        "用于复现结果；留空或 0 = 每次随机"),
    # reward_weights 子字段
    "rw.alive":               ("存活奖励",        "每步存活的奖励（负值=惩罚）；推荐 -0.01"),
    "rw.food":                ("吃食物",          "吃到普通食物的奖励；作为基准设 1.0"),
    "rw.bonusFood":           ("奖励食物",        "吃到奖励食物的奖励"),
    "rw.death":               ("死亡惩罚",        "碰墙/自咬的惩罚（负值）；推荐 -1.5"),
    "rw.timeout":             ("超时惩罚",        "无食超时的惩罚（负值）；推荐 -1.0"),
    "rw.levelUp":             ("升级奖励",        "升级事件奖励"),
    "rw.victory":             ("填满奖励",        "填满整张地图的大奖励"),
    "rw.foodDistanceK":       ("靠近食物系数",    "每步靠近食物给正奖励；0~0.5；过大会抖动"),
}


class ConfigFormEditor(ttk.Frame):
    """将 TrainConfig 字典渲染为分 Tab 页的带说明表单。

    对外接口与之前的 tk.Text 兼容：
      get_json_text() -> str     # 读取当前表单 → JSON 文本
      set_json_text(text: str)   # 写入 JSON → 填充表单
    """

    def __init__(self, master: tk.Widget, **kw: Any) -> None:
        super().__init__(master, **kw)
        self._vars: dict[str, tk.Variable] = {}
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)
        self._nb = nb

        self._tab_basic  = self._make_scrollable_tab(nb, "基础")
        self._tab_env    = self._make_scrollable_tab(nb, "环境")
        self._tab_reward = self._make_scrollable_tab(nb, "奖励")
        self._tab_adv    = self._make_scrollable_tab(nb, "高级")

        self._fill_basic(self._tab_basic)
        self._fill_env(self._tab_env)
        self._fill_reward(self._tab_reward)
        self._fill_adv(self._tab_adv)

    def _make_scrollable_tab(self, nb: ttk.Notebook, title: str) -> ttk.Frame:
        outer = ttk.Frame(nb)
        nb.add(outer, text=title)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_resize(evt: Any) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_resize(evt: Any) -> None:
            canvas.itemconfig(win_id, width=evt.width)

        inner.bind("<Configure>", _on_inner_resize)
        canvas.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(evt: Any) -> None:
            canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        return inner

    # ---------------------------------------------------------------- helpers
    def _row(self, parent: ttk.Frame, key: str,
             widget_type: str = "entry",
             choices: list[str] | None = None,
             width: int = 20) -> None:
        """向 parent 添加一行：[标签] [控件] [说明文字]"""
        label_text, tip_text = _FIELD_TIPS.get(key, (key, ""))

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=3)

        lbl = ttk.Label(row, text=label_text, width=14, anchor="e")
        lbl.pack(side=tk.LEFT, padx=(0, 6))

        if widget_type == "check":
            var: tk.Variable = tk.BooleanVar(value=False)
            w = ttk.Checkbutton(row, variable=var)
            w.pack(side=tk.LEFT)
        elif widget_type == "combo" and choices:
            var = tk.StringVar(value=choices[0])
            w = ttk.Combobox(row, textvariable=var, values=choices,
                             state="readonly", width=width)
            w.pack(side=tk.LEFT)
        elif widget_type == "spinbox_int":
            var = tk.StringVar(value="0")
            w = ttk.Spinbox(row, textvariable=var, from_=0, to=10_000_000,
                            width=width)
            w.pack(side=tk.LEFT)
        elif widget_type == "spinbox_float":
            var = tk.StringVar(value="0.0")
            w = ttk.Spinbox(row, textvariable=var, from_=0.0, to=1.0,
                            increment=0.01, format="%.4f", width=width)
            w.pack(side=tk.LEFT)
        else:  # entry (default)
            var = tk.StringVar(value="")
            w = ttk.Entry(row, textvariable=var, width=width)
            w.pack(side=tk.LEFT)

        tip = ttk.Label(row, text=tip_text, foreground="#888888",
                        font=("", 8), wraplength=480, justify=tk.LEFT)
        tip.pack(side=tk.LEFT, padx=(8, 0))

        self._vars[key] = var

    def _section(self, parent: ttk.Frame, title: str) -> None:
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=(10, 2))
        ttk.Label(parent, text=title, font=("", 9, "bold"),
                  foreground="#aaaaaa").pack(anchor="w", padx=12, pady=(0, 4))

    # ---------------------------------------------------------------- tabs
    def _fill_basic(self, p: ttk.Frame) -> None:
        self._section(p, "▸ 实验标识")
        self._row(p, "run_name",  "entry", width=22)
        self._row(p, "output_root", "entry", width=22)
        self._row(p, "device",    "combo",
                  choices=["auto", "cpu", "cuda"], width=10)

        self._section(p, "▸ 训练规模")
        self._row(p, "episodes",              "spinbox_int", width=12)
        self._row(p, "max_steps_per_episode", "spinbox_int", width=12)
        self._row(p, "eval_episodes",         "spinbox_int", width=10)

        self._section(p, "▸ 网络结构")
        self._row(p, "model_type", "combo",
                  choices=["small_cnn", "adaptive_cnn", "hybrid"], width=14)
        self._row(p, "local_patch_size", "spinbox_int", width=8)

        self._section(p, "▸ 探索策略（ε-greedy）")
        self._row(p, "epsilon_start",       "spinbox_float", width=10)
        self._row(p, "epsilon_end",         "spinbox_float", width=10)
        self._row(p, "epsilon_decay_steps", "spinbox_int",   width=12)

    def _fill_env(self, p: ttk.Frame) -> None:
        self._section(p, "▸ 地图设置")
        self._row(p, "env.board_size",  "spinbox_int", width=8)
        self._row(p, "env.mode",        "combo",
                  choices=["classic", "wrap"], width=10)
        self._row(p, "env.difficulty",  "combo",
                  choices=["easy", "normal", "hard"], width=10)
        self._row(p, "env.max_steps_without_food", "spinbox_int", width=10)
        self._row(p, "env.seed", "entry", width=10)

        self._section(p, "▸ 进阶功能（入门建议全部关闭）")
        self._row(p, "env.enable_bonus_food", "check")
        self._row(p, "env.enable_obstacles",  "check")
        self._row(p, "env.allow_leveling",    "check")

    def _fill_reward(self, p: ttk.Frame) -> None:
        self._section(p, "▸ 奖励权重")
        for key in ("rw.alive", "rw.food", "rw.bonusFood", "rw.death",
                    "rw.timeout", "rw.levelUp", "rw.victory", "rw.foodDistanceK"):
            self._row(p, key, "entry", width=10)

    def _fill_adv(self, p: ttk.Frame) -> None:
        self._section(p, "▸ 学习超参数")
        self._row(p, "learning_rate",          "entry", width=12)
        self._row(p, "weight_decay",           "entry", width=12)
        self._row(p, "gamma",                  "spinbox_float", width=10)
        self._row(p, "grad_clip_norm",         "entry", width=10)
        self._row(p, "target_update_interval", "spinbox_int",   width=10)

        self._section(p, "▸ 经验回放")
        self._row(p, "batch_size",       "spinbox_int", width=10)
        self._row(p, "replay_capacity",  "spinbox_int", width=12)
        self._row(p, "min_replay_size",  "spinbox_int", width=10)
        self._row(p, "train_frequency",  "spinbox_int", width=8)

        self._section(p, "▸ 日志与输出")
        self._row(p, "checkpoint_interval",     "spinbox_int", width=10)
        self._row(p, "log_interval",            "spinbox_int", width=8)
        self._row(p, "tensorboard_log_interval","spinbox_int", width=8)
        self._row(p, "jsonl_flush_interval",    "spinbox_int", width=8)
        self._row(p, "moving_avg_window",       "spinbox_int", width=8)
        self._row(p, "tensorboard",       "check")
        self._row(p, "save_csv",          "check")
        self._row(p, "save_jsonl",        "check")
        self._row(p, "live_plot",         "check")
        self._row(p, "lightweight_step_info", "check")

    # ------------------------------------------------------------ public API
    def set_json_text(self, text: str) -> None:
        """将 JSON 字符串解析后填入表单（忽略 $schema 等未知键）。"""
        try:
            data: dict[str, Any] = json.loads(text)
        except Exception:
            return

        def _sv(key: str, val: Any) -> None:
            var = self._vars.get(key)
            if var is None:
                return
            if isinstance(var, tk.BooleanVar):
                var.set(bool(val))
            else:
                var.set(str(val) if val is not None else "")

        for k, v in data.items():
            if k in self._vars:
                _sv(k, v)

        env = data.get("env") or {}
        for k, v in env.items():
            _sv(f"env.{k}", v)

        rw = data.get("reward_weights") or {}
        for k, v in rw.items():
            _sv(f"rw.{k}", v)

    def get_json_text(self) -> str:
        """将表单当前值序列化为 JSON 字符串（结构与 TrainConfig 一致）。"""
        def _get(key: str, cast: type = str) -> Any:
            var = self._vars.get(key)
            if var is None:
                return None
            raw = var.get()
            if cast is bool:
                return bool(raw)
            try:
                return cast(raw)
            except Exception:
                return raw

        def _float(key: str) -> float:
            try:
                return float(self._vars[key].get())
            except Exception:
                return 0.0

        def _int(key: str) -> int:
            try:
                return int(self._vars[key].get())
            except Exception:
                return 0

        def _bool(key: str) -> bool:
            try:
                return bool(self._vars[key].get())
            except Exception:
                return False

        seed_raw = _get("env.seed", str)
        try:
            seed_val: int | None = int(seed_raw) if seed_raw.strip() else None
        except Exception:
            seed_val = None

        d: dict[str, Any] = {
            "run_name":               _get("run_name"),
            "output_root":            _get("output_root"),
            "device":                 _get("device"),
            "episodes":               _int("episodes"),
            "max_steps_per_episode":  _int("max_steps_per_episode"),
            "eval_episodes":          _int("eval_episodes"),
            "model_type":             _get("model_type"),
            "local_patch_size":       _int("local_patch_size"),
            "epsilon_start":          _float("epsilon_start"),
            "epsilon_end":            _float("epsilon_end"),
            "epsilon_decay_steps":    _int("epsilon_decay_steps"),
            "learning_rate":          _float("learning_rate"),
            "weight_decay":           _float("weight_decay"),
            "gamma":                  _float("gamma"),
            "grad_clip_norm":         _float("grad_clip_norm"),
            "target_update_interval": _int("target_update_interval"),
            "batch_size":             _int("batch_size"),
            "replay_capacity":        _int("replay_capacity"),
            "min_replay_size":        _int("min_replay_size"),
            "train_frequency":        _int("train_frequency"),
            "checkpoint_interval":    _int("checkpoint_interval"),
            "log_interval":           _int("log_interval"),
            "tensorboard_log_interval": _int("tensorboard_log_interval"),
            "jsonl_flush_interval":   _int("jsonl_flush_interval"),
            "moving_avg_window":      _int("moving_avg_window"),
            "tensorboard":            _bool("tensorboard"),
            "save_csv":               _bool("save_csv"),
            "save_jsonl":             _bool("save_jsonl"),
            "live_plot":              _bool("live_plot"),
            "lightweight_step_info":  _bool("lightweight_step_info"),
            "env": {
                "board_size":              _int("env.board_size"),
                "difficulty":              _get("env.difficulty"),
                "mode":                    _get("env.mode"),
                "max_steps_without_food":  _int("env.max_steps_without_food"),
                "enable_bonus_food":       _bool("env.enable_bonus_food"),
                "enable_obstacles":        _bool("env.enable_obstacles"),
                "allow_leveling":          _bool("env.allow_leveling"),
                "seed":                    seed_val,
            },
            "reward_weights": {
                "alive":         _float("rw.alive"),
                "food":          _float("rw.food"),
                "bonusFood":     _float("rw.bonusFood"),
                "death":         _float("rw.death"),
                "timeout":       _float("rw.timeout"),
                "levelUp":       _float("rw.levelUp"),
                "victory":       _float("rw.victory"),
                "foodDistanceK": _float("rw.foodDistanceK"),
            },
            # curriculum / random_board / parallel 保留原始值（不在表单里编辑）
            "curriculum":    self._extra.get("curriculum"),
            "random_board":  self._extra.get("random_board"),
            "parallel":      self._extra.get("parallel"),
        }
        return json.dumps(d, ensure_ascii=False, indent=2)

    def load_extra(self, data: dict[str, Any]) -> None:
        """保存 curriculum / random_board / parallel 原始值，不在表单显示，但序列化时保留。"""
        self._extra = {
            "curriculum":   data.get("curriculum"),
            "random_board": data.get("random_board"),
            "parallel":     data.get("parallel"),
        }

    _extra: dict[str, Any] = {}


@contextmanager
def _suppress_stderr_during_tk_init():
    """Suppress noisy libpng iCCP warnings emitted by some Tk builds."""
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


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
        self._estimating_time = False
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

        default_scheme = str(self._gui_state.get("scheme", "custom"))
        if default_scheme not in SCHEME_INFO:
            default_scheme = "custom"

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="训练方案:").pack(side=tk.LEFT)
        self.scheme_var = tk.StringVar(value=default_scheme)
        cb = ttk.Combobox(
            row1,
            textvariable=self.scheme_var,
            values=list(SCHEME_INFO.keys()),  # custom 已排在首位
            state="readonly",
            width=14,
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
        self.estimate_btn = ttk.Button(row2, text="估算训练时间", command=self.estimate_time)
        self.estimate_btn.pack(side=tk.LEFT, padx=(0, 12))

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

        self.custom_config_frame = ttk.LabelFrame(frame, text="自定义配置 (custom) —— 主推方案", padding=6)
        path_row = ttk.Frame(self.custom_config_frame)
        path_row.pack(fill=tk.X)
        ttk.Label(path_row, text="配置文件:").pack(side=tk.LEFT)
        _default_custom_path = str(
            self._gui_state.get("custom_config_path", str(PROJECT_ROOT / "custom_train_config.json"))
        )
        self.custom_config_path_var = tk.StringVar(value=_default_custom_path)
        ttk.Entry(path_row, textvariable=self.custom_config_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(path_row, text="浏览…", command=self._browse_custom_config_path).pack(side=tk.LEFT)

        # 若默认 custom 配置文件不存在，自动用模板创建，避免首次使用时「文件不存在」错误
        _default_custom_path_obj = Path(_default_custom_path)
        if not _default_custom_path_obj.exists():
            try:
                _default_custom_path_obj.parent.mkdir(parents=True, exist_ok=True)
                _default_custom_path_obj.write_text(
                    _train_config_to_json_text(default_custom_train_config()),
                    encoding="utf-8",
                )
            except Exception:
                pass

        btn_row = ttk.Frame(self.custom_config_frame)
        btn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_row, text="从文件加载", command=self._custom_load_from_file).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(btn_row, text="保存到文件", command=self._custom_save_to_file).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(btn_row, text="验证并保存", command=self._custom_apply_to_session).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(btn_row, text="参数说明 ↗", command=self._custom_open_help_doc).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        # 切换「表单视图 ↔ JSON 原文」
        self._form_view_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            btn_row, text="表单视图", variable=self._form_view_var,
            command=self._toggle_config_view,
        ).pack(side=tk.LEFT)

        # ── 表单编辑器（默认显示）──────────────────────────────────────────
        self.custom_form_editor = ConfigFormEditor(self.custom_config_frame)
        self.custom_form_editor.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        # ── JSON 原文编辑框（默认隐藏）────────────────────────────────────
        editor_wrap = ttk.Frame(self.custom_config_frame)
        self.custom_json_text = tk.Text(
            editor_wrap,
            height=14,
            wrap=tk.NONE,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            selectbackground="#264f78",
        )
        vsb_c = ttk.Scrollbar(editor_wrap, orient=tk.VERTICAL, command=self.custom_json_text.yview)
        hsb_c = ttk.Scrollbar(editor_wrap, orient=tk.HORIZONTAL, command=self.custom_json_text.xview)
        self.custom_json_text.configure(yscrollcommand=vsb_c.set, xscrollcommand=hsb_c.set)
        self.custom_json_text.grid(row=0, column=0, sticky="nsew")
        vsb_c.grid(row=0, column=1, sticky="ns")
        hsb_c.grid(row=1, column=0, sticky="ew")
        editor_wrap.grid_rowconfigure(0, weight=1)
        editor_wrap.grid_columnconfigure(0, weight=1)
        self._json_editor_wrap = editor_wrap  # 切换时 pack/forget 用

        # 优先从磁盘文件加载编辑器内容（保证编辑器与文件同步）
        _custom_file_loaded = False
        _custom_path_obj = Path(_default_custom_path)
        if _custom_path_obj.is_file():
            try:
                _init_text = _custom_path_obj.read_text(encoding="utf-8")
                self._set_custom_editor_text(_init_text)
                _custom_file_loaded = True
            except Exception:
                pass
        if not _custom_file_loaded:
            _saved_json = self._gui_state.get("custom_config_json")
            if isinstance(_saved_json, str) and _saved_json.strip():
                self._set_custom_editor_text(_saved_json)
            else:
                self._set_custom_editor_text(_train_config_to_json_text(default_custom_train_config()))

        row4 = ttk.Frame(frame)
        self._train_progress_row = row4
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

        self._refresh_custom_section_visibility()

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

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=(10, 0))
        self.service_status_var = tk.StringVar(value="")
        ttk.Label(row2, textvariable=self.service_status_var, style="Status.TLabel").pack(
            side=tk.LEFT
        )

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
        self._refresh_custom_section_visibility()

    def _refresh_custom_section_visibility(self) -> None:
        if self.scheme_var.get() == "custom":
            self.custom_config_frame.pack(
                fill=tk.BOTH, expand=True, pady=(8, 0), before=self._train_progress_row
            )
        else:
            self.custom_config_frame.pack_forget()

    def _toggle_config_view(self) -> None:
        """在表单视图和 JSON 原文视图之间切换，切换时双向同步数据。"""
        if self._form_view_var.get():
            # JSON → 表单
            raw = self.custom_json_text.get("1.0", tk.END)
            try:
                data = json.loads(raw)
                self.custom_form_editor.load_extra(data)
                self.custom_form_editor.set_json_text(raw)
            except Exception:
                pass
            self._json_editor_wrap.pack_forget()
            self.custom_form_editor.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        else:
            # 表单 → JSON
            raw = self.custom_form_editor.get_json_text()
            self._raw_set_json_text(raw)
            self.custom_form_editor.pack_forget()
            self._json_editor_wrap.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _raw_set_json_text(self, text: str) -> None:
        """直接写入 JSON 文本框（不经过表单同步）。"""
        self.custom_json_text.config(state=tk.NORMAL)
        self.custom_json_text.delete("1.0", tk.END)
        self.custom_json_text.insert("1.0", text)
        self.custom_json_text.config(state=tk.NORMAL)

    def _set_custom_editor_text(self, text: str) -> None:
        """加载 JSON 文本：同时刷新表单和文本框两侧。"""
        self._raw_set_json_text(text)
        try:
            data = json.loads(text)
            self.custom_form_editor.load_extra(data)
            self.custom_form_editor.set_json_text(text)
        except Exception:
            pass

    def _get_custom_editor_text(self) -> str:
        """读取当前编辑内容：表单视图时从表单序列化，否则直接取文本框。"""
        if self._form_view_var.get():
            return self.custom_form_editor.get_json_text()
        return self.custom_json_text.get("1.0", tk.END)

    def _parse_and_validate_custom_json(self, raw: str) -> TrainConfig:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON 根必须是对象")
        cfg = train_config_from_dict(data)
        validate_config(cfg)
        return cfg

    def _browse_custom_config_path(self) -> None:
        initial = self.custom_config_path_var.get().strip() or str(PROJECT_ROOT / "custom_train_config.json")
        initial_dir = str(Path(initial).parent) if Path(initial).parent.is_dir() else str(PROJECT_ROOT)
        path = filedialog.askopenfilename(
            title="选择 TrainConfig JSON",
            initialdir=initial_dir,
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.custom_config_path_var.set(path)

    def _custom_load_from_file(self) -> None:
        p = Path(self.custom_config_path_var.get().strip())
        if not p.is_file():
            messagebox.showerror("错误", f"文件不存在:\n{p}")
            return
        try:
            text = p.read_text(encoding="utf-8")
            cfg = self._parse_and_validate_custom_json(text)
            self._set_custom_editor_text(_train_config_to_json_text(cfg))
            self.log(f"[{self._ts()}] 已从文件加载自定义配置: {p}")
        except Exception as exc:
            messagebox.showerror("错误", f"加载失败:\n{exc}")

    def _custom_save_to_file(self) -> None:
        p = Path(self.custom_config_path_var.get().strip())
        if not p.parent.is_dir():
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                messagebox.showerror("错误", f"无法创建目录:\n{exc}")
                return
        try:
            cfg = self._parse_and_validate_custom_json(self._get_custom_editor_text())
            p.write_text(_train_config_to_json_text(cfg), encoding="utf-8")
            self.log(f"[{self._ts()}] 已保存自定义配置到: {p}")
        except Exception as exc:
            messagebox.showerror("错误", f"保存失败:\n{exc}")

    def _custom_apply_to_session(self) -> None:
        """校验编辑器中的 JSON 并写入磁盘文件（等同于"验证并保存到文件"）。"""
        p = Path(self.custom_config_path_var.get().strip())
        if not p.parent.is_dir():
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                messagebox.showerror("错误", f"无法创建目录:\n{exc}")
                return
        try:
            cfg = self._parse_and_validate_custom_json(self._get_custom_editor_text())
            normalized = _train_config_to_json_text(cfg)
            p.write_text(normalized, encoding="utf-8")
            self._set_custom_editor_text(normalized)
            self.log(
                f"[{self._ts()}] 自定义配置已校验并写入文件: {p}"
            )
            self._save_gui_state()
        except Exception as exc:
            messagebox.showerror("错误", f"验证/保存失败:\n{exc}")

    def _custom_open_help_doc(self) -> None:
        """用浏览器打开参数说明文档。"""
        doc_path = PROJECT_ROOT / "docs" / "custom-train-config.html"
        if doc_path.exists():
            import urllib.parse
            url = "file:///" + urllib.parse.quote(str(doc_path).replace("\\", "/"), safe=":/")
            webbrowser.open(url)
            self.log(f"[{self._ts()}] 已在浏览器中打开参数说明文档")
        else:
            messagebox.showinfo("参数说明", "文档文件不存在，请查看项目根目录下的 docs/custom-train-config.html")

    @staticmethod
    def _as_int(value: Any, default: int, min_v: int, max_v: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(min_v, min(max_v, parsed))

    def _current_training_params(self) -> tuple[str, bool, int, int]:
        scheme = self.scheme_var.get().strip()
        if scheme not in SCHEME_INFO:
            scheme = "scheme1"
        use_parallel = bool(self.parallel_var.get())
        workers = self._as_int(self.parallel_workers_var.get(), default=4, min_v=1, max_v=64)
        sync_interval = self._as_int(
            self.parallel_sync_var.get(),
            default=512,
            min_v=1,
            max_v=100000,
        )
        return scheme, use_parallel, workers, sync_interval

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
            "monitor_port": int(self.monitor_port_var.get()),
            "inference_port": int(self.inference_port_var.get()),
            "custom_config_path": self.custom_config_path_var.get().strip(),
            # 不再存储完整 JSON 文本（防止状态文件膨胀和内容漂移）
            # 编辑器内容以磁盘文件为唯一来源，重新打开时从文件加载
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

    def _prepare_progress_context(self, scheme: str) -> None:
        self._progress_total_episodes = 0
        self._progress_stage_prefix.clear()
        self._progress_current_episode = 0
        self._progress_stage_text = "-"
        self._progress_last_avg_reward = None
        self._progress_last_epsilon = None
        try:
            if scheme == "custom":
                p = Path(self.custom_config_path_var.get().strip())
                cfg = get_config(scheme="custom", custom_config_path=p)
            else:
                cfg = get_config(scheme=scheme)
            if cfg.curriculum is not None and cfg.curriculum.stages:
                total = 0
                for idx, stage in enumerate(cfg.curriculum.stages, start=1):
                    self._progress_stage_prefix[idx] = total
                    total += int(stage.episodes)
                self._progress_total_episodes = max(0, total)
            else:
                self._progress_total_episodes = max(0, int(cfg.episodes))
        except Exception as exc:
            self._progress_total_episodes = 0
            self.log(f"[{self._ts()}] 警告：读取配置计算总局数失败（进度将显示 ?）: {exc}")
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

        scheme, use_parallel, workers, sync_interval = self._current_training_params()
        if scheme == "custom":
            cp = Path(self.custom_config_path_var.get().strip())
            if not cp.is_file():
                messagebox.showerror(
                    "错误",
                    "custom 模式需要磁盘上存在有效的 JSON 配置文件。\n"
                    "请填写路径后点击「验证并保存」或「保存到文件」。",
                )
                return
            # 启动前完整校验 JSON 内容，尽早报错避免子进程启动后才失败
            try:
                text = cp.read_text(encoding="utf-8")
                self._parse_and_validate_custom_json(text)
            except Exception as exc:
                messagebox.showerror(
                    "配置错误",
                    f"custom 配置文件校验失败，无法启动训练：\n{exc}\n\n"
                    "请在编辑器中修正后点击「验证并保存」再重试。",
                )
                return
        self._prepare_progress_context(scheme)
        cmd = [sys.executable, "-m", "snake_rl.cli", "train", "--scheme", scheme]
        if scheme == "custom":
            cmd.extend(["--custom-config", str(Path(self.custom_config_path_var.get().strip()))])
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

    def _dashboard_url(self, _run_name: str | None) -> str:
        base, _port = self._monitor_base_url()
        return f"{base}/"

    def _wait_monitor_then_open(self, run_name: str | None, attempt: int = 0) -> None:
        base, _port = self._monitor_base_url()
        health = f"{base}/"
        dashboard_url = self._dashboard_url(run_name)
        if self._http_ok(health):
            self.log(f"[{self._ts()}] TensorBoard 已就绪")
            if run_name:
                self.log(f"[{self._ts()}] 在 TensorBoard 中选择 run（运行目录名）: {run_name}")
            webbrowser.open(dashboard_url)
            return
        if self.monitor_proc is not None and self.monitor_proc.poll() is not None:
            self.log(f"[{self._ts()}] TensorBoard 进程已退出，请查看 [monitor] 日志")
            return
        if attempt >= 50:
            msg = "等待 TensorBoard 启动超时，未自动打开浏览器；请查看 [monitor] 日志后重试。"
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
        health = f"{base}/"
        dashboard_url = self._dashboard_url(run_name)

        if self.monitor_proc is not None and self.monitor_proc.poll() is None:
            self.log("TensorBoard（由本 GUI 启动）已在运行，打开浏览器...")
            if self._http_ok(health):
                if run_name:
                    self.log(f"[{self._ts()}] 在 TensorBoard 中选择 run: {run_name}")
                webbrowser.open(dashboard_url)
            else:
                self._wait_monitor_then_open(run_name)
            return

        if self._tcp_port_open("127.0.0.1", port) and self._http_ok(health):
            self.log(f"检测到端口 {port} 已有 TensorBoard（或 Web 服务），直接打开浏览器")
            if run_name:
                self.log(f"[{self._ts()}] 在 TensorBoard 中选择 run: {run_name}")
            webbrowser.open(dashboard_url)
            return

        if self._tcp_port_open("127.0.0.1", port) and not self._http_ok(health):
            messagebox.showwarning(
                "提示",
                f"端口 {port} 已被占用，但未能识别为可访问的 Web 服务。\n"
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

        self.log(f"[{self._ts()}] 已启动 TensorBoard，等待就绪 ...")
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
            self.log(f"[{self._ts()}] TensorBoard 已停止")
            self._update_service_status()

    def open_game(self) -> None:
        game = WEB_DIR / "index.html"
        if game.exists():
            webbrowser.open(game.as_uri())
        else:
            messagebox.showerror("错误", "找不到 web/index.html")

    def estimate_time(self) -> None:
        if self._estimating_time:
            messagebox.showinfo("提示", "已有估算任务在进行中，请稍候。")
            return

        scheme, use_parallel, workers, sync_interval = self._current_training_params()
        if scheme == "custom":
            cp = Path(self.custom_config_path_var.get().strip())
            if not cp.is_file():
                messagebox.showerror(
                    "错误",
                    "custom 模式估算需要有效的配置文件。\n"
                    "请先「验证并保存」或「保存到文件」生成 JSON 文件。",
                )
                return
        mode_text = f"{scheme} | {'并行' if use_parallel else '串行'}"
        if use_parallel:
            mode_text += f" | workers={workers} | sync={sync_interval}"
        if scheme == "custom":
            mode_text += f" | config={self.custom_config_path_var.get().strip()}"
        self.log(f"[{self._ts()}] 正在估算训练时间: {mode_text}")
        self._estimating_time = True
        self.estimate_btn.config(state=tk.DISABLED)

        def _run() -> None:
            try:
                cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "snake_rl.cli",
                    "estimate",
                    "--scheme",
                    scheme,
                ]
                if scheme == "custom":
                    cmd.extend(
                        ["--custom-config", str(Path(self.custom_config_path_var.get().strip()))]
                    )
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
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                assert proc.stdout is not None
                code = 0
                try:
                    for line in proc.stdout:
                        stripped = line.rstrip("\n\r")
                        if stripped:
                            self.root.after(0, self.log, stripped)
                    code = proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    terminate_process(proc)
                    self.root.after(0, self._on_estimate_done, -1, "估算超时（超过 10 分钟）")
                    return
                except Exception:
                    terminate_process(proc)
                    raise
                self.root.after(0, self._on_estimate_done, code)
            except subprocess.TimeoutExpired:
                self.root.after(0, self._on_estimate_done, -1, "估算超时（超过 10 分钟）")
            except Exception as exc:
                self.root.after(0, self._on_estimate_done, -1, f"估算失败: {exc}")

        threading.Thread(target=_run, daemon=True).start()

    def _on_estimate_done(self, code: int, extra: str = "") -> None:
        self._estimating_time = False
        self.estimate_btn.config(state=tk.NORMAL)
        if extra:
            self.log(extra)
        if code != 0 and not extra:
            self.log(f"[{self._ts()}] 估算子进程退出码: {code}")
        self.log(f"[{self._ts()}] --- 训练时间估算结束 ---")

    def on_close(self) -> None:
        self._save_gui_state()
        for proc in (self.training_proc, self.monitor_proc, self.inference_proc):
            if proc is not None and proc.poll() is None:
                terminate_process(proc, timeout_s=2.0)
        self.root.destroy()


def main() -> None:
    # On some Windows Python/Tk builds, root initialization may print many
    # harmless libpng iCCP warnings to stderr.
    with _suppress_stderr_during_tk_init():
        root = tb.Window(themename=DEFAULT_THEME) if HAS_TTKBOOTSTRAP else tk.Tk()
    app = TrainingManager(root)
    # First idle render may trigger the same warnings; warm up once silently.
    with _suppress_stderr_during_tk_init():
        root.update_idletasks()
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
