"use strict";

window.SnakeConsole = window.SnakeConsole || {};

SnakeConsole.formatDuration = function formatDuration(seconds) {
  const sec = Number(seconds);
  if (!Number.isFinite(sec) || sec < 0) return "";
  if (sec < 60) return Math.max(1, Math.round(sec)) + " 秒";
  if (sec < 3600) return Math.round(sec / 60) + " 分钟";
  const h = Math.floor(sec / 3600);
  const m = Math.round((sec % 3600) / 60);
  return m ? h + " 小时 " + m + " 分钟" : h + " 小时";
};

SnakeConsole.formatMetric = function formatMetric(value, digits) {
  if (value === null || value === undefined || value === "" || value === "-") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return n.toFixed(digits == null ? 3 : digits);
};

SnakeConsole.logKind = function logKind(text) {
  const s = String(text || "");
  if (/失败|错误|Error|Traceback|Exception|exit [1-9]/i.test(s)) return "error";
  if (/\[Episode|avg_reward=|SNAKE_EVENT|训练完成|已启动/.test(s)) return "progress";
  return "info";
};

SnakeConsole.canDemo = function canDemo(run) {
  if (!run) return false;
  if (run.can_demo === true) return true;
  if (run.has_best_checkpoint || run.has_latest_checkpoint) return true;
  return Boolean(run.badges && run.badges !== "-");
};

SnakeConsole.canResume = function canResume(run) {
  if (!run) return false;
  if (run.can_resume === true) return true;
  return Boolean(run.has_training_state);
};

SnakeConsole.copyText = async function copyText(text) {
  const value = String(text || "");
  if (!value) throw new Error("没有可复制的内容");
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const el = document.createElement("textarea");
  el.value = value;
  el.setAttribute("readonly", "");
  el.style.position = "fixed";
  el.style.left = "-9999px";
  document.body.appendChild(el);
  el.select();
  document.execCommand("copy");
  document.body.removeChild(el);
};

SnakeConsole.isTypingTarget = function isTypingTarget(el) {
  if (!el) return false;
  const tag = (el.tagName || "").toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
};

SnakeConsole.stampName = function stampName(prefix) {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    prefix +
    "_" +
    d.getFullYear() +
    pad(d.getMonth() + 1) +
    pad(d.getDate()) +
    "_" +
    pad(d.getHours()) +
    pad(d.getMinutes())
  );
};

SnakeConsole.ESSENTIAL_KEYS = new Set([
  "run_name",
  "device",
  "episodes",
  "model_type",
  "env.board_size",
  "env.mode",
  "env.difficulty",
  "learning_rate",
]);

SnakeConsole.PRESETS = [
  {
    id: "trial",
    title: "5 分钟试跑",
    desc: "tiny 网络 + 8×8 小图，用来确认训练流程能跑通。",
    scheme: "custom",
    apply(config) {
      config.model_type = "tiny";
      config.episodes = 800;
      config.max_steps_per_episode = 400;
      config.run_name = SnakeConsole.stampName("trial");
      config.curriculum = null;
      config.random_board = null;
      if (!config.env) config.env = {};
      config.env.board_size = 8;
      config.env.max_steps_without_food = 64;
      config.env.enable_bonus_food = false;
      config.env.enable_obstacles = false;
      config.env.allow_leveling = false;
    },
  },
  {
    id: "standard",
    title: "标准训练",
    desc: "adaptive_cnn + 12×12，适合大多数第一次认真训练。",
    scheme: "custom",
    apply(config) {
      config.model_type = "adaptive_cnn";
      config.episodes = 20000;
      config.max_steps_per_episode = 1500;
      config.run_name = SnakeConsole.stampName("std");
      config.curriculum = null;
      config.random_board = null;
      if (!config.env) config.env = {};
      config.env.board_size = 12;
      config.env.max_steps_without_food = 144;
    },
  },
  {
    id: "long",
    title: "长训泛化",
    desc: "使用内置 scheme4：课程学习 + 随机地图 + hybrid。",
    scheme: "scheme4",
    apply() {},
  },
];
