"use strict";

window.SnakeConsole = window.SnakeConsole || {};

SnakeConsole.DEFAULT_CURRICULUM = {
  carry_replay: false,
  scale_timeout: true,
  stages: [
    {
      board_size: 10,
      episodes: 2000,
      max_steps_without_food: 100,
      epsilon_start: 1.0,
      epsilon_end: 0.05,
      epsilon_decay_steps: 50000,
      replay_capacity: 15000,
      min_replay_size: 1500,
      promotion_threshold_foods: 0,
      promotion_window: 100,
      promotion_min_episodes: 200,
    },
  ],
};

SnakeConsole.DEFAULT_RANDOM_BOARD = {
  board_sizes: [8, 10, 12, 16],
  weights: null,
  max_steps_scale: 1.0,
};

SnakeConsole.ensureConfigShape = function ensureConfigShape(c) {
  if (!c.env) c.env = {};
  if (!c.reward_weights) c.reward_weights = {};
  if (c.curriculum === undefined) c.curriculum = null;
  if (c.random_board === undefined) c.random_board = null;
  if (!c.parallel) {
    c.parallel = {
      enabled: false,
      num_workers: 4,
      queue_capacity: 8192,
      weight_sync_interval_steps: 512,
      actor_loop_sleep_ms: 0,
      actor_seed_stride: 100000,
      actor_device: "cpu",
    };
  }
};

SnakeConsole.createFieldAccessors = function createFieldAccessors(config) {
  function getField(key) {
    if (key === "curriculum_enabled") return config.curriculum != null;
    if (key === "random_board_enabled") return config.random_board != null;
    if (key.startsWith("rw.")) return config.reward_weights[key.slice(3)];
    if (key.startsWith("env.")) return config.env[key.slice(4)];
    if (key.startsWith("curriculum.")) {
      if (!config.curriculum) return key === "curriculum.stages" ? [] : null;
      return config.curriculum[key.slice("curriculum.".length)];
    }
    if (key.startsWith("random_board.")) {
      if (!config.random_board) return key === "random_board.board_sizes" ? [] : null;
      return config.random_board[key.slice("random_board.".length)];
    }
    return config[key];
  }

  function setField(key, val) {
    if (key === "curriculum_enabled") {
      config.curriculum = val
        ? JSON.parse(JSON.stringify(config.curriculum || SnakeConsole.DEFAULT_CURRICULUM))
        : null;
      if (val) config.random_board = null;
      return;
    }
    if (key === "random_board_enabled") {
      config.random_board = val
        ? JSON.parse(JSON.stringify(config.random_board || SnakeConsole.DEFAULT_RANDOM_BOARD))
        : null;
      if (val) config.curriculum = null;
      return;
    }
    if (key.startsWith("rw.")) config.reward_weights[key.slice(3)] = val;
    else if (key.startsWith("env.")) config.env[key.slice(4)] = val;
    else if (key.startsWith("curriculum.")) {
      if (!config.curriculum) config.curriculum = JSON.parse(JSON.stringify(SnakeConsole.DEFAULT_CURRICULUM));
      config.curriculum[key.slice("curriculum.".length)] = val;
      config.random_board = null;
    } else if (key.startsWith("random_board.")) {
      if (!config.random_board) config.random_board = JSON.parse(JSON.stringify(SnakeConsole.DEFAULT_RANDOM_BOARD));
      config.random_board[key.slice("random_board.".length)] = val;
      config.curriculum = null;
    } else {
      config[key] = val;
    }
  }

  function fieldStr(key) {
    const v = getField(key);
    if (v === null || v === undefined) return "";
    if (Array.isArray(v)) return v.join(",");
    if (typeof v === "object") return JSON.stringify(v, null, 2);
    return String(v);
  }

  function parseCsvNumbers(raw, asFloat) {
    const text = String(raw || "").trim();
    if (!text) return [];
    return text
      .split(/[,，\s]+/)
      .filter(Boolean)
      .map((part) => (asFloat ? parseFloat(part) : parseInt(part, 10)))
      .filter((n) => !isNaN(n));
  }

  function setFieldCoerce(key, raw, type) {
    if (type === "number") {
      if (raw === "" || raw === null || raw === undefined) return;
      const n = parseInt(raw, 10);
      if (!isNaN(n)) setField(key, n);
    } else if (type === "float") {
      if (raw === "" || raw === null || raw === undefined) return;
      const n = parseFloat(raw);
      if (!isNaN(n)) setField(key, n);
    } else if (type === "json") {
      try {
        setField(key, JSON.parse(raw));
      } catch (_) {
        /* keep editing */
      }
    } else if (type === "csv_int") {
      setField(key, parseCsvNumbers(raw, false));
    } else if (type === "csv_float") {
      const nums = parseCsvNumbers(raw, true);
      setField(key, nums.length ? nums : null);
    } else if (key === "env.seed") {
      setField(key, raw === "" ? null : isNaN(Number(raw)) ? null : Number(raw));
    } else {
      setField(key, raw);
    }
  }

  return { getField, setField, fieldStr, setFieldCoerce };
};
