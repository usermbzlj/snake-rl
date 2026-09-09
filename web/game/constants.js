"use strict";

const DIRS = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
};

const OPPOSITE = {
  up: "down",
  down: "up",
  left: "right",
  right: "left",
};

const DIRECTION_ORDER = ["up", "right", "down", "left"];
const DIRECTION_INDEX = {
  up: 0,
  right: 1,
  down: 2,
  left: 3,
};

const AGENT_ACTIONS = Object.freeze({
  STRAIGHT: 0,
  TURN_LEFT: 1,
  TURN_RIGHT: 2,
});

const AGENT_OBSERVATION_CHANNELS = Object.freeze([
  "snakeHead",
  "snakeBody",
  "food",
  "bonusFood",
  "obstacle",
  "dirUp",
  "dirRight",
  "dirDown",
  "dirLeft",
]);

const TINY_FEAT_DIM = 10;

const TINY_RAYS = Object.freeze({
  up:    [[0,-1], [-1,-1], [1,-1], [-1,0], [1,0], [-1,1], [1,1]],
  right: [[1,0],  [1,-1],  [1,1],  [0,-1], [0,1], [-1,-1],[-1,1]],
  down:  [[0,1],  [1,1],   [-1,1], [1,0],  [-1,0],[1,-1], [-1,-1]],
  left:  [[-1,0], [-1,1],  [-1,-1],[0,1],  [0,-1],[1,1],  [1,-1]],
});

const AGENT_API_VERSION = "2.0.0";

const DEFAULT_AGENT_ENV_CONFIG = Object.freeze({
  difficulty: "normal",
  mode: "classic",
  boardSize: 22,
  enableBonusFood: true,
  enableObstacles: true,
  allowLeveling: true,
  maxStepsWithoutFood: 0,
});

const DEFAULT_REWARD_WEIGHTS = Object.freeze({
  alive: -0.01,
  food: 1.0,
  bonusFood: 1.5,
  death: -1.5,
  timeout: -1.0,
  levelUp: 0.2,
  victory: 5.0,
  foodDistanceK: 0.4,
});

const TERMINAL_REASONS = Object.freeze({
  WALL: "wall",
  OBSTACLE: "obstacle",
  SELF: "self",
  BOARD_FULL: "board_full",
  TIMEOUT: "timeout",
  NOT_RUNNING: "not_running",
});

const TERMINAL_REASON_LABEL = Object.freeze({
  wall: "撞墙了",
  obstacle: "撞到障碍物了",
  self: "咬到自己了",
  board_full: "地图已被你填满，完美通关",
  timeout: "长时间未吃到食物，回合结束",
  not_running: "当前回合已结束",
});

const DIFFICULTY_CONFIG = {
  easy: {
    baseTick: 180,
    perLevelFaster: 4,
    minTick: 105,
    levelStepByFoods: 7,
    bonusChance: 0.28,
    maxObstacles: 8,
    bonusLifeMs: 9000,
  },
  normal: {
    baseTick: 145,
    perLevelFaster: 5,
    minTick: 90,
    levelStepByFoods: 6,
    bonusChance: 0.32,
    maxObstacles: 12,
    bonusLifeMs: 8000,
  },
  hard: {
    baseTick: 120,
    perLevelFaster: 6,
    minTick: 78,
    levelStepByFoods: 5,
    bonusChance: 0.35,
    maxObstacles: 16,
    bonusLifeMs: 7200,
  },
  expert: {
    baseTick: 98,
    perLevelFaster: 6,
    minTick: 68,
    levelStepByFoods: 4,
    bonusChance: 0.4,
    maxObstacles: 20,
    bonusLifeMs: 6500,
  },
};

const sleep = (ms) =>
  new Promise((resolve) => {
    window.setTimeout(resolve, Math.max(0, ms));
  });

function isTypingInFormControl(target) {
  if (!target || typeof target !== "object") {
    return false;
  }
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
    return true;
  }
  return Boolean(target.isContentEditable);
}
