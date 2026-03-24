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
  death: -1.0,
  timeout: -0.6,
  levelUp: 0.2,
  victory: 2.0,
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

class SnakeGame {
  constructor() {
    this.canvas = document.getElementById("gameCanvas");
    this.ctx = this.canvas.getContext("2d");

    this.overlay = document.getElementById("overlay");
    this.overlayTitle = document.getElementById("overlayTitle");
    this.overlayText = document.getElementById("overlayText");

    this.startBtn = document.getElementById("startBtn");
    this.pauseBtn = document.getElementById("pauseBtn");
    this.difficultySelect = document.getElementById("difficulty");
    this.modeSelect = document.getElementById("mode");
    this.boardSizeSelect = document.getElementById("boardSize");

    this.scoreValue = document.getElementById("scoreValue");
    this.bestScoreValue = document.getElementById("bestScoreValue");
    this.lengthValue = document.getElementById("lengthValue");
    this.levelValue = document.getElementById("levelValue");
    this.speedValue = document.getElementById("speedValue");

    this.pixelPerCell = 32;
    this.state = "ready";
    this.rafId = 0;
    this.lastFrameTime = 0;
    this.accumulator = 0;
    this.randomSeed = null;
    this.randomState = 0;
    this.randomFn = Math.random;
    this.agentControlled = false;
    this.renderEnabled = true;
    this.isClosed = false;
    this.lastTerminalReason = "";
    this.lastTransition = null;
    this.agentConfig = { ...DEFAULT_AGENT_ENV_CONFIG };
    this.episodeIndex = 0;
    this.episodeStats = this.createEmptyEpisodeStats();
    this.agentHooks = {
      reset: new Set(),
      transition: new Set(),
      done: new Set(),
    };

    this.rewardWeights = { ...DEFAULT_REWARD_WEIGHTS };
    this.bestScore = this.loadNumberFromStorage("snake.bestScore");
    this.bestLength = this.loadNumberFromStorage("snake.bestLength");

    this.applySettingsFromUI();
    this.resetRuntime();
    this.resetEpisodeStats();
    this.bindEvents();
    this.updateHUD();
    this.render(performance.now());
  }

  bindEvents() {
    this.boundStartClick = () => this.startNewGame();
    this.boundPauseClick = () => this.togglePause();
    this.boundKeydown = (event) => this.onKeyDown(event);

    this.startBtn.addEventListener("click", this.boundStartClick);
    this.pauseBtn.addEventListener("click", this.boundPauseClick);
    document.addEventListener("keydown", this.boundKeydown);

    this.touchButtonHandlers = [];
    document.querySelectorAll(".touch-controls button").forEach((button) => {
      const handler = () => {
        const turn = button.dataset.turn;
        this.queueRelativeTurn(turn);
      };
      button.addEventListener("click", handler);
      this.touchButtonHandlers.push({ button, handler });
    });
  }

  unbindEvents() {
    if (this.boundStartClick) {
      this.startBtn.removeEventListener("click", this.boundStartClick);
    }
    if (this.boundPauseClick) {
      this.pauseBtn.removeEventListener("click", this.boundPauseClick);
    }
    if (this.boundKeydown) {
      document.removeEventListener("keydown", this.boundKeydown);
    }

    if (Array.isArray(this.touchButtonHandlers)) {
      this.touchButtonHandlers.forEach(({ button, handler }) => {
        button.removeEventListener("click", handler);
      });
    }
    this.touchButtonHandlers = [];
  }

  onKeyDown(event) {
    if (event.code === "KeyA" || event.code === "ArrowLeft") {
      event.preventDefault();
      this.queueRelativeTurn("left");
      return;
    }

    if (event.code === "KeyD" || event.code === "ArrowRight") {
      event.preventDefault();
      this.queueRelativeTurn("right");
      return;
    }

    if (event.code === "Space" || event.code === "KeyP") {
      event.preventDefault();
      this.togglePause();
      return;
    }

    if (event.code === "Enter") {
      event.preventDefault();
      if (this.state !== "running") {
        this.startNewGame();
      }
      return;
    }

    if (event.code === "KeyR") {
      event.preventDefault();
      this.startNewGame();
    }
  }

  applySettingsFromUI() {
    const config = this.sanitizeAgentConfig({
      difficulty: this.difficultySelect.value,
      mode: this.modeSelect.value,
      boardSize: Number(this.boardSizeSelect.value),
      enableBonusFood: true,
      enableObstacles: true,
      allowLeveling: true,
      maxStepsWithoutFood: 0,
    });
    this.applyRuntimeConfig(config, { syncUI: false });
    this.agentConfig = { ...config };
  }

  normalizeBoardSize(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return DEFAULT_AGENT_ENV_CONFIG.boardSize;
    }
    return Math.min(64, Math.max(8, Math.round(num)));
  }

  sanitizeAgentConfig(inputConfig = {}) {
    const next = {
      ...DEFAULT_AGENT_ENV_CONFIG,
      ...this.agentConfig,
      ...inputConfig,
    };

    next.difficulty = DIFFICULTY_CONFIG[next.difficulty]
      ? next.difficulty
      : DEFAULT_AGENT_ENV_CONFIG.difficulty;
    next.mode = next.mode === "wrap" ? "wrap" : "classic";
    next.boardSize = this.normalizeBoardSize(next.boardSize);
    next.enableBonusFood = Boolean(next.enableBonusFood);
    next.enableObstacles = Boolean(next.enableObstacles);
    next.allowLeveling = Boolean(next.allowLeveling);

    const timeout = Number(next.maxStepsWithoutFood);
    next.maxStepsWithoutFood =
      Number.isFinite(timeout) && timeout > 0 ? Math.round(timeout) : 0;

    return next;
  }

  applyRuntimeConfig(config, options = {}) {
    const { syncUI = false } = options;
    this.boardSize = config.boardSize;
    this.gameMode = config.mode;
    this.difficulty = config.difficulty;
    this.config = DIFFICULTY_CONFIG[this.difficulty];

    this.canvas.width = this.boardSize * this.pixelPerCell;
    this.canvas.height = this.boardSize * this.pixelPerCell;
    this.cellSize = this.pixelPerCell;

    if (syncUI) {
      this.difficultySelect.value = this.difficulty;
      this.modeSelect.value = this.gameMode;

      const boardValue = String(this.boardSize);
      const matched = Array.from(this.boardSizeSelect.options).some(
        (option) => option.value === boardValue
      );
      if (matched) {
        this.boardSizeSelect.value = boardValue;
      }
    }
  }

  setSettingsLocked(locked) {
    this.difficultySelect.disabled = locked;
    this.modeSelect.disabled = locked;
    this.boardSizeSelect.disabled = locked;
  }

  createEmptyEpisodeStats() {
    return {
      episode: this.episodeIndex,
      steps: 0,
      totalReward: 0,
      foods: 0,
      bonusFoods: 0,
      levelUps: 0,
      maxLength: 0,
      scoreStart: 0,
      scoreEnd: 0,
      done: false,
      terminalReason: "",
      terminalReasonLabel: "",
      startedAtMs: Date.now(),
      endedAtMs: 0,
    };
  }

  resetEpisodeStats() {
    this.episodeStats = this.createEmptyEpisodeStats();
    this.episodeStats.scoreStart = this.score;
    this.episodeStats.scoreEnd = this.score;
    this.episodeStats.maxLength = this.snake.length;
  }

  resetRuntime() {
    const center = Math.floor(this.boardSize / 2);
    this.snake = [
      { x: center, y: center },
      { x: center - 1, y: center },
      { x: center - 2, y: center },
    ];

    this.direction = "right";
    this.directionQueue = [];
    this.score = 0;
    this.level = 1;
    this.foodsEaten = 0;
    this.obstacles = new Set();
    this.food = null;
    this.bonusFood = null;
    this.flashText = "";
    this.flashUntil = 0;
    this.flashHighlight = false;
    this.lastTerminalReason = "";
    this.lastTransition = null;
    this.stepsSinceLastFood = 0;

    this.spawnFood();
    this.updateHUD();
  }

  startNewGame() {
    if (this.isClosed) {
      return;
    }

    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }

    this.agentControlled = false;
    this.applySettingsFromUI();
    this.resetRuntime();
    this.state = "running";
    this.episodeIndex += 1;
    this.resetEpisodeStats();
    this.lastFrameTime = performance.now();
    this.accumulator = 0;

    this.setSettingsLocked(true);
    this.pauseBtn.disabled = false;
    this.pauseBtn.textContent = "暂停";
    this.setOverlay(false);

    this.rafId = requestAnimationFrame((ts) => this.gameLoop(ts));
  }

  togglePause() {
    if (this.isClosed) {
      return;
    }

    if (this.agentControlled) {
      return;
    }

    if (this.state !== "running" && this.state !== "paused") {
      return;
    }

    if (this.state === "running") {
      this.state = "paused";
      this.pauseBtn.textContent = "继续";
      this.setOverlay(true, "已暂停", "按空格 / P 或点击「继续」返回游戏");
      return;
    }

    this.state = "running";
    this.pauseBtn.textContent = "暂停";
    this.setOverlay(false);
    this.lastFrameTime = performance.now();
    this.accumulator = 0;

    if (!this.rafId) {
      this.rafId = requestAnimationFrame((ts) => this.gameLoop(ts));
    }
  }

  gameLoop(timestamp) {
    if (this.isClosed) {
      this.rafId = 0;
      return;
    }

    if (this.state !== "running" || this.agentControlled) {
      this.rafId = 0;
      return;
    }

    let delta = timestamp - this.lastFrameTime;
    this.lastFrameTime = timestamp;

    if (delta > 250) {
      delta = 250;
    }

    this.accumulator += delta;
    const tickDuration = this.getTickDuration();

    while (this.accumulator >= tickDuration && this.state === "running") {
      this.accumulator -= tickDuration;
      const transition = this.advanceOneTick(timestamp, {
        useQueuedDirection: true,
        autoFinalize: true,
        silentGameOver: false,
      });
      if (transition.done) {
        break;
      }
    }

    if (this.renderEnabled) {
      this.render(timestamp);
    }

    if (this.state === "running" && !this.agentControlled) {
      this.rafId = requestAnimationFrame((ts) => this.gameLoop(ts));
    } else {
      this.rafId = 0;
    }
  }

  advanceOneTick(now, options = {}) {
    const {
      useQueuedDirection = false,
      agentAction,
      autoFinalize = true,
      silentGameOver = false,
    } = options;

    const transition = {
      observation: null,
      reward: this.rewardWeights.alive,
      done: false,
      info: {
        step: this.episodeStats.steps + 1,
        action: AGENT_ACTIONS.STRAIGHT,
        ateFood: false,
        ateBonusFood: false,
        levelUp: false,
        scoreGain: 0,
        scoreBefore: this.score,
        scoreAfter: this.score,
        lengthBefore: this.snake.length,
        lengthAfter: this.snake.length,
        levelBefore: this.level,
        levelAfter: this.level,
        foodsEaten: this.foodsEaten,
        stepsSinceFoodBefore: this.stepsSinceLastFood,
        stepsSinceFoodAfter: this.stepsSinceLastFood,
        terminalReason: "",
        terminalReasonLabel: "",
      },
    };

    const setTerminal = (reasonCode, extraReward = 0) => {
      if (transition.done) {
        return;
      }
      transition.done = true;
      transition.reward += extraReward;
      transition.info.terminalReason = reasonCode;
      transition.info.terminalReasonLabel = this.getTerminalReasonLabel(reasonCode);
    };

    if (useQueuedDirection) {
      this.consumeDirectionQueue();
    } else if (agentAction !== undefined) {
      transition.info.action = this.applyRelativeActionImmediately(agentAction);
    }

    const head = this.snake[0];
    const move = DIRS[this.direction];
    let nextX = head.x + move.x;
    let nextY = head.y + move.y;

    if (this.gameMode === "wrap") {
      nextX = (nextX + this.boardSize) % this.boardSize;
      nextY = (nextY + this.boardSize) % this.boardSize;
    } else if (!this.inBounds(nextX, nextY)) {
      setTerminal(TERMINAL_REASONS.WALL, this.rewardWeights.death);
    }

    if (!transition.done && this.obstacles.has(this.cellKey(nextX, nextY))) {
      setTerminal(TERMINAL_REASONS.OBSTACLE, this.rewardWeights.death);
    }

    if (!transition.done) {
      const hitFood = this.food && nextX === this.food.x && nextY === this.food.y;
      const hitBonus =
        this.bonusFood &&
        nextX === this.bonusFood.x &&
        nextY === this.bonusFood.y;
      const willGrow = hitFood || Boolean(hitBonus);

      const bodyForCollision = willGrow ? this.snake : this.snake.slice(0, -1);
      const hitSelf = bodyForCollision.some(
        (node) => node.x === nextX && node.y === nextY
      );
      if (hitSelf) {
        setTerminal(TERMINAL_REASONS.SELF, this.rewardWeights.death);
      } else {
        this.snake.unshift({ x: nextX, y: nextY });
        this.stepsSinceLastFood += 1;

        if (hitFood) {
          const eatInfo = this.onEatFood(now);
          transition.info.ateFood = true;
          transition.info.scoreGain += eatInfo.scoreGain;
          transition.reward += this.rewardWeights.food;
          this.stepsSinceLastFood = 0;

          if (eatInfo.levelUp) {
            transition.info.levelUp = true;
            transition.reward += this.rewardWeights.levelUp;
          }

          if (eatInfo.boardFilled) {
            setTerminal(TERMINAL_REASONS.BOARD_FULL, this.rewardWeights.victory);
          }
        } else if (hitBonus) {
          const bonusInfo = this.onEatBonus(now);
          transition.info.ateBonusFood = true;
          transition.info.scoreGain += bonusInfo.scoreGain;
          transition.reward += this.rewardWeights.bonusFood;
          this.stepsSinceLastFood = 0;
        } else {
          this.snake.pop();
        }
      }
    }

    if (
      !transition.done &&
      this.agentControlled &&
      this.agentConfig.maxStepsWithoutFood > 0 &&
      this.stepsSinceLastFood >= this.agentConfig.maxStepsWithoutFood
    ) {
      setTerminal(TERMINAL_REASONS.TIMEOUT, this.rewardWeights.timeout);
    }

    if (this.bonusFood && now >= this.bonusFood.expiresAt) {
      this.bonusFood = null;
    }

    if (this.flashUntil && now > this.flashUntil) {
      this.flashText = "";
      this.flashUntil = 0;
    }

    this.updateHUD();
    transition.info.scoreAfter = this.score;
    transition.info.lengthAfter = this.snake.length;
    transition.info.levelAfter = this.level;
    transition.info.foodsEaten = this.foodsEaten;
    transition.info.stepsSinceFoodAfter = this.stepsSinceLastFood;
    transition.observation = this.getObservationTensor();

    this.recordEpisodeTransition(transition);
    this.lastTransition = transition;

    if (transition.done && autoFinalize) {
      this.finalizeGameOver(transition.info.terminalReason, {
        silentUI: silentGameOver,
      });
    }

    return transition;
  }

  consumeDirectionQueue() {
    if (!this.directionQueue.length) {
      return;
    }

    const current = this.direction;
    while (this.directionQueue.length) {
      const candidate = this.directionQueue.shift();
      if (candidate !== current && candidate !== OPPOSITE[current]) {
        this.direction = candidate;
        return;
      }
    }
  }

  queueDirection(direction) {
    if (!DIRS[direction] || this.state !== "running") {
      return;
    }

    const lastQueued =
      this.directionQueue.length > 0
        ? this.directionQueue[this.directionQueue.length - 1]
        : this.direction;

    if (direction === lastQueued || direction === OPPOSITE[lastQueued]) {
      return;
    }

    if (this.directionQueue.length < 3) {
      this.directionQueue.push(direction);
    }
  }

  queueRelativeTurn(turn) {
    if (this.state !== "running") {
      return;
    }

    const baseDirection =
      this.directionQueue.length > 0
        ? this.directionQueue[this.directionQueue.length - 1]
        : this.direction;
    const targetDirection = this.rotateDirection(baseDirection, turn);
    this.queueDirection(targetDirection);
  }

  rotateDirection(direction, turn) {
    const index = DIRECTION_ORDER.indexOf(direction);
    if (index < 0) {
      return direction;
    }
    const delta = turn === "left" ? -1 : 1;
    return DIRECTION_ORDER[(index + delta + DIRECTION_ORDER.length) % DIRECTION_ORDER.length];
  }

  normalizeAgentAction(action) {
    if (
      action === AGENT_ACTIONS.STRAIGHT ||
      action === AGENT_ACTIONS.TURN_LEFT ||
      action === AGENT_ACTIONS.TURN_RIGHT
    ) {
      return action;
    }

    if (typeof action === "string") {
      const value = action.trim().toLowerCase();
      if (value === "0" || value === "straight" || value === "forward") {
        return AGENT_ACTIONS.STRAIGHT;
      }
      if (value === "1" || value === "left" || value === "turn_left") {
        return AGENT_ACTIONS.TURN_LEFT;
      }
      if (value === "2" || value === "right" || value === "turn_right") {
        return AGENT_ACTIONS.TURN_RIGHT;
      }
    }

    return AGENT_ACTIONS.STRAIGHT;
  }

  applyRelativeActionImmediately(action) {
    const normalized = this.normalizeAgentAction(action);
    if (normalized === AGENT_ACTIONS.TURN_LEFT) {
      this.direction = this.rotateDirection(this.direction, "left");
    } else if (normalized === AGENT_ACTIONS.TURN_RIGHT) {
      this.direction = this.rotateDirection(this.direction, "right");
    }
    return normalized;
  }

  onEatFood(now) {
    const scoreGain = 10 * this.level;
    this.foodsEaten += 1;
    this.score += scoreGain;

    const levelUp = this.agentConfig.allowLeveling
      ? this.recalculateLevel(now)
      : false;

    this.spawnFood();
    const boardFilled = !this.food;

    if (
      !boardFilled &&
      this.agentConfig.enableBonusFood &&
      !this.bonusFood &&
      this.foodsEaten >= 3 &&
      this.foodsEaten % 3 === 0 &&
      this.randomFn() < this.config.bonusChance
    ) {
      this.spawnBonusFood(now);
    }

    this.flash(`+${scoreGain}`, now);
    this.updateBestRecords();

    return {
      scoreGain,
      levelUp,
      boardFilled,
    };
  }

  onEatBonus(now) {
    if (!this.bonusFood) {
      return { scoreGain: 0 };
    }

    const remainMs = Math.max(0, this.bonusFood.expiresAt - now);
    const bonusScore = 20 * this.level + Math.floor(remainMs / 130);
    this.score += bonusScore;
    this.bonusFood = null;

    this.flash(`奖励 +${bonusScore}`, now, true);
    this.updateBestRecords();

    return { scoreGain: bonusScore };
  }

  recalculateLevel(now) {
    if (!this.agentConfig.allowLeveling) {
      return false;
    }

    const nextLevel = 1 + Math.floor(this.foodsEaten / this.config.levelStepByFoods);
    if (nextLevel <= this.level) {
      return false;
    }

    const delta = nextLevel - this.level;
    this.level = nextLevel;

    if (this.agentConfig.enableObstacles) {
      for (let i = 0; i < delta; i += 1) {
        this.trySpawnObstacle();
      }
    }

    this.flash(`升级到 Lv.${this.level}`, now, true);
    return true;
  }

  trySpawnObstacle() {
    if (this.obstacles.size >= this.config.maxObstacles) {
      return;
    }

    for (let i = 0; i < this.boardSize * this.boardSize; i += 1) {
      const candidate = this.randomCell();
      if (!candidate) {
        return;
      }

      const key = this.cellKey(candidate.x, candidate.y);
      if (this.obstacles.has(key)) {
        continue;
      }
      if (this.isSnakeAt(candidate.x, candidate.y)) {
        continue;
      }
      if (this.food && candidate.x === this.food.x && candidate.y === this.food.y) {
        continue;
      }
      if (
        this.bonusFood &&
        candidate.x === this.bonusFood.x &&
        candidate.y === this.bonusFood.y
      ) {
        continue;
      }

      const head = this.snake[0];
      const distance =
        Math.abs(candidate.x - head.x) + Math.abs(candidate.y - head.y);
      if (distance < 4) {
        continue;
      }

      this.obstacles.add(key);
      return;
    }
  }

  spawnFood() {
    this.food = this.randomEmptyCell();
  }

  spawnBonusFood(now) {
    const cell = this.randomEmptyCell();
    if (!cell) {
      return;
    }

    this.bonusFood = {
      x: cell.x,
      y: cell.y,
      expiresAt: now + this.config.bonusLifeMs,
    };
  }

  randomEmptyCell() {
    const maxTry = this.boardSize * this.boardSize * 2;
    for (let i = 0; i < maxTry; i += 1) {
      const cell = this.randomCell();
      if (!cell) {
        return null;
      }

      const key = this.cellKey(cell.x, cell.y);
      const occupied =
        this.isSnakeAt(cell.x, cell.y) ||
        this.obstacles.has(key) ||
        (this.food && this.food.x === cell.x && this.food.y === cell.y) ||
        (this.bonusFood &&
          this.bonusFood.x === cell.x &&
          this.bonusFood.y === cell.y);

      if (!occupied) {
        return cell;
      }
    }

    return null;
  }

  randomCell() {
    if (this.boardSize <= 0) {
      return null;
    }
    return {
      x: Math.floor(this.randomFn() * this.boardSize),
      y: Math.floor(this.randomFn() * this.boardSize),
    };
  }

  inBounds(x, y) {
    return x >= 0 && x < this.boardSize && y >= 0 && y < this.boardSize;
  }

  isSnakeAt(x, y) {
    return this.snake.some((cell) => cell.x === x && cell.y === y);
  }

  getTickDuration() {
    const value =
      this.config.baseTick - (this.level - 1) * this.config.perLevelFaster;
    return Math.max(this.config.minTick, value);
  }

  updateHUD() {
    this.scoreValue.textContent = String(this.score);
    this.bestScoreValue.textContent = String(this.bestScore);
    this.lengthValue.textContent = String(this.snake.length);
    this.levelValue.textContent = String(this.level);

    const speed = this.config.baseTick / this.getTickDuration();
    this.speedValue.textContent = `${speed.toFixed(2)}x`;
  }

  updateBestRecords() {
    if (this.score > this.bestScore) {
      this.bestScore = this.score;
      this.saveNumberToStorage("snake.bestScore", this.bestScore);
    }

    if (this.snake.length > this.bestLength) {
      this.bestLength = this.snake.length;
      this.saveNumberToStorage("snake.bestLength", this.bestLength);
    }
  }

  loadNumberFromStorage(key) {
    try {
      const value = Number(localStorage.getItem(key));
      return Number.isFinite(value) ? value : 0;
    } catch {
      return 0;
    }
  }

  saveNumberToStorage(key, value) {
    try {
      localStorage.setItem(key, String(value));
    } catch {
      // localStorage 在某些隐私环境下可能不可写
    }
  }

  setSeed(seed) {
    if (seed === null || seed === undefined || seed === "") {
      this.randomSeed = null;
      this.randomFn = Math.random;
      return null;
    }

    const num = Number(seed);
    if (!Number.isFinite(num)) {
      throw new Error("seed 必须是可转换为数字的值");
    }

    const normalized = (Math.floor(num) >>> 0) || 1;
    this.randomSeed = normalized;
    let state = normalized;
    this.randomFn = () => {
      state += 0x6d2b79f5;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return this.randomSeed;
  }

  getSeed() {
    return this.randomSeed;
  }

  subscribeAgentEvent(eventName, handler) {
    if (
      !Object.prototype.hasOwnProperty.call(this.agentHooks, eventName) ||
      typeof handler !== "function"
    ) {
      return () => {};
    }

    const bucket = this.agentHooks[eventName];
    bucket.add(handler);
    return () => {
      bucket.delete(handler);
    };
  }

  emitAgentEvent(eventName, payload) {
    const bucket = this.agentHooks[eventName];
    if (!bucket) {
      return;
    }

    bucket.forEach((handler) => {
      try {
        handler(payload);
      } catch {
        // 外部订阅器不应影响游戏主循环
      }
    });
  }

  recordEpisodeTransition(transition) {
    if (!this.episodeStats) {
      return;
    }

    this.episodeStats.steps += 1;
    this.episodeStats.totalReward += transition.reward;
    this.episodeStats.scoreEnd = this.score;
    this.episodeStats.maxLength = Math.max(
      this.episodeStats.maxLength,
      this.snake.length
    );

    if (transition.info.ateFood) {
      this.episodeStats.foods += 1;
    }
    if (transition.info.ateBonusFood) {
      this.episodeStats.bonusFoods += 1;
    }
    if (transition.info.levelUp) {
      this.episodeStats.levelUps += 1;
    }

    if (transition.done) {
      this.episodeStats.done = true;
      this.episodeStats.terminalReason = transition.info.terminalReason;
      this.episodeStats.terminalReasonLabel = transition.info.terminalReasonLabel;
      this.episodeStats.endedAtMs = Date.now();
    }
  }

  getEpisodeStats() {
    const ended =
      this.episodeStats.endedAtMs > 0 ? this.episodeStats.endedAtMs : Date.now();
    return {
      ...this.episodeStats,
      durationMs: Math.max(0, ended - this.episodeStats.startedAtMs),
    };
  }

  flash(text, now, highlight = false) {
    this.flashText = text;
    this.flashUntil = now + 900;
    this.flashHighlight = highlight;
  }

  setOverlay(visible, title = "", text = "") {
    this.overlay.classList.toggle("hidden", !visible);

    if (title) {
      this.overlayTitle.textContent = title;
    }
    if (text) {
      this.overlayText.textContent = text;
    }
  }

  getTerminalReasonLabel(reasonCode) {
    return TERMINAL_REASON_LABEL[reasonCode] || reasonCode || "";
  }

  finalizeGameOver(reasonCode, options = {}) {
    if (this.state === "over") {
      return;
    }

    const { silentUI = false } = options;
    this.state = "over";
    this.lastTerminalReason = reasonCode || "";
    this.pauseBtn.disabled = true;
    this.pauseBtn.textContent = "暂停";

    if (!this.episodeStats.done) {
      this.episodeStats.done = true;
      this.episodeStats.terminalReason = this.lastTerminalReason;
      this.episodeStats.terminalReasonLabel = this.getTerminalReasonLabel(
        this.lastTerminalReason
      );
      this.episodeStats.endedAtMs = Date.now();
    }

    this.updateBestRecords();
    this.updateHUD();

    if (silentUI) {
      this.setOverlay(false);
      return;
    }

    this.setSettingsLocked(false);
    const reasonLabel = this.getTerminalReasonLabel(reasonCode);
    this.setOverlay(
      true,
      "游戏结束",
      `${reasonLabel}，最终分数 ${this.score}。按 R 或点击「开始 / 重开」再来一局`
    );
  }

  cellKey(x, y) {
    return `${x},${y}`;
  }

  render(now) {
    if (!this.renderEnabled) {
      return;
    }

    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    const bg = ctx.createLinearGradient(0, 0, w, h);
    bg.addColorStop(0, "#06122a");
    bg.addColorStop(1, "#0b1a39");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    this.drawGrid(ctx);
    this.drawObstacles(ctx);
    this.drawFood(ctx);
    this.drawBonusFood(ctx, now);
    this.drawSnake(ctx);
    this.drawFlashText(ctx, now);
  }

  drawGrid(ctx) {
    ctx.strokeStyle = "rgba(120,150,230,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let i = 0; i <= this.boardSize; i += 1) {
      const p = i * this.cellSize + 0.5;
      ctx.moveTo(p, 0);
      ctx.lineTo(p, this.canvas.height);
      ctx.moveTo(0, p);
      ctx.lineTo(this.canvas.width, p);
    }
    ctx.stroke();
  }

  drawObstacles(ctx) {
    const pad = this.cellSize * 0.12;
    this.obstacles.forEach((entry) => {
      const [x, y] = entry.split(",").map(Number);
      const px = x * this.cellSize + pad;
      const py = y * this.cellSize + pad;
      const size = this.cellSize - pad * 2;
      ctx.fillStyle = "#273e68";
      ctx.fillRect(px, py, size, size);
      ctx.strokeStyle = "#5e7fb9";
      ctx.strokeRect(px + 1, py + 1, size - 2, size - 2);
    });
  }

  drawFood(ctx) {
    if (!this.food) {
      return;
    }

    const cx = (this.food.x + 0.5) * this.cellSize;
    const cy = (this.food.y + 0.5) * this.cellSize;
    const r = this.cellSize * 0.3;

    ctx.save();
    ctx.shadowBlur = 12;
    ctx.shadowColor = "#ff5f7c";
    ctx.fillStyle = "#ff5f7c";
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  drawBonusFood(ctx, now) {
    if (!this.bonusFood) {
      return;
    }

    const remain = this.bonusFood.expiresAt - now;
    if (remain <= 0) {
      return;
    }

    const blinkOn = remain > 1600 || Math.floor(remain / 180) % 2 === 0;
    if (!blinkOn) {
      return;
    }

    const cx = (this.bonusFood.x + 0.5) * this.cellSize;
    const cy = (this.bonusFood.y + 0.5) * this.cellSize;
    const r = this.cellSize * 0.34;

    ctx.save();
    ctx.shadowBlur = 12;
    ctx.shadowColor = "#ffd66b";
    ctx.fillStyle = "#ffd66b";
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx + r, cy);
    ctx.lineTo(cx, cy + r);
    ctx.lineTo(cx - r, cy);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  drawSnake(ctx) {
    const length = this.snake.length;
    for (let i = length - 1; i >= 0; i -= 1) {
      const node = this.snake[i];
      const progress = i / Math.max(1, length - 1);
      const x = node.x * this.cellSize;
      const y = node.y * this.cellSize;
      const pad = i === 0 ? this.cellSize * 0.08 : this.cellSize * 0.14;

      if (i === 0) {
        ctx.fillStyle = "#6bffd7";
      } else {
        const light = Math.round(42 + (1 - progress) * 20);
        ctx.fillStyle = `hsl(146 83% ${light}%)`;
      }

      ctx.fillRect(
        x + pad,
        y + pad,
        this.cellSize - pad * 2,
        this.cellSize - pad * 2
      );
    }

    this.drawSnakeEyes(ctx);
  }

  drawSnakeEyes(ctx) {
    const head = this.snake[0];
    const x = head.x * this.cellSize;
    const y = head.y * this.cellSize;
    const r = Math.max(2, this.cellSize * 0.08);
    const offsetA = this.cellSize * 0.28;
    const offsetB = this.cellSize * 0.68;
    ctx.fillStyle = "#0d2135";

    if (this.direction === "up" || this.direction === "down") {
      const yy = this.direction === "up" ? y + offsetA : y + offsetB;
      ctx.beginPath();
      ctx.arc(x + offsetA, yy, r, 0, Math.PI * 2);
      ctx.arc(x + offsetB, yy, r, 0, Math.PI * 2);
      ctx.fill();
      return;
    }

    const xx = this.direction === "left" ? x + offsetA : x + offsetB;
    ctx.beginPath();
    ctx.arc(xx, y + offsetA, r, 0, Math.PI * 2);
    ctx.arc(xx, y + offsetB, r, 0, Math.PI * 2);
    ctx.fill();
  }

  drawFlashText(ctx, now) {
    if (!this.flashText || now >= this.flashUntil) {
      return;
    }

    const alpha = Math.max(0, (this.flashUntil - now) / 900);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = "bold 24px Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = this.flashHighlight ? "#ffe07a" : "#9cf2ff";
    ctx.fillText(this.flashText, this.canvas.width / 2, 36);
    ctx.restore();
  }

  setRewardWeights(partialWeights) {
    if (!partialWeights || typeof partialWeights !== "object") {
      return { ...this.rewardWeights };
    }

    const next = { ...this.rewardWeights };
    for (const key of Object.keys(DEFAULT_REWARD_WEIGHTS)) {
      if (Object.prototype.hasOwnProperty.call(partialWeights, key)) {
        const value = Number(partialWeights[key]);
        if (Number.isFinite(value)) {
          next[key] = value;
        }
      }
    }
    this.rewardWeights = next;
    return { ...this.rewardWeights };
  }

  setRenderEnabled(enabled) {
    this.renderEnabled = Boolean(enabled);
    if (this.renderEnabled) {
      this.render(performance.now());
    }
    return this.renderEnabled;
  }

  getActionSpace() {
    return {
      type: "discrete",
      size: 3,
      semantics: "relative_turn",
      actions: {
        straight: AGENT_ACTIONS.STRAIGHT,
        turnLeft: AGENT_ACTIONS.TURN_LEFT,
        turnRight: AGENT_ACTIONS.TURN_RIGHT,
      },
    };
  }

  getObservationSpace() {
    return {
      type: "tensor",
      dtype: "float32",
      layout: "HWC",
      shape: [this.boardSize, this.boardSize, AGENT_OBSERVATION_CHANNELS.length],
      channels: [...AGENT_OBSERVATION_CHANNELS],
    };
  }

  getObservationTensor() {
    const size = this.boardSize;
    const channels = AGENT_OBSERVATION_CHANNELS.length;
    const data = new Float32Array(size * size * channels);

    const setCell = (x, y, channel, value) => {
      if (!this.inBounds(x, y)) {
        return;
      }
      const index = (y * size + x) * channels + channel;
      data[index] = value;
    };

    this.snake.forEach((cell, idx) => {
      setCell(cell.x, cell.y, idx === 0 ? 0 : 1, 1);
    });

    if (this.food) {
      setCell(this.food.x, this.food.y, 2, 1);
    }

    if (this.bonusFood) {
      setCell(this.bonusFood.x, this.bonusFood.y, 3, 1);
    }

    this.obstacles.forEach((entry) => {
      const [x, y] = entry.split(",").map(Number);
      setCell(x, y, 4, 1);
    });

    const directionIdx = DIRECTION_INDEX[this.direction];
    if (directionIdx !== undefined) {
      const channelIndex = 5 + directionIdx;
      for (let idx = channelIndex; idx < data.length; idx += channels) {
        data[idx] = 1;
      }
    }

    return {
      data,
      shape: [size, size, channels],
      channels: [...AGENT_OBSERVATION_CHANNELS],
      dtype: "float32",
    };
  }

  getAgentMetadata() {
    return {
      apiVersion: AGENT_API_VERSION,
      seed: this.getSeed(),
      mode: this.gameMode,
      difficulty: this.difficulty,
      boardSize: this.boardSize,
      envConfig: { ...this.agentConfig },
      rewardWeights: { ...this.rewardWeights },
      terminalReasons: { ...TERMINAL_REASONS },
      actionSpace: this.getActionSpace(),
      observationSpace: this.getObservationSpace(),
    };
  }

  getSupportedConfigs() {
    return {
      difficulties: Object.keys(DIFFICULTY_CONFIG),
      modes: ["classic", "wrap"],
      boardSize: {
        min: 8,
        max: 64,
      },
    };
  }

  getAgentStateSnapshot() {
    return {
      state: this.state,
      mode: this.gameMode,
      difficulty: this.difficulty,
      boardSize: this.boardSize,
      direction: this.direction,
      score: this.score,
      level: this.level,
      foodsEaten: this.foodsEaten,
      stepsSinceLastFood: this.stepsSinceLastFood,
      snake: this.snake.map((cell) => ({ x: cell.x, y: cell.y })),
      food: this.food ? { ...this.food } : null,
      bonusFood: this.bonusFood
        ? {
            x: this.bonusFood.x,
            y: this.bonusFood.y,
            expiresAt: this.bonusFood.expiresAt,
          }
        : null,
      obstacles: Array.from(this.obstacles).map((entry) => {
        const [x, y] = entry.split(",").map(Number);
        return { x, y };
      }),
      rewardWeights: { ...this.rewardWeights },
      envConfig: { ...this.agentConfig },
      seed: this.getSeed(),
      episodeStats: this.getEpisodeStats(),
      lastTerminalReason: this.lastTerminalReason,
    };
  }

  pickEnvConfig(raw = {}) {
    const keys = [
      "difficulty",
      "mode",
      "boardSize",
      "enableBonusFood",
      "enableObstacles",
      "allowLeveling",
      "maxStepsWithoutFood",
    ];
    const picked = {};
    keys.forEach((key) => {
      if (Object.prototype.hasOwnProperty.call(raw, key)) {
        picked[key] = raw[key];
      }
    });
    return picked;
  }

  configureAgent(options = {}) {
    const source =
      options && typeof options.envConfig === "object"
        ? options.envConfig
        : options;
    const next = this.sanitizeAgentConfig({
      ...this.agentConfig,
      ...this.pickEnvConfig(source || {}),
    });
    this.agentConfig = next;
    this.applyRuntimeConfig(next, { syncUI: false });
    return { ...this.agentConfig };
  }

  setAgentStateSnapshot(snapshot, options = {}) {
    if (!snapshot || typeof snapshot !== "object") {
      throw new Error("state snapshot 无效");
    }

    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }

    const snapshotConfig = this.pickEnvConfig({
      ...snapshot,
      ...(snapshot.envConfig || {}),
    });
    const mergedConfig = this.sanitizeAgentConfig({
      ...this.agentConfig,
      ...snapshotConfig,
    });
    this.agentConfig = mergedConfig;
    this.applyRuntimeConfig(mergedConfig, { syncUI: false });

    if (snapshot.rewardWeights) {
      this.setRewardWeights(snapshot.rewardWeights);
    }
    if (Object.prototype.hasOwnProperty.call(snapshot, "seed")) {
      this.setSeed(snapshot.seed);
    }

    const normalizedSnake = Array.isArray(snapshot.snake)
      ? snapshot.snake
          .map((cell) => ({
            x: Math.round(Number(cell.x)),
            y: Math.round(Number(cell.y)),
          }))
          .filter((cell) => this.inBounds(cell.x, cell.y))
      : [];
    this.snake = normalizedSnake.length > 0 ? normalizedSnake : [{ x: 1, y: 1 }];

    const normalizeCellOrNull = (cell) => {
      if (!cell || typeof cell !== "object") {
        return null;
      }
      const x = Math.round(Number(cell.x));
      const y = Math.round(Number(cell.y));
      if (!this.inBounds(x, y)) {
        return null;
      }
      return { x, y };
    };

    this.food = normalizeCellOrNull(snapshot.food);

    const bonusCell = normalizeCellOrNull(snapshot.bonusFood);
    this.bonusFood = bonusCell
      ? {
          x: bonusCell.x,
          y: bonusCell.y,
          expiresAt:
            Number(snapshot.bonusFood.expiresAt) || performance.now() + this.config.bonusLifeMs,
        }
      : null;

    this.obstacles = new Set();
    if (Array.isArray(snapshot.obstacles)) {
      snapshot.obstacles.forEach((cell) => {
        const normalized = normalizeCellOrNull(cell);
        if (normalized) {
          this.obstacles.add(this.cellKey(normalized.x, normalized.y));
        }
      });
    }

    this.direction = DIRECTION_INDEX[snapshot.direction] !== undefined ? snapshot.direction : "right";
    this.directionQueue = [];
    this.score = Number.isFinite(Number(snapshot.score)) ? Number(snapshot.score) : 0;
    this.level = Number.isFinite(Number(snapshot.level))
      ? Math.max(1, Math.round(Number(snapshot.level)))
      : 1;
    this.foodsEaten = Number.isFinite(Number(snapshot.foodsEaten))
      ? Math.max(0, Math.round(Number(snapshot.foodsEaten)))
      : 0;
    this.stepsSinceLastFood = Number.isFinite(Number(snapshot.stepsSinceLastFood))
      ? Math.max(0, Math.round(Number(snapshot.stepsSinceLastFood)))
      : 0;
    this.lastTerminalReason = snapshot.lastTerminalReason || "";

    this.flashText = "";
    this.flashUntil = 0;
    this.flashHighlight = false;
    this.lastTransition = null;

    if (snapshot.episodeStats && typeof snapshot.episodeStats === "object") {
      this.episodeStats = {
        ...this.createEmptyEpisodeStats(),
        ...snapshot.episodeStats,
      };
    } else {
      this.resetEpisodeStats();
    }

    const validStates = new Set(["ready", "running", "paused", "over"]);
    this.state = validStates.has(snapshot.state) ? snapshot.state : "running";
    this.agentControlled =
      options.agentControlled !== undefined
        ? Boolean(options.agentControlled)
        : true;
    this.lastFrameTime = performance.now();
    this.accumulator = 0;

    if (this.agentControlled) {
      this.setSettingsLocked(true);
      this.pauseBtn.disabled = true;
      this.setOverlay(false);
    } else {
      this.pauseBtn.disabled = this.state !== "running" && this.state !== "paused";
      this.setSettingsLocked(this.state === "running" || this.state === "paused");
    }

    this.updateHUD();
    if (options.render !== false && this.renderEnabled) {
      this.render(performance.now());
    }

    return this.getAgentStateSnapshot();
  }

  resetForAgent(options = {}) {
    if (this.isClosed) {
      throw new Error("游戏实例已关闭，无法重置");
    }

    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }

    if (Object.prototype.hasOwnProperty.call(options, "seed")) {
      this.setSeed(options.seed);
    }
    if (Object.prototype.hasOwnProperty.call(options, "rewardWeights")) {
      this.setRewardWeights(options.rewardWeights);
    }
    if (typeof options.renderEnabled === "boolean") {
      this.setRenderEnabled(options.renderEnabled);
    }

    this.configureAgent(options);
    this.applyRuntimeConfig(this.agentConfig, { syncUI: false });
    this.resetRuntime();

    this.agentControlled = true;
    this.state = "running";
    this.episodeIndex += 1;
    this.resetEpisodeStats();
    this.lastFrameTime = performance.now();
    this.accumulator = 0;

    this.setSettingsLocked(true);
    this.pauseBtn.disabled = true;
    this.pauseBtn.textContent = "暂停";
    this.setOverlay(false);

    if (this.renderEnabled) {
      this.render(performance.now());
    }
    const transition = {
      observation: this.getObservationTensor(),
      reward: 0,
      done: false,
      info: {
        episode: this.episodeIndex,
        reset: true,
        scoreBefore: 0,
        scoreAfter: this.score,
        lengthBefore: this.snake.length,
        lengthAfter: this.snake.length,
        levelBefore: this.level,
        levelAfter: this.level,
        terminalReason: "",
        terminalReasonLabel: "",
      },
    };

    this.lastTransition = transition;
    this.emitAgentEvent("reset", {
      transition,
      metadata: this.getAgentMetadata(),
      state: this.getAgentStateSnapshot(),
    });

    return transition;
  }

  stepForAgent(action = AGENT_ACTIONS.STRAIGHT, options = {}) {
    if (this.isClosed) {
      throw new Error("游戏实例已关闭，无法执行 step");
    }

    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
    this.agentControlled = true;

    const repeatRaw = Number(options.repeat);
    const repeat =
      Number.isFinite(repeatRaw) && repeatRaw > 1 ? Math.round(repeatRaw) : 1;
    const actionList = Array.isArray(action) ? action : null;
    const transitions = [];

    for (let i = 0; i < repeat; i += 1) {
      const stepAction = actionList
        ? actionList[Math.min(i, actionList.length - 1)]
        : action;

      if (this.state !== "running") {
        const reasonCode = this.lastTerminalReason || TERMINAL_REASONS.NOT_RUNNING;
        const terminalTransition = {
          observation: this.getObservationTensor(),
          reward: 0,
          done: true,
          info: {
            episode: this.episodeIndex,
            step: this.episodeStats.steps,
            action: this.normalizeAgentAction(stepAction),
            scoreBefore: this.score,
            scoreAfter: this.score,
            lengthBefore: this.snake.length,
            lengthAfter: this.snake.length,
            levelBefore: this.level,
            levelAfter: this.level,
            foodsEaten: this.foodsEaten,
            terminalReason: reasonCode,
            terminalReasonLabel: this.getTerminalReasonLabel(reasonCode),
          },
        };
        transitions.push(terminalTransition);
        break;
      }

      const transition = this.advanceOneTick(performance.now(), {
        agentAction: stepAction,
        autoFinalize: true,
        silentGameOver: true,
      });
      transition.info.episode = this.episodeIndex;
      this.emitAgentEvent("transition", {
        transition,
        episodeStats: this.getEpisodeStats(),
        metadata: this.getAgentMetadata(),
      });

      transitions.push(transition);
      if (transition.done) {
        this.emitAgentEvent("done", {
          transition,
          episodeStats: this.getEpisodeStats(),
          metadata: this.getAgentMetadata(),
        });
        break;
      }
    }

    if (this.renderEnabled) {
      this.render(performance.now());
    }

    if (options.returnAllTransitions) {
      return transitions;
    }

    if (transitions.length > 0) {
      return transitions[transitions.length - 1];
    }

    const reasonCode = this.lastTerminalReason || TERMINAL_REASONS.NOT_RUNNING;
    return {
      observation: this.getObservationTensor(),
      reward: 0,
      done: true,
      info: {
        episode: this.episodeIndex,
        step: this.episodeStats.steps,
        action: this.normalizeAgentAction(action),
        scoreBefore: this.score,
        scoreAfter: this.score,
        lengthBefore: this.snake.length,
        lengthAfter: this.snake.length,
        levelBefore: this.level,
        levelAfter: this.level,
        foodsEaten: this.foodsEaten,
        terminalReason: reasonCode,
        terminalReasonLabel: this.getTerminalReasonLabel(reasonCode),
      },
    };
  }

  stepBatch(actions = [], options = {}) {
    if (!Array.isArray(actions)) {
      throw new Error("stepBatch 需要传入动作数组");
    }

    const stopOnDone = options.stopOnDone !== false;
    const transitions = [];
    for (let i = 0; i < actions.length; i += 1) {
      const transition = this.stepForAgent(actions[i], { returnAllTransitions: false });
      transitions.push(transition);
      if (stopOnDone && transition.done) {
        break;
      }
    }
    return transitions;
  }

  sampleAction() {
    return Math.floor(this.randomFn() * 3);
  }

  getObservationFlat() {
    const obs = this.getObservationTensor();
    return {
      data: obs.data,
      shape: [obs.data.length],
      dtype: obs.dtype,
      channels: [...obs.channels],
      originalShape: [...obs.shape],
    };
  }

  renderFrame() {
    this.render(performance.now());
    return true;
  }

  close() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }

    this.state = "over";
    this.agentControlled = false;
    this.isClosed = true;
    this.unbindEvents();
    this.setOverlay(true, "已关闭", "该实例已关闭，请刷新页面后重新创建");
    this.agentHooks.reset.clear();
    this.agentHooks.transition.clear();
    this.agentHooks.done.clear();
    return true;
  }

  resumeHumanControl() {
    if (this.isClosed) {
      return;
    }

    this.agentControlled = false;
    this.pauseBtn.disabled = this.state !== "running" && this.state !== "paused";

    if (this.state === "running") {
      this.lastFrameTime = performance.now();
      this.accumulator = 0;
      if (!this.rafId) {
        this.rafId = requestAnimationFrame((ts) => this.gameLoop(ts));
      }
      this.setOverlay(false);
      return;
    }

    if (this.state === "paused") {
      this.setOverlay(true, "已暂停", "按空格 / P 或点击「继续」返回游戏");
      return;
    }

    this.setSettingsLocked(false);
    this.setOverlay(true, "准备开始", "按「开始 / 重开」或 Enter 开始游戏");
  }

  createAgentAPI() {
    return Object.freeze({
      API_VERSION: AGENT_API_VERSION,
      ACTIONS: AGENT_ACTIONS,
      TERMINAL_REASONS,
      DEFAULT_REWARD_WEIGHTS: { ...DEFAULT_REWARD_WEIGHTS },
      DEFAULT_ENV_CONFIG: { ...DEFAULT_AGENT_ENV_CONFIG },
      OBSERVATION_CHANNELS: [...AGENT_OBSERVATION_CHANNELS],
      reset: (options) => this.resetForAgent(options),
      step: (action, options) => this.stepForAgent(action, options),
      stepBatch: (actions, options) => this.stepBatch(actions, options),
      sampleAction: () => this.sampleAction(),
      configure: (options) => this.configureAgent(options),
      getObservation: () => this.getObservationTensor(),
      getObservationFlat: () => this.getObservationFlat(),
      getActionSpace: () => this.getActionSpace(),
      getObservationSpace: () => this.getObservationSpace(),
      getSupportedConfigs: () => this.getSupportedConfigs(),
      getMetadata: () => this.getAgentMetadata(),
      getState: () => this.getAgentStateSnapshot(),
      setState: (snapshot, options) => this.setAgentStateSnapshot(snapshot, options),
      getEpisodeStats: () => this.getEpisodeStats(),
      getLastTransition: () => this.lastTransition,
      setRewardWeights: (weights) => this.setRewardWeights(weights),
      getRewardWeights: () => ({ ...this.rewardWeights }),
      setRenderEnabled: (enabled) => this.setRenderEnabled(enabled),
      renderFrame: () => this.renderFrame(),
      setSeed: (seed) => this.setSeed(seed),
      getSeed: () => this.getSeed(),
      subscribe: (eventName, handler) =>
        this.subscribeAgentEvent(eventName, handler),
      resumeHumanControl: () => this.resumeHumanControl(),
      startHumanGame: () => this.startNewGame(),
      close: () => this.close(),
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const game = new SnakeGame();
  const agentAPI = game.createAgentAPI();

  class RemoteInferenceController {
    constructor(gameInstance, api) {
      this.game = gameInstance;
      this.agentAPI = api;
      this.modelInfo = null;
      this.running = false;
      this.runToken = 0;

      this.serverUrlInput = document.getElementById("inferenceServerUrl");
      this.checkpointPathInput = document.getElementById("checkpointPath");
      this.stepDelayInput = document.getElementById("aiStepDelay");
      this.autoLoopCheckbox = document.getElementById("aiAutoLoop");
      this.loadModelBtn = document.getElementById("loadModelBtn");
      this.startAiBtn = document.getElementById("startAiBtn");
      this.stopAiBtn = document.getElementById("stopAiBtn");
      this.aiStatus = document.getElementById("aiStatus");
      this.touchButtons = Array.from(document.querySelectorAll(".touch-controls button"));

      this.blockHumanKeys = (event) => {
        if (!this.running) {
          return;
        }
        const blocked = new Set([
          "KeyA",
          "KeyD",
          "ArrowLeft",
          "ArrowRight",
          "Space",
          "KeyP",
          "Enter",
          "KeyR",
        ]);
        if (blocked.has(event.code)) {
          event.preventDefault();
          event.stopImmediatePropagation();
        }
      };

      document.addEventListener("keydown", this.blockHumanKeys, true);
      this.bindEvents();
      this.updateButtons();
    }

    bindEvents() {
      this.loadModelBtn.addEventListener("click", () => {
        this.loadModel();
      });
      this.startAiBtn.addEventListener("click", () => {
        this.start();
      });
      this.stopAiBtn.addEventListener("click", () => {
        this.stop("已停止 AI，恢复人工接管。", "success");
      });
    }

    normalizeServerUrl() {
      const raw = String(this.serverUrlInput.value || "").trim();
      if (!raw) {
        throw new Error("请先填写推理服务地址");
      }
      return raw.replace(/\/+$/, "");
    }

    getStepDelayMs() {
      const delay = Number(this.stepDelayInput.value);
      if (!Number.isFinite(delay) || delay < 0) {
        return 0;
      }
      return Math.round(delay);
    }

    setStatus(message, kind = "info") {
      this.aiStatus.textContent = message;
      this.aiStatus.dataset.kind = kind;
    }

    formatError(error) {
      if (error && typeof error === "object" && "message" in error) {
        return String(error.message);
      }
      return String(error);
    }

    updateButtons() {
      this.loadModelBtn.disabled = this.running;
      this.startAiBtn.disabled = this.running || !this.modelInfo;
      this.stopAiBtn.disabled = !this.running;
    }

    setHumanControlsDisabled(disabled) {
      this.game.startBtn.disabled = disabled;
      if (disabled) {
        this.game.pauseBtn.disabled = true;
      } else if (!this.running && !this.game.agentControlled) {
        this.game.pauseBtn.disabled = this.game.state !== "running" && this.game.state !== "paused";
      }
      this.touchButtons.forEach((button) => {
        button.disabled = disabled;
      });
    }

    async postJson(path, payload) {
      const serverUrl = this.normalizeServerUrl();
      const response = await fetch(`${serverUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || `请求失败: ${response.status}`);
      }
      return data;
    }

    applyRecommendedUI(modelInfo) {
      const env = modelInfo.recommendedEnvConfig || {};
      const fixedBoardSize =
        modelInfo.modelType === "small_cnn"
          ? Number(env.board_size || modelInfo.inputSize)
          : Number(env.board_size || this.game.boardSizeSelect.value);

      if (env.difficulty && DIFFICULTY_CONFIG[env.difficulty]) {
        this.game.difficultySelect.value = env.difficulty;
      }
      if (env.mode === "classic" || env.mode === "wrap") {
        this.game.modeSelect.value = env.mode;
      }
      if (Array.from(this.game.boardSizeSelect.options).some((opt) => Number(opt.value) === fixedBoardSize)) {
        this.game.boardSizeSelect.value = String(fixedBoardSize);
      }
    }

    async loadModel() {
      const checkpoint = String(this.checkpointPathInput.value || "").trim();
      if (!checkpoint) {
        this.setStatus("请先填写模型路径。", "error");
        return;
      }

      this.setStatus("正在加载模型...", "running");
      try {
        const modelInfo = await this.postJson("/v1/load", { checkpoint });
        this.modelInfo = modelInfo;
        this.applyRecommendedUI(modelInfo);
        this.updateButtons();

        const sizeText = modelInfo.supportsVariableBoard
          ? "支持可变地图"
          : `固定地图 ${modelInfo.inputSize}x${modelInfo.inputSize}`;
        this.setStatus(
          `模型已加载：${modelInfo.modelType}，${sizeText}。现在可以点击“AI 接管”。`,
          "success"
        );
      } catch (error) {
        this.modelInfo = null;
        this.updateButtons();
        this.setStatus(`模型加载失败：${this.formatError(error)}`, "error");
      }
    }

    buildAgentResetOptions() {
      const recommended = this.modelInfo?.recommendedEnvConfig || {};
      const boardSize =
        this.modelInfo?.modelType === "small_cnn"
          ? Number(recommended.board_size || this.modelInfo.inputSize)
          : Number(this.game.boardSizeSelect.value);

      return {
        difficulty: this.game.difficultySelect.value,
        mode: this.game.modeSelect.value,
        boardSize,
        enableBonusFood:
          recommended.enable_bonus_food !== undefined ? Boolean(recommended.enable_bonus_food) : true,
        enableObstacles:
          recommended.enable_obstacles !== undefined ? Boolean(recommended.enable_obstacles) : true,
        allowLeveling:
          recommended.allow_leveling !== undefined ? Boolean(recommended.allow_leveling) : true,
        maxStepsWithoutFood:
          recommended.max_steps_without_food !== undefined
            ? Number(recommended.max_steps_without_food)
            : 0,
        renderEnabled: true,
      };
    }

    async requestAction() {
      const state = this.agentAPI.getState();
      const result = await this.postJson("/v1/act", { state });
      return Number(result.action);
    }

    restoreAfterEpisode(transition) {
      const terminalLabel =
        transition?.info?.terminalReasonLabel ||
        transition?.info?.terminalReason ||
        "本局结束";
      this.running = false;
      this.game.agentControlled = false;
      this.game.setSettingsLocked(false);
      this.game.startBtn.disabled = false;
      this.game.pauseBtn.disabled = true;
      this.game.pauseBtn.textContent = "暂停";
      this.touchButtons.forEach((button) => {
        button.disabled = false;
      });
      this.game.setOverlay(true, "AI 对局结束", `${terminalLabel}。可重新开始，或再次让 AI 接管。`);
      this.updateButtons();
      this.setStatus(`AI 对局结束：${terminalLabel}`, "success");
    }

    stop(message = "AI 已停止。", kind = "success") {
      if (!this.running) {
        return;
      }
      this.running = false;
      this.runToken += 1;
      this.setHumanControlsDisabled(false);
      this.agentAPI.resumeHumanControl();
      this.updateButtons();
      this.setStatus(message, kind);
    }

    async start() {
      if (this.running || !this.modelInfo) {
        return;
      }

      const token = this.runToken + 1;
      this.runToken = token;
      this.running = true;
      this.updateButtons();
      this.setHumanControlsDisabled(true);
      this.setStatus("AI 正在接管游戏...", "running");

      try {
        do {
          let transition = this.agentAPI.reset(this.buildAgentResetOptions());

          while (this.running && this.runToken === token && !transition.done) {
            const action = await this.requestAction();
            if (!this.running || this.runToken !== token) {
              return;
            }
            transition = this.agentAPI.step(action);
            const delayMs = this.getStepDelayMs();
            if (delayMs > 0) {
              await sleep(delayMs);
            }
          }

          if (!this.running || this.runToken !== token) {
            return;
          }

          if (!this.autoLoopCheckbox.checked) {
            this.restoreAfterEpisode(transition);
            return;
          }

          this.setStatus("本局结束，AI 即将自动重开...", "running");
          await sleep(800);
        } while (this.running && this.runToken === token);
      } catch (error) {
        this.running = false;
        this.setHumanControlsDisabled(false);
        this.agentAPI.resumeHumanControl();
        this.updateButtons();
        this.setStatus(`AI 运行失败：${this.formatError(error)}`, "error");
      }
    }
  }

  const remoteInferenceController = new RemoteInferenceController(game, agentAPI);

  window.snakeGame = game;
  window.snakeAgentAPI = agentAPI;
  window.remoteInferenceController = remoteInferenceController;
});
