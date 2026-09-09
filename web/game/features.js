"use strict";

function normalizeBoardCoord(value, size, wrap) {
  if (wrap) {
    return ((value % size) + size) % size;
  }
  return value;
}

function computeObservationTensor(state) {
  const size = state.boardSize;
  const channels = AGENT_OBSERVATION_CHANNELS.length;
  const data = new Float32Array(size * size * channels);
  const inBounds = (x, y) => x >= 0 && y >= 0 && x < size && y < size;
  const setCell = (x, y, channel, value) => {
    if (!inBounds(x, y)) {
      return;
    }
    data[(y * size + x) * channels + channel] = value;
  };

  (state.snake || []).forEach((cell, idx) => {
    setCell(cell.x, cell.y, idx === 0 ? 0 : 1, 1);
  });
  if (state.food) {
    setCell(state.food.x, state.food.y, 2, 1);
  }
  if (state.bonusFood) {
    setCell(state.bonusFood.x, state.bonusFood.y, 3, 1);
  }
  (state.obstacles || []).forEach((cell) => {
    setCell(cell.x, cell.y, 4, 1);
  });

  const directionIdx = DIRECTION_INDEX[state.direction];
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

function computeTinyFeatures(state) {
  const feat = new Float32Array(TINY_FEAT_DIM);
  const size = state.boardSize;
  const head = state.snake[0];
  const bodySet = new Set((state.snake || []).slice(1).map((c) => `${c.x},${c.y}`));
  const obstacleSet = new Set((state.obstacles || []).map((c) => `${c.x},${c.y}`));
  const wrap = state.mode === "wrap";

  const castRay = (hx, hy, rdx, rdy) => {
    for (let step = 1; step <= size; step += 1) {
      let nx = hx + rdx * step;
      let ny = hy + rdy * step;
      if (wrap) {
        nx = normalizeBoardCoord(nx, size, true);
        ny = normalizeBoardCoord(ny, size, true);
      } else if (nx < 0 || nx >= size || ny < 0 || ny >= size) {
        return step;
      }
      if (bodySet.has(`${nx},${ny}`) || obstacleSet.has(`${nx},${ny}`)) {
        return step;
      }
    }
    return size;
  };

  const rays = TINY_RAYS[state.direction];
  for (let i = 0; i < 7; i += 1) {
    feat[i] = castRay(head.x, head.y, rays[i][0], rays[i][1]) / size;
  }

  if (state.food) {
    let dx = state.food.x - head.x;
    let dy = state.food.y - head.y;
    if (wrap && size > 1) {
      const half = Math.floor(size / 2);
      if (dx > half) dx -= size;
      else if (dx < -half) dx += size;
      if (dy > half) dy -= size;
      else if (dy < -half) dy += size;
    }
    const hdx = DIRS[state.direction].x;
    const hdy = DIRS[state.direction].y;
    const norm = Math.max(1, size - 1);
    feat[7] = (dx * hdx + dy * hdy) / norm;
    feat[8] = (dx * -hdy + dy * hdx) / norm;
  }

  feat[9] = state.snake.length / (size * size);
  return { data: feat, shape: [TINY_FEAT_DIM], dtype: "float32" };
}

function computeLocalPatch(state, patchSize = 11) {
  const size = Number(patchSize);
  if (!Number.isInteger(size) || size <= 0 || size % 2 === 0) {
    throw new Error("patch_size 必须是正奇数");
  }
  const radius = Math.floor(size / 2);
  const board = state.boardSize;
  const channels = AGENT_OBSERVATION_CHANNELS.length;
  const data = new Float32Array(size * size * channels);
  const head = state.snake[0];
  const directionIdx = DIRECTION_INDEX[state.direction];
  const wrap = state.mode === "wrap";
  const bodySet = new Set((state.snake || []).slice(1).map((c) => `${c.x},${c.y}`));
  const obstacleSet = new Set((state.obstacles || []).map((c) => `${c.x},${c.y}`));

  for (let py = 0; py < size; py += 1) {
    for (let px = 0; px < size; px += 1) {
      const base = (py * size + px) * channels;
      if (directionIdx !== undefined) {
        data[base + 5 + directionIdx] = 1;
      }
      let sx = head.x + (px - radius);
      let sy = head.y + (py - radius);
      if (wrap) {
        sx = normalizeBoardCoord(sx, board, true);
        sy = normalizeBoardCoord(sy, board, true);
      } else if (sx < 0 || sx >= board || sy < 0 || sy >= board) {
        continue;
      }
      if (sx === head.x && sy === head.y) {
        data[base] = 1;
      } else if (bodySet.has(`${sx},${sy}`)) {
        data[base + 1] = 1;
      }
      if (state.food && state.food.x === sx && state.food.y === sy) {
        data[base + 2] = 1;
      }
      if (state.bonusFood && state.bonusFood.x === sx && state.bonusFood.y === sy) {
        data[base + 3] = 1;
      }
      if (obstacleSet.has(`${sx},${sy}`)) {
        data[base + 4] = 1;
      }
    }
  }

  return {
    data,
    shape: [size, size, channels],
    channels: [...AGENT_OBSERVATION_CHANNELS],
    dtype: "float32",
  };
}

function featureStateFromGame(game) {
  const obstacles = [];
  if (game.obstacles && typeof game.obstacles.forEach === "function") {
    game.obstacles.forEach((entry) => {
      if (typeof entry === "string") {
        const [x, y] = entry.split(",").map(Number);
        obstacles.push({ x, y });
      } else if (entry && typeof entry === "object") {
        obstacles.push({ x: Number(entry.x), y: Number(entry.y) });
      }
    });
  }
  return {
    boardSize: game.boardSize,
    mode: game.gameMode,
    direction: game.direction,
    snake: game.snake,
    food: game.food,
    bonusFood: game.bonusFood,
    obstacles,
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    computeObservationTensor,
    computeTinyFeatures,
    computeLocalPatch,
    featureStateFromGame,
  };
}
