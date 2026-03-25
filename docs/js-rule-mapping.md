# JS to Python rule mapping

This file documents how `web/game.js` was mirrored in `snake_rl/env.py`.

## Actions

- JS `STRAIGHT=0`, `TURN_LEFT=1`, `TURN_RIGHT=2`
- Python `ACTIONS` uses the same integer mapping

## Observation tensor

- JS shape: `[H, W, 9]`, dtype `float32`
- Python `SnakeEnv.get_observation()` keeps the same shape and channel order:
  1. `snakeHead`
  2. `snakeBody`
  3. `food`
  4. `bonusFood`
  5. `obstacle`
  6. `dirUp`
  7. `dirRight`
  8. `dirDown`
  9. `dirLeft`

## Reward defaults

- `alive=-0.01`
- `food=+1.0`
- `bonusFood=+1.5`
- `death=-1.0`
- `timeout=-0.6`
- `levelUp=+0.2`
- `victory=+2.0`

## Terminal reasons

- wall
- obstacle
- self
- board_full
- timeout
- not_running

## Gameplay rules

- Relative-turn action semantics are identical.
- `classic` mode ends on wall collision; `wrap` mode uses edge wrapping.
- Self-collision check uses tail-exclusion when snake does not grow this step.
- Bonus food spawn chance, level-up pacing, and obstacle spawn limits are copied
  from JS difficulty configs.
- Timeout rule mirrors JS: terminate when `steps_since_last_food >= max_steps_without_food`.
