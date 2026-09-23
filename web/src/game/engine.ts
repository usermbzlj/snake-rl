/**
 * TypeScript Snake engine matching SPEC game rules.
 * Directions: 0=UP 1=RIGHT 2=DOWN 3=LEFT
 * Relative actions (AI): 0=straight 1=turn left 2=turn right
 * Human input uses absolute directions with a small buffer (no 180°).
 */

export type Dir = 0 | 1 | 2 | 3
export type RelativeAction = 0 | 1 | 2
export type Cell = [number, number]
export type Cause = 0 | 1 | 2 | 3 | 4

export const DIR_DELTAS: ReadonlyArray<readonly [number, number]> = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
]

export interface EngineOptions {
  size: number
  hungerFactor?: number
  rng?: () => number
}

export interface EngineState {
  size: number
  body: Cell[]
  food: Cell
  dir: Dir
  score: number
  steps: number
  stepsSinceFood: number
  done: boolean
  cause: Cause
}

export function hungerLimit(size: number, hungerFactor = 1.0): number {
  return Math.max(Math.floor(hungerFactor * size * size), 4 * size)
}

export function applyRelative(dir: Dir, action: RelativeAction): Dir {
  if (action === 1) return ((dir + 3) % 4) as Dir
  if (action === 2) return ((dir + 1) % 4) as Dir
  return dir
}

export function absoluteToRelative(current: Dir, desired: Dir): RelativeAction | null {
  if (desired === current) return 0
  if (((desired + 2) % 4) as Dir === current) return null
  if (((current + 3) % 4) as Dir === desired) return 1
  if (((current + 1) % 4) as Dir === desired) return 2
  return null
}

function cellsEqual(a: Cell, b: Cell): boolean {
  return a[0] === b[0] && a[1] === b[1]
}

function bodySet(body: Cell[]): Set<string> {
  const s = new Set<string>()
  for (const [r, c] of body) s.add(`${r},${c}`)
  return s
}

export class SnakeEngine {
  size: number
  hungerFactor: number
  private rng: () => number
  body: Cell[] = []
  food: Cell = [0, 0]
  dir: Dir = 1
  score = 0
  steps = 0
  stepsSinceFood = 0
  done = false
  cause: Cause = 0
  private inputQueue: Dir[] = []

  constructor(opts: EngineOptions) {
    const size = Math.max(5, Math.min(32, Math.floor(opts.size)))
    this.size = size
    this.hungerFactor = opts.hungerFactor ?? 1.0
    this.rng = opts.rng ?? Math.random
    this.reset()
  }

  get hungerCap(): number {
    return hungerLimit(this.size, this.hungerFactor)
  }

  get length(): number {
    return this.body.length
  }

  snapshot(): EngineState {
    return {
      size: this.size,
      body: this.body.map((c) => [c[0], c[1]] as Cell),
      food: [this.food[0], this.food[1]],
      dir: this.dir,
      score: this.score,
      steps: this.steps,
      stepsSinceFood: this.stepsSinceFood,
      done: this.done,
      cause: this.cause,
    }
  }

  reset(size?: number): void {
    if (size != null) this.size = Math.max(5, Math.min(32, Math.floor(size)))
    const s = this.size
    const hr = Math.floor(s / 2)
    const hc = Math.floor(s / 2)
    this.body = [
      [hr, hc],
      [hr, hc - 1],
      [hr, hc - 2],
    ]
    this.dir = 1
    this.score = 0
    this.steps = 0
    this.stepsSinceFood = 0
    this.done = false
    this.cause = 0
    this.inputQueue = []
    this.placeFood()
  }

  queueAbsolute(desired: Dir): void {
    if (this.done) return
    const base = this.inputQueue.length > 0 ? this.inputQueue[this.inputQueue.length - 1]! : this.dir
    if (((desired + 2) % 4) as Dir === base) return
    if (desired === base && this.inputQueue.length > 0) return
    if (this.inputQueue.length >= 2) return
    if (desired === this.dir && this.inputQueue.length === 0) return
    this.inputQueue.push(desired)
  }

  stepRelative(action: RelativeAction): EngineState {
    if (this.done) return this.snapshot()
    this.dir = applyRelative(this.dir, action)
    return this.advance()
  }

  stepHuman(): EngineState {
    if (this.done) return this.snapshot()
    if (this.inputQueue.length > 0) {
      this.dir = this.inputQueue.shift()!
    }
    return this.advance()
  }

  private advance(): EngineState {
    const [dr, dc] = DIR_DELTAS[this.dir]!
    const head = this.body[0]!
    const nr = head[0] + dr
    const nc = head[1] + dc

    if (nr < 0 || nc < 0 || nr >= this.size || nc >= this.size) {
      this.done = true
      this.cause = 1
      this.steps += 1
      this.stepsSinceFood += 1
      return this.snapshot()
    }

    const newHead: Cell = [nr, nc]
    const eating = cellsEqual(newHead, this.food)
    const occupied = bodySet(this.body)

    if (!eating) {
      const tail = this.body[this.body.length - 1]!
      occupied.delete(`${tail[0]},${tail[1]}`)
    }

    if (occupied.has(`${nr},${nc}`)) {
      this.done = true
      this.cause = 2
      this.steps += 1
      this.stepsSinceFood += 1
      return this.snapshot()
    }

    this.body.unshift(newHead)
    this.steps += 1

    if (eating) {
      this.score += 1
      this.stepsSinceFood = 0
      if (this.body.length >= this.size * this.size) {
        this.done = true
        this.cause = 4
        return this.snapshot()
      }
      this.placeFood()
    } else {
      this.body.pop()
      this.stepsSinceFood += 1
      if (this.stepsSinceFood >= this.hungerCap) {
        this.done = true
        this.cause = 3
      }
    }

    return this.snapshot()
  }

  private placeFood(): void {
    const occupied = bodySet(this.body)
    const empty: Cell[] = []
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        if (!occupied.has(`${r},${c}`)) empty.push([r, c])
      }
    }
    if (empty.length === 0) {
      this.done = true
      this.cause = 4
      return
    }
    const idx = Math.floor(this.rng() * empty.length)
    this.food = empty[idx]!
  }
}
