import { describe, expect, it } from 'vitest'
import {
  SnakeEngine,
  absoluteToRelative,
  applyRelative,
  hungerLimit,
  type Dir,
} from '../game/engine'
import { downsampleEven, formatNumber, formatSteps } from '../utils/format'

describe('hungerLimit', () => {
  it('uses max(hunger_factor*s*s, 4*s)', () => {
    expect(hungerLimit(8, 1)).toBe(Math.max(64, 32))
    expect(hungerLimit(8, 0.2)).toBe(32)
    expect(hungerLimit(20, 1)).toBe(400)
  })
})

describe('relative turns', () => {
  it('turns left/right/straight', () => {
    expect(applyRelative(1, 0)).toBe(1)
    expect(applyRelative(1, 1)).toBe(0) // RIGHT → UP
    expect(applyRelative(1, 2)).toBe(2) // RIGHT → DOWN
    expect(applyRelative(0, 1)).toBe(3)
  })

  it('maps absolute to relative and rejects 180', () => {
    expect(absoluteToRelative(1, 1)).toBe(0)
    expect(absoluteToRelative(1, 0)).toBe(1)
    expect(absoluteToRelative(1, 2)).toBe(2)
    expect(absoluteToRelative(1, 3)).toBeNull()
  })
})

describe('SnakeEngine', () => {
  it('starts length 3 heading right at center', () => {
    const e = new SnakeEngine({ size: 8, rng: () => 0 })
    expect(e.length).toBe(3)
    expect(e.dir).toBe(1)
    expect(e.body[0]).toEqual([4, 4])
    expect(e.body[1]).toEqual([4, 3])
    expect(e.body[2]).toEqual([4, 2])
  })

  it('allows moving into vacating tail cell', () => {
    // 5x5 board; craft a looping path where next cell is current tail
    const e = new SnakeEngine({ size: 5, rng: () => 0.99 })
    // Force body: head (2,2) right, then (2,1), (2,0) — wait need length 3 circling
    e.body = [
      [2, 2],
      [2, 1],
      [1, 1],
    ]
    e.dir = 3 as Dir // LEFT → new head (2,1) which is current neck... bad
    // Better: body head(1,2), (1,1), (1,0) dir DOWN → (2,2); then RIGHT etc.
    // Classic: snake in a 2x2 square, moving into tail
    e.body = [
      [2, 2],
      [2, 1],
      [1, 1],
    ]
    e.dir = 0 as Dir // UP → (1,2) empty
    e.food = [0, 0]
    e.done = false
    e.stepRelative(0)
    expect(e.done).toBe(false)
    expect(e.body[0]).toEqual([1, 2])

    // Now body: (1,2),(2,2),(2,1) — turn left (from UP → LEFT) to (1,1) which is vacated tail
    e.food = [0, 4]
    e.stepRelative(1) // turn left → LEFT
    // After previous step dir was UP; left → LEFT; new head (1,1)
    // Wait stepRelative applies turn then moves. dir was UP, action left → LEFT, head (1,1)
    // Before move body was (1,2),(2,2),(2,1); tail (2,1) vacates; (1,1) was not in body. Hmm.

    // Reset for clear tail-follow:
    // body: H(2,1) (2,2) T(1,2), dir LEFT. Next (2,0) if exists...
    // Square chase: H(1,1),(1,2),(2,2) dir DOWN → (2,1). Tail (2,2) stays until move.
    // After: H(2,1),(1,1),(1,2). Turn left (DOWN→RIGHT) → (2,2) which equals old mid... 
    // After first: body (2,1)(1,1)(1,2), dir DOWN. Right turn → RIGHT, new=(2,2).
    // Occupied without tail: (2,1)(1,1). Tail (1,2) vacated. (2,2) free? Yes empty.
    
    // Known legal tail-follow: 
    // body head→tail: (2,2),(2,1),(2,0) length 3 horizontal, dir RIGHT
    // Move straight to (2,3). Not tail.
    // body: (1,2),(2,2),(2,1) dir UP. Move to (0,2).
    // Let's do: body (2,2),(3,2),(3,1), dir LEFT → new (2,1).
    // Cells occupied excl tail: (2,2),(3,2). Tail (3,1) vacates. (2,1) empty — not interesting.

    // The classic case: snake length 4 in a tight turn
    // body: (2,2),(2,1),(1,1),(1,2) — a 2x2 ring, dir RIGHT → (2,3) off...
    // On 5 board: body (2,2),(2,1),(1,1) dir UP → (1,2)
    // occupied excl tail (1,1): {(2,2),(2,1)}. (1,2) free.
    // Next: body (1,2),(2,2),(2,1) dir UP, turn right → RIGHT, new (1,3)
    
    // Tail-follow: body (2,1),(1,1),(1,2) dir RIGHT. New head (2,2).
    // excl tail (1,2): {(2,1),(1,1)}. (2,2) free.
    
    // Real tail-follow: body (2,2),(2,1),(1,1) dir LEFT → (2,0)
    // Better from SPEC: "Moving into the cell the tail is currently leaving is legal"
    // body: H(1,1) (1,2) T(2,2), heading DOWN. New head = (2,1).
    // Without eating, tail (2,2) leaves. Is (2,1) occupied? No.
    // Need head moving INTO where tail is NOW.
    // body: H(1,2) (2,2) T(2,1), heading LEFT → new (1,1) — not tail.
    // body: H(2,1) (1,1) T(1,2), heading RIGHT → (2,2). Tail is (1,2). Not.
    // body: H(1,1) (1,2) T(2,2), heading RIGHT — wall if edge.
    // body: H(2,2) (1,2) T(1,1), heading LEFT → (2,1). Tail (1,1).
    // body: H(2,2) (2,1) T(1,1), heading UP → (1,2). 
    // body: H(2,1) (2,2) T(1,2), heading UP → (1,1). Tail is (1,2). 
    // body: H(1,2) (1,1) T(2,1), heading DOWN → (2,2).
    // THE CASE: H(2,2),(1,2),(1,1) dir DOWN → new head (3,2) ...
    // Ring of 3: positions (1,1)=H, (1,2), (2,2)=T, dir DOWN. Next=(2,1) empty.
    // Ring: H(1,1),(2,1),(2,2) dir RIGHT → (1,2). Occupied excl T(2,2): {(1,1),(2,1)}. (1,2) free.
    // Ring: H(1,2),(1,1),(2,1) dir DOWN → (2,2). excl T: {(1,2),(1,1)}. (2,2) free.
    // Ring: H(2,2),(1,2),(1,1) dir LEFT → (2,1). excl T(1,1): {(2,2),(1,2)}. (2,1) free — AND (2,1) was never body.
    
    // For length-4: H(2,2),(2,1),(1,1),(1,2) clockwise, dir RIGHT.
    // New = (2,3). 
    // dir UP from H(2,2),(1,2),(1,1),(2,1): new (1,2) which is neck — collision with >1.
    // Self collision: body[new]>1; tail==1 legal.
    // So: H(2,2), (2,1), (1,1), (1,2)=tail, heading LEFT → (2,1) is neck (life>1) → collision.
    // Heading UP → (1,2) which is TAIL → legal when not eating!
    e.body = [
      [2, 2],
      [2, 1],
      [1, 1],
      [1, 2],
    ]
    e.dir = 0 as Dir // UP into tail (1,2)
    e.food = [0, 0]
    e.done = false
    e.cause = 0
    e.score = 1
    e.steps = 10
    e.stepsSinceFood = 0
    e.stepRelative(0)
    expect(e.done).toBe(false)
    expect(e.body[0]).toEqual([1, 2])
    expect(e.length).toBe(4)
  })

  it('detects self collision (non-tail)', () => {
    const e = new SnakeEngine({ size: 8, rng: () => 0 })
    e.body = [
      [3, 3],
      [3, 2],
      [2, 2],
      [2, 3],
      [2, 4],
    ]
    e.dir = 0 as Dir // UP → (2,3) which is body index 3, not tail
    e.food = [0, 0]
    e.stepRelative(0)
    expect(e.done).toBe(true)
    expect(e.cause).toBe(2)
  })

  it('detects wall collision', () => {
    const e = new SnakeEngine({ size: 5, rng: () => 0 })
    e.body = [
      [0, 2],
      [1, 2],
      [2, 2],
    ]
    e.dir = 0 as Dir
    e.food = [2, 0]
    e.stepRelative(0)
    expect(e.done).toBe(true)
    expect(e.cause).toBe(1)
  })

  it('grows when eating and increments score', () => {
    const e = new SnakeEngine({ size: 8, rng: () => 0.99 })
    e.body = [
      [4, 4],
      [4, 3],
      [4, 2],
    ]
    e.dir = 1
    e.food = [4, 5]
    const before = e.length
    e.stepRelative(0)
    expect(e.length).toBe(before + 1)
    expect(e.score).toBe(1)
    expect(e.done).toBe(false)
  })

  it('wins when snake fills the board', () => {
    // 5x5 = 25 cells; start with 24, eat last
    const e = new SnakeEngine({ size: 5, rng: () => 0 })
    const cells: [number, number][] = []
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 5; c++) {
        if (!(r === 0 && c === 0)) cells.push([r, c])
      }
    }
    // head at (0,1) heading LEFT toward food (0,0); body fills rest
    e.body = [[0, 1], ...cells.filter(([r, c]) => !(r === 0 && c === 1))]
    // Ensure length 24
    expect(e.body.length).toBe(24)
    e.dir = 3 as Dir
    e.food = [0, 0]
    e.stepRelative(0)
    expect(e.done).toBe(true)
    expect(e.cause).toBe(4)
    expect(e.length).toBe(25)
  })

  it('buffers absolute inputs without 180', () => {
    const e = new SnakeEngine({ size: 8, rng: () => 0 })
    e.queueAbsolute(3) // LEFT while facing RIGHT — reject 180
    e.stepHuman()
    expect(e.dir).toBe(1)
    e.queueAbsolute(0) // UP
    e.queueAbsolute(3) // LEFT relative to buffered UP — ok
    e.stepHuman()
    expect(e.dir).toBe(0)
    e.stepHuman()
    expect(e.dir).toBe(3)
  })
})

describe('format utils', () => {
  it('downsamples evenly keeping ends', () => {
    const arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    expect(downsampleEven(arr, 5)).toEqual([0, 2, 5, 7, 9])
    expect(downsampleEven(arr, 100)).toEqual(arr)
  })

  it('formats numbers', () => {
    expect(formatNumber(null)).toBe('—')
    expect(formatSteps(1_500_000)).toBe('1.50M')
    expect(formatSteps(2500)).toBe('2.5k')
  })
})
