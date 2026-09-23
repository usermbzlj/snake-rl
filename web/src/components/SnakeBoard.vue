<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import type { Cell } from '@/api'

const props = withDefaults(
  defineProps<{
    size: number
    snake: Cell[]
    food: Cell
    dir?: number
    heatmap?: number[] | null
    showHeatmap?: boolean
    dead?: boolean
    maxCssSize?: number
    fitContainer?: boolean
  }>(),
  {
    dir: 1,
    heatmap: null,
    showHeatmap: false,
    dead: false,
    maxCssSize: 520,
    fitContainer: false,
  },
)

const canvasRef = ref<HTMLCanvasElement | null>(null)
const wrapRef = ref<HTMLDivElement | null>(null)
let raf = 0
let pulseRaf = 0
let reduceMotion = false

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function lerpColor(a: [number, number, number], b: [number, number, number], t: number) {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)] as [
    number,
    number,
    number,
  ]
}

function rgb(c: [number, number, number], a = 1) {
  return `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`
}

/** Transparent → amber → hot; low values fully transparent. */
function heatColor(v: number): string {
  const t = Math.max(0, Math.min(1, v))
  if (t < 0.08) return 'rgba(0,0,0,0)'
  if (t < 0.45) {
    const u = (t - 0.08) / 0.37
    return `rgba(255, 180, 60, ${0.12 + u * 0.35})`
  }
  if (t < 0.75) {
    const u = (t - 0.45) / 0.3
    return `rgba(255, ${Math.round(140 - u * 60)}, ${Math.round(40 - u * 20)}, ${0.47 + u * 0.25})`
  }
  const u = (t - 0.75) / 0.25
  return `rgba(255, ${Math.round(80 - u * 40)}, ${Math.round(20 + u * 40)}, ${0.72 + u * 0.2})`
}

function cellCenter(pad: number, cell: number, r: number, c: number) {
  return { x: pad + c * cell + cell / 2, y: pad + r * cell + cell / 2 }
}

function desaturate(c: [number, number, number], amount: number): [number, number, number] {
  const g = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
  return [
    lerp(c[0], g, amount),
    lerp(c[1], g, amount),
    lerp(c[2], g, amount),
  ]
}

function draw() {
  const canvas = canvasRef.value
  const wrap = wrapRef.value
  if (!canvas || !wrap) return
  const size = Math.max(5, props.size)
  const avail = wrap.clientWidth || props.maxCssSize
  const cssW = props.fitContainer
    ? Math.max(120, avail)
    : Math.min(avail, props.maxCssSize)
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5)
  canvas.style.width = `${cssW}px`
  canvas.style.height = `${cssW}px`
  canvas.width = Math.floor(cssW * dpr)
  canvas.height = Math.floor(cssW * dpr)
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const pad = Math.max(5, cssW * 0.018)
  const board = cssW - pad * 2
  const cell = board / size

  // frame background
  ctx.clearRect(0, 0, cssW, cssW)
  roundRect(ctx, 0, 0, cssW, cssW, Math.min(14, cssW * 0.04))
  const bg = ctx.createLinearGradient(0, 0, cssW, cssW)
  bg.addColorStop(0, '#071018')
  bg.addColorStop(1, '#0c1828')
  ctx.fillStyle = bg
  ctx.fill()

  // playfield
  roundRect(ctx, pad, pad, board, board, Math.min(10, cell * 0.35))
  ctx.fillStyle = '#081421'
  ctx.fill()

  // subtle grid
  ctx.strokeStyle = 'rgba(130, 165, 210, 0.07)'
  ctx.lineWidth = 1
  for (let i = 1; i < size; i++) {
    const x = pad + i * cell
    ctx.beginPath()
    ctx.moveTo(x + 0.5, pad)
    ctx.lineTo(x + 0.5, pad + board)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(pad, pad + i * cell + 0.5)
    ctx.lineTo(pad + board, pad + i * cell + 0.5)
    ctx.stroke()
  }

  // saliency under snake/food
  if (props.showHeatmap && props.heatmap && props.heatmap.length >= size * size) {
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const v = props.heatmap[r * size + c] ?? 0
        const col = heatColor(v)
        if (col === 'rgba(0,0,0,0)') continue
        const inset = cell * 0.08
        roundRect(
          ctx,
          pad + c * cell + inset,
          pad + r * cell + inset,
          cell - inset * 2,
          cell - inset * 2,
          cell * 0.18,
        )
        ctx.fillStyle = col
        ctx.fill()
      }
    }
  }

  // food with optional pulse
  const pulse = reduceMotion
    ? 1
    : 0.85 + 0.15 * Math.sin(performance.now() / 420)
  const [fr, fc] = props.food
  if (fr >= 0 && fc >= 0 && fr < size && fc < size) {
    const fx = pad + fc * cell + cell / 2
    const fy = pad + fr * cell + cell / 2
    const frad = cell * 0.26 * pulse
    const glow = ctx.createRadialGradient(fx, fy, 0, fx, fy, cell * 0.72 * pulse)
    glow.addColorStop(0, `rgba(255, 107, 138, ${0.55 * pulse})`)
    glow.addColorStop(1, 'rgba(255, 107, 138, 0)')
    ctx.fillStyle = glow
    ctx.beginPath()
    ctx.arc(fx, fy, cell * 0.72 * pulse, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#ff6b8a'
    ctx.beginPath()
    ctx.arc(fx, fy, frad, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = 'rgba(255,255,255,0.5)'
    ctx.beginPath()
    ctx.arc(fx - frad * 0.28, fy - frad * 0.28, frad * 0.3, 0, Math.PI * 2)
    ctx.fill()
  }

  // continuous snake body along path
  const body = props.snake.filter((p) => p[0] >= 0 && p[1] >= 0 && p[0] < size && p[1] < size)
  const n = body.length
  let headCol: [number, number, number] = [94, 255, 215]
  let midCol: [number, number, number] = [45, 200, 175]
  let tailCol: [number, number, number] = [22, 110, 125]
  if (props.dead) {
    headCol = [240, 110, 120]
    midCol = desaturate([90, 140, 150], 0.55)
    tailCol = desaturate([50, 80, 95], 0.7)
  }

  if (n > 0) {
    const lineW = cell * (props.dead ? 0.58 : 0.62)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    if (n === 1) {
      const p = cellCenter(pad, cell, body[0]![0], body[0]![1])
      ctx.fillStyle = rgb(headCol)
      ctx.beginPath()
      ctx.arc(p.x, p.y, lineW * 0.55, 0, Math.PI * 2)
      ctx.fill()
    } else {
      // draw tail→head segments so head paints on top
      for (let i = n - 2; i >= 0; i--) {
        const a = body[i + 1]!
        const b = body[i]!
        const pa = cellCenter(pad, cell, a[0], a[1])
        const pb = cellCenter(pad, cell, b[0], b[1])
        const t0 = (i + 1) / (n - 1)
        const t1 = i / (n - 1)
        const c0 = t0 < 0.5 ? lerpColor(headCol, midCol, t0 * 2) : lerpColor(midCol, tailCol, (t0 - 0.5) * 2)
        const c1 = t1 < 0.5 ? lerpColor(headCol, midCol, t1 * 2) : lerpColor(midCol, tailCol, (t1 - 0.5) * 2)
        const grad = ctx.createLinearGradient(pa.x, pa.y, pb.x, pb.y)
        grad.addColorStop(0, rgb(c0, props.dead ? 0.75 : 1))
        grad.addColorStop(1, rgb(c1, props.dead ? 0.75 : 1))
        ctx.strokeStyle = grad
        ctx.lineWidth = lineW * (0.88 + (1 - t1) * 0.12)
        ctx.beginPath()
        ctx.moveTo(pa.x, pa.y)
        ctx.lineTo(pb.x, pb.y)
        ctx.stroke()
      }
    }

    // enlarged head disc
    const head = body[0]!
    const hp = cellCenter(pad, cell, head[0], head[1])
    const headR = cell * (props.dead ? 0.38 : 0.4)
    const headGrad = ctx.createRadialGradient(
      hp.x - headR * 0.2,
      hp.y - headR * 0.2,
      headR * 0.1,
      hp.x,
      hp.y,
      headR,
    )
    headGrad.addColorStop(0, rgb(headCol))
    headGrad.addColorStop(1, rgb(lerpColor(headCol, midCol, 0.35)))
    ctx.fillStyle = headGrad
    ctx.beginPath()
    ctx.arc(hp.x, hp.y, headR, 0, Math.PI * 2)
    ctx.fill()

    if (props.dead) {
      ctx.strokeStyle = 'rgba(240, 113, 120, 0.85)'
      ctx.lineWidth = 2
      ctx.stroke()
    }

    // eyes facing heading
    const eye = cell * 0.085
    const off = cell * 0.15
    const dir = ((props.dir % 4) + 4) % 4
    let e1x = 0,
      e1y = 0,
      e2x = 0,
      e2y = 0,
      px = 0,
      py = 0
    if (dir === 0) {
      e1x = -off
      e1y = -off * 0.35
      e2x = off
      e2y = -off * 0.35
      px = 0
      py = -eye * 0.65
    } else if (dir === 1) {
      e1x = off * 0.35
      e1y = -off
      e2x = off * 0.35
      e2y = off
      px = eye * 0.65
      py = 0
    } else if (dir === 2) {
      e1x = -off
      e1y = off * 0.35
      e2x = off
      e2y = off * 0.35
      px = 0
      py = eye * 0.65
    } else {
      e1x = -off * 0.35
      e1y = -off
      e2x = -off * 0.35
      e2y = off
      px = -eye * 0.65
      py = 0
    }
    for (const [ex, ey] of [
      [e1x, e1y],
      [e2x, e2y],
    ]) {
      ctx.fillStyle = props.dead ? 'rgba(255,220,220,0.9)' : '#f4fffc'
      ctx.beginPath()
      ctx.arc(hp.x + ex, hp.y + ey, eye, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = props.dead ? '#5a1520' : '#0a2030'
      ctx.beginPath()
      ctx.arc(hp.x + ex + px * 0.55, hp.y + ey + py * 0.55, eye * 0.48, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const rr = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.arcTo(x + w, y, x + w, y + h, rr)
  ctx.arcTo(x + w, y + h, x, y + h, rr)
  ctx.arcTo(x, y + h, x, y, rr)
  ctx.arcTo(x, y, x + w, y, rr)
  ctx.closePath()
}

function schedule() {
  cancelAnimationFrame(raf)
  raf = requestAnimationFrame(draw)
}

function pulseLoop() {
  if (reduceMotion) return
  schedule()
  pulseRaf = requestAnimationFrame(pulseLoop)
}

let ro: ResizeObserver | null = null
onMounted(() => {
  reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  schedule()
  if (!reduceMotion) pulseRaf = requestAnimationFrame(pulseLoop)
  if (wrapRef.value) {
    ro = new ResizeObserver(() => schedule())
    ro.observe(wrapRef.value)
  }
})
onUnmounted(() => {
  cancelAnimationFrame(raf)
  cancelAnimationFrame(pulseRaf)
  ro?.disconnect()
})

watch(
  () => [
    props.size,
    props.snake,
    props.food,
    props.dir,
    props.heatmap,
    props.showHeatmap,
    props.dead,
    props.maxCssSize,
    props.fitContainer,
  ],
  () => schedule(),
  { deep: true },
)
</script>

<template>
  <div ref="wrapRef" class="board-wrap" :class="{ fit: fitContainer }">
    <canvas
      ref="canvasRef"
      role="img"
      :aria-label="`棋盘 ${size}×${size}${dead ? '，已结束' : ''}`"
    />
  </div>
</template>

<style scoped>
.board-wrap {
  width: 100%;
  max-width: v-bind(maxCssSize + 'px');
  margin: 0 auto;
  line-height: 0;
}
.board-wrap.fit {
  max-width: none;
}
canvas {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 14px;
}
</style>
