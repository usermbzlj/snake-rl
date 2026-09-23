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
  }>(),
  {
    dir: 1,
    heatmap: null,
    showHeatmap: false,
    dead: false,
    maxCssSize: 420,
  },
)

const canvasRef = ref<HTMLCanvasElement | null>(null)
const wrapRef = ref<HTMLDivElement | null>(null)
let raf = 0

function lerpColor(a: [number, number, number], b: [number, number, number], t: number) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ] as [number, number, number]
}

function rgb([r, g, b]: [number, number, number], a = 1) {
  return `rgba(${r | 0},${g | 0},${b | 0},${a})`
}

function draw() {
  const canvas = canvasRef.value
  const wrap = wrapRef.value
  if (!canvas || !wrap) return
  const size = Math.max(5, props.size)
  const cssW = Math.min(wrap.clientWidth || props.maxCssSize, props.maxCssSize)
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5)
  canvas.style.width = `${cssW}px`
  canvas.style.height = `${cssW}px`
  canvas.width = Math.floor(cssW * dpr)
  canvas.height = Math.floor(cssW * dpr)
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const pad = 6
  const board = cssW - pad * 2
  const cell = board / size

  // background
  ctx.clearRect(0, 0, cssW, cssW)
  const bg = ctx.createLinearGradient(0, 0, cssW, cssW)
  bg.addColorStop(0, '#0a1220')
  bg.addColorStop(1, '#0e1a2e')
  roundRect(ctx, 0, 0, cssW, cssW, 14)
  ctx.fillStyle = bg
  ctx.fill()

  // playfield
  roundRect(ctx, pad, pad, board, board, 10)
  ctx.fillStyle = '#0b1526'
  ctx.fill()

  // grid
  ctx.strokeStyle = 'rgba(120, 150, 190, 0.08)'
  ctx.lineWidth = 1
  for (let i = 1; i < size; i++) {
    const x = pad + i * cell
    ctx.beginPath()
    ctx.moveTo(x, pad)
    ctx.lineTo(x, pad + board)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(pad, pad + i * cell)
    ctx.lineTo(pad + board, pad + i * cell)
    ctx.stroke()
  }

  // heatmap
  if (props.showHeatmap && props.heatmap && props.heatmap.length >= size * size) {
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const v = props.heatmap[r * size + c] ?? 0
        if (v <= 0.02) continue
        ctx.fillStyle = `rgba(255, 120, 80, ${0.15 + v * 0.55})`
        ctx.fillRect(pad + c * cell, pad + r * cell, cell, cell)
      }
    }
  }

  // food glow
  const [fr, fc] = props.food
  const fx = pad + fc * cell + cell / 2
  const fy = pad + fr * cell + cell / 2
  const frad = cell * 0.28
  const glow = ctx.createRadialGradient(fx, fy, 0, fx, fy, cell * 0.7)
  glow.addColorStop(0, 'rgba(255, 107, 138, 0.55)')
  glow.addColorStop(1, 'rgba(255, 107, 138, 0)')
  ctx.fillStyle = glow
  ctx.beginPath()
  ctx.arc(fx, fy, cell * 0.7, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = '#ff6b8a'
  ctx.beginPath()
  ctx.arc(fx, fy, frad, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = 'rgba(255,255,255,0.45)'
  ctx.beginPath()
  ctx.arc(fx - frad * 0.25, fy - frad * 0.25, frad * 0.28, 0, Math.PI * 2)
  ctx.fill()

  // snake
  const n = props.snake.length
  const headCol: [number, number, number] = [94, 255, 215]
  const tailCol: [number, number, number] = [26, 120, 130]
  for (let i = n - 1; i >= 0; i--) {
    const [r, c] = props.snake[i]!
    const t = n <= 1 ? 0 : i / (n - 1)
    const col = lerpColor(headCol, tailCol, t)
    const cx = pad + c * cell + cell / 2
    const cy = pad + r * cell + cell / 2
    const rad = cell * (i === 0 ? 0.42 : 0.36)
    ctx.fillStyle = rgb(col, props.dead ? 0.45 : 1)
    roundRect(ctx, cx - rad, cy - rad, rad * 2, rad * 2, rad * 0.55)
    ctx.fill()
  }

  // eyes on head
  if (props.snake.length > 0 && !props.dead) {
    const [hr, hc] = props.snake[0]!
    const hx = pad + hc * cell + cell / 2
    const hy = pad + hr * cell + cell / 2
    const eye = cell * 0.08
    const off = cell * 0.14
    const dir = props.dir % 4
    // eye pair perpendicular to heading
    let e1x = 0,
      e1y = 0,
      e2x = 0,
      e2y = 0,
      px = 0,
      py = 0
    if (dir === 0) {
      // UP
      e1x = -off
      e1y = -off * 0.4
      e2x = off
      e2y = -off * 0.4
      px = 0
      py = -eye * 0.6
    } else if (dir === 1) {
      e1x = off * 0.4
      e1y = -off
      e2x = off * 0.4
      e2y = off
      px = eye * 0.6
      py = 0
    } else if (dir === 2) {
      e1x = -off
      e1y = off * 0.4
      e2x = off
      e2y = off * 0.4
      px = 0
      py = eye * 0.6
    } else {
      e1x = -off * 0.4
      e1y = -off
      e2x = -off * 0.4
      e2y = off
      px = -eye * 0.6
      py = 0
    }
    for (const [ex, ey] of [
      [e1x, e1y],
      [e2x, e2y],
    ]) {
      ctx.fillStyle = '#eafffa'
      ctx.beginPath()
      ctx.arc(hx + ex, hy + ey, eye, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#0a2030'
      ctx.beginPath()
      ctx.arc(hx + ex + px * 0.5, hy + ey + py * 0.5, eye * 0.45, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  if (props.dead) {
    ctx.fillStyle = 'rgba(8, 12, 20, 0.35)'
    roundRect(ctx, pad, pad, board, board, 10)
    ctx.fill()
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

let ro: ResizeObserver | null = null
onMounted(() => {
  schedule()
  if (wrapRef.value) {
    ro = new ResizeObserver(() => schedule())
    ro.observe(wrapRef.value)
  }
})
onUnmounted(() => {
  cancelAnimationFrame(raf)
  ro?.disconnect()
})

watch(
  () => [props.size, props.snake, props.food, props.dir, props.heatmap, props.showHeatmap, props.dead],
  () => schedule(),
  { deep: true },
)
</script>

<template>
  <div ref="wrapRef" class="board-wrap">
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
canvas {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 14px;
}
</style>
