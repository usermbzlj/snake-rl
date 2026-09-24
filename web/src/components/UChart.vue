<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import uPlot from 'uplot'
import { trimTrailingZeros } from '@/utils/format'

export interface ChartSeries {
  label: string
  data: (number | null | undefined)[]
  color?: string
  fill?: string
}

export interface ChartMarker {
  x: number
  label: string
  color?: string
}

const props = withDefaults(
  defineProps<{
    title?: string
    hint?: string
    x: number[]
    series: ChartSeries[]
    markers?: ChartMarker[]
    height?: number
    cursorX?: number | null
  }>(),
  {
    title: '',
    hint: '',
    markers: () => [],
    height: 200,
    cursorX: null,
  },
)

const root = ref<HTMLDivElement | null>(null)
let plot: uPlot | null = null
let ro: ResizeObserver | null = null
let hovering = false

const COLORS = ['#3dd6c6', '#6db3f2', '#f0c35a', '#f07178', '#b794f6', '#5ddea0']

function latestValue(data: (number | null | undefined)[]): string {
  for (let i = data.length - 1; i >= 0; i--) {
    const v = data[i]
    if (v != null && !Number.isNaN(Number(v))) {
      const n = Number(v)
      if (Math.abs(n) >= 100) return n.toFixed(0)
      if (Math.abs(n) >= 10) return n.toFixed(1)
      return trimTrailingZeros(n.toFixed(3))
    }
  }
  return '—'
}

function paintLegendLatest(u: uPlot) {
  if (hovering) return
  const vals = u.root.querySelectorAll('.u-legend .u-series .u-value')
  // First series row is x (步数)
  if (vals[0]) {
    const xs = u.data[0] as number[]
    const lastX = xs.length ? xs[xs.length - 1]! : null
    vals[0].textContent = lastX == null ? '—' : String(Math.round(lastX))
  }
  props.series.forEach((s, i) => {
    const el = vals[i + 1]
    if (el) el.textContent = latestValue(s.data)
  })
}

function buildOpts(width: number): uPlot.Options {
  const series: uPlot.Series[] = [{ label: '步数' }]
  for (let i = 0; i < props.series.length; i++) {
    const s = props.series[i]!
    series.push({
      label: s.label,
      stroke: s.color ?? COLORS[i % COLORS.length],
      width: 2,
      fill: s.fill,
      points: { show: false },
    })
  }

  return {
    width,
    height: props.height,
    title: props.title || undefined,
    series,
    scales: {
      x: { time: false },
    },
    axes: [
      {
        stroke: '#6b7c9c',
        grid: { stroke: 'rgba(148,178,220,0.08)', width: 1 },
        ticks: { stroke: 'rgba(148,178,220,0.2)' },
        font: '11px Segoe UI, sans-serif',
        size: 42,
      },
      {
        stroke: '#6b7c9c',
        grid: { stroke: 'rgba(148,178,220,0.08)', width: 1 },
        ticks: { stroke: 'rgba(148,178,220,0.2)' },
        font: '11px Segoe UI, sans-serif',
        size: 48,
      },
    ],
    legend: { show: true, live: true },
    cursor: {
      drag: { x: false, y: false },
    },
    hooks: {
      setCursor: [
        (u) => {
          const idx = u.cursor.idx
          hovering = typeof idx === 'number' && idx >= 0
          if (!hovering) {
            // defer so uPlot finishes writing '--'
            queueMicrotask(() => paintLegendLatest(u))
          }
        },
      ],
      setData: [
        (u) => {
          hovering = false
          queueMicrotask(() => paintLegendLatest(u))
        },
      ],
      ready: [
        (u) => {
          hovering = false
          queueMicrotask(() => paintLegendLatest(u))
          // also after a frame — uPlot may overwrite legend on first paint
          requestAnimationFrame(() => paintLegendLatest(u))
        },
      ],
      draw: [
        (u) => {
          const ctx = u.ctx
          const placedY: number[] = []
          for (const m of props.markers) {
            const cx = u.valToPos(m.x, 'x', true)
            if (cx < u.bbox.left || cx > u.bbox.left + u.bbox.width) continue
            ctx.save()
            ctx.strokeStyle = m.color ?? 'rgba(240,195,90,0.7)'
            ctx.lineWidth = 1.25
            ctx.setLineDash([4, 3])
            ctx.beginPath()
            ctx.moveTo(cx, u.bbox.top)
            ctx.lineTo(cx, u.bbox.top + u.bbox.height)
            ctx.stroke()
            ctx.setLineDash([])
            ctx.fillStyle = m.color ?? '#f0c35a'
            ctx.font = '600 10px Segoe UI, PingFang SC, sans-serif'
            let ty = u.bbox.top + 12
            for (const py of placedY) {
              if (Math.abs(py - ty) < 12 && Math.abs(cx - (u.bbox.left + 8)) < 200) ty += 12
            }
            placedY.push(ty)
            const text = m.label.length > 18 ? `${m.label.slice(0, 17)}…` : m.label
            ctx.fillText(text, cx + 4, ty)
            ctx.restore()
          }
          if (props.cursorX != null) {
            const cx = u.valToPos(props.cursorX, 'x', true)
            ctx.save()
            ctx.strokeStyle = 'rgba(61,214,198,0.85)'
            ctx.lineWidth = 1.5
            ctx.beginPath()
            ctx.moveTo(cx, u.bbox.top)
            ctx.lineTo(cx, u.bbox.top + u.bbox.height)
            ctx.stroke()
            ctx.restore()
          }
        },
      ],
    },
  }
}

function toData(): uPlot.AlignedData {
  const ys = props.series.map((s) =>
    s.data.map((v) => (v == null || Number.isNaN(v) ? null : Number(v))),
  )
  return [props.x, ...ys] as uPlot.AlignedData
}

function recreate() {
  if (!root.value) return
  plot?.destroy()
  plot = null
  hovering = false
  const width = root.value.clientWidth || 320
  if (props.x.length === 0) return
  plot = new uPlot(buildOpts(width), toData(), root.value)
}

function update() {
  if (!plot || !root.value) {
    recreate()
    return
  }
  if (plot.series.length - 1 !== props.series.length) {
    recreate()
    return
  }
  plot.setData(toData())
  plot.redraw()
  paintLegendLatest(plot)
}

onMounted(() => {
  recreate()
  if (root.value) {
    ro = new ResizeObserver(() => {
      if (!plot || !root.value) return
      plot.setSize({ width: root.value.clientWidth || 320, height: props.height })
      paintLegendLatest(plot)
    })
    ro.observe(root.value)
  }
})

onUnmounted(() => {
  ro?.disconnect()
  plot?.destroy()
  plot = null
})

watch(
  () => [props.x, props.series, props.markers, props.cursorX, props.height, props.title],
  () => update(),
  { deep: true },
)
</script>

<template>
  <div class="uchart">
    <p v-if="hint" class="chart-hint">{{ hint }}</p>
    <div v-if="x.length === 0" class="empty">暂无数据，开训后这里会出现曲线</div>
    <div ref="root" class="plot-host" :class="{ hidden: x.length === 0 }" />
  </div>
</template>

<style scoped>
.uchart {
  width: 100%;
}
.chart-hint {
  margin: 0 0 6px;
  font-size: 0.78rem;
  color: var(--text-dim);
  line-height: 1.4;
}
.plot-host {
  width: 100%;
}
.plot-host.hidden {
  display: none;
}
.empty {
  min-height: 120px;
}
</style>
