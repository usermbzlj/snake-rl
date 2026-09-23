<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import uPlot from 'uplot'

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
    x: number[]
    series: ChartSeries[]
    markers?: ChartMarker[]
    height?: number
    cursorX?: number | null
  }>(),
  {
    title: '',
    markers: () => [],
    height: 200,
    cursorX: null,
  },
)

const root = ref<HTMLDivElement | null>(null)
let plot: uPlot | null = null
let ro: ResizeObserver | null = null

const COLORS = ['#3dd6c6', '#6db3f2', '#f0c35a', '#f07178', '#b794f6', '#5ddea0']

function buildOpts(width: number): uPlot.Options {
  const series: uPlot.Series[] = [{ label: 'x' }]
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
      },
      {
        stroke: '#6b7c9c',
        grid: { stroke: 'rgba(148,178,220,0.08)', width: 1 },
        ticks: { stroke: 'rgba(148,178,220,0.2)' },
        font: '11px Segoe UI, sans-serif',
      },
    ],
    legend: { show: true, live: true },
    cursor: {
      drag: { x: false, y: false },
    },
    hooks: {
      draw: [
        (u) => {
          const ctx = u.ctx
          for (const m of props.markers) {
            const cx = u.valToPos(m.x, 'x', true)
            if (cx < u.bbox.left || cx > u.bbox.left + u.bbox.width) continue
            ctx.save()
            ctx.strokeStyle = m.color ?? 'rgba(240,195,90,0.75)'
            ctx.lineWidth = 1.5
            ctx.setLineDash([4, 3])
            ctx.beginPath()
            ctx.moveTo(cx, u.bbox.top)
            ctx.lineTo(cx, u.bbox.top + u.bbox.height)
            ctx.stroke()
            ctx.setLineDash([])
            ctx.fillStyle = m.color ?? '#f0c35a'
            ctx.font = '10px Segoe UI, sans-serif'
            ctx.fillText(m.label, cx + 4, u.bbox.top + 12)
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
}

onMounted(() => {
  recreate()
  if (root.value) {
    ro = new ResizeObserver(() => {
      if (!plot || !root.value) return
      plot.setSize({ width: root.value.clientWidth || 320, height: props.height })
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
    <div v-if="x.length === 0" class="empty">暂无数据，开训后这里会出现曲线</div>
    <div ref="root" class="plot-host" :class="{ hidden: x.length === 0 }" />
  </div>
</template>

<style scoped>
.uchart {
  width: 100%;
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
