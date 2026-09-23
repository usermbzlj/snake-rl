<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api, ApiError, type ExperimentSummary, type Trajectory } from '@/api'
import SnakeBoard from '@/components/SnakeBoard.vue'
import StatusPill from '@/components/StatusPill.vue'
import UChart from '@/components/UChart.vue'
import { downsampleEven, formatNumber } from '@/utils/format'

const list = ref<ExperimentSummary[]>([])
const selected = ref<string[]>([])
const metricKey = ref('score_mean')
const loading = ref(true)
const error = ref('')

const boardSize = ref(8)
const seed = ref(42)
const comparing = ref(false)
const trajectories = ref<Trajectory[]>([])
const playIdx = ref(0)
const playing = ref(false)
let timer: ReturnType<typeof setInterval> | null = null

const metricOptions = [
  { key: 'score_mean', label: '平均得分' },
  { key: 'score_max', label: '最高得分' },
  { key: 'eval_score_mean', label: '评估均分' },
  { key: 'return_mean', label: '回报' },
  { key: 'entropy', label: '熵' },
  { key: 'epsilon', label: 'ε' },
  { key: 'loss_policy', label: '策略损失' },
  { key: 'loss_q', label: 'Q 损失' },
]

const COLORS = ['#3dd6c6', '#6db3f2', '#f0c35a', '#f07178']

interface CurveBundle {
  labels: string[]
  series: { label: string; data: (number | null)[]; color: string }[]
  x: number[]
}

const curves = ref<CurveBundle | null>(null)

function toggle(id: string) {
  if (selected.value.includes(id)) {
    selected.value = selected.value.filter((x) => x !== id)
  } else if (selected.value.length < 4) {
    selected.value = [...selected.value, id]
  }
}

async function loadList() {
  loading.value = true
  try {
    list.value = await api.listExperiments()
    if (selected.value.length === 0 && list.value[0]) {
      selected.value = [list.value[0].id]
      if (list.value[1]) selected.value.push(list.value[1].id)
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : '加载失败'
  } finally {
    loading.value = false
  }
}

async function loadCurves() {
  if (selected.value.length < 1) {
    curves.value = null
    return
  }
  error.value = ''
  try {
    const details = await Promise.all(selected.value.map((id) => api.getExperiment(id)))
    // Align on union of env_steps via per-series sparse — use each exp's own x overlapping visually
    // Simpler: resample each to shared max-length grid by env_steps interpolation isn't needed;
    // overlay with separate x by padding — uPlot wants shared x. Use downsample to common length
    // and plot vs index of env_steps from first, showing each series vs its own steps on a merged axis.
    const allSteps = new Set<number>()
    for (const d of details) {
      for (const m of d.metrics) {
        if (m.env_steps != null) allSteps.add(m.env_steps)
      }
    }
    const x = [...allSteps].sort((a, b) => a - b)
    const xUse = downsampleEven(x, 400)
    const series = details.map((d, i) => {
      const map = new Map<number, number>()
      for (const m of d.metrics) {
        const v = m[metricKey.value]
        if (m.env_steps != null && typeof v === 'number') map.set(m.env_steps, v)
      }
      // forward-fill
      let last: number | null = null
      const data = xUse.map((step) => {
        if (map.has(step)) last = map.get(step)!
        return last
      })
      return {
        label: d.experiment.name,
        data,
        color: COLORS[i % COLORS.length]!,
      }
    })
    curves.value = {
      labels: details.map((d) => d.experiment.name),
      series,
      x: xUse,
    }
  } catch (e) {
    error.value = e instanceof ApiError ? e.detail : e instanceof Error ? e.message : '曲线加载失败'
  }
}

async function runCompare() {
  if (selected.value.length < 2) {
    error.value = '请至少选择 2 个实验做同局对战'
    return
  }
  comparing.value = true
  error.value = ''
  stopPlay()
  try {
    const res = await api.compare({
      entries: selected.value.map((experiment_id) => ({
        experiment_id,
        checkpoint: 'best',
      })),
      board_size: boardSize.value,
      seed: seed.value,
    })
    trajectories.value = res.trajectories
    playIdx.value = 0
  } catch (e) {
    error.value = e instanceof ApiError ? e.detail : e instanceof Error ? e.message : '对比失败'
    trajectories.value = []
  } finally {
    comparing.value = false
  }
}

function stopPlay() {
  playing.value = false
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

function togglePlay() {
  if (playing.value) {
    stopPlay()
    return
  }
  const maxLen = Math.max(...trajectories.value.map((t) => t.steps.length), 0)
  if (!maxLen) return
  playing.value = true
  timer = setInterval(() => {
    if (playIdx.value >= maxLen - 1) {
      stopPlay()
      return
    }
    playIdx.value += 1
  }, 120)
}

const maxSteps = computed(() => Math.max(...trajectories.value.map((t) => t.steps.length), 1))

watch([selected, metricKey], () => void loadCurves())
onMounted(async () => {
  await loadList()
  await loadCurves()
})
onUnmounted(() => stopPlay())
</script>

<template>
  <div class="page">
    <header class="page-header">
      <div>
        <h1>对比</h1>
        <p>叠曲线看谁学得快；同一种子同盘对战，并排看谁走得更聪明。</p>
      </div>
    </header>

    <div v-if="loading" class="loading">加载实验列表…</div>
    <p v-if="error" class="error-banner" role="alert">{{ error }}</p>

    <section v-else class="panel">
      <h2 class="panel-title">选择实验（2–4 个）</h2>
      <ul class="pick-list">
        <li v-for="exp in list" :key="exp.id">
          <label class="pick">
            <input
              type="checkbox"
              :checked="selected.includes(exp.id)"
              :disabled="!selected.includes(exp.id) && selected.length >= 4"
              @change="toggle(exp.id)"
            />
            <span class="name">{{ exp.name }}</span>
            <StatusPill :status="exp.status" />
            <span class="dim">{{ exp.algo.toUpperCase() }}</span>
          </label>
        </li>
      </ul>
      <div v-if="list.length === 0" class="empty">还没有实验可对比。</div>
    </section>

    <section class="panel" style="margin-top: 16px">
      <div class="curve-head">
        <h2 class="panel-title">指标曲线</h2>
        <label>
          指标
          <select v-model="metricKey" aria-label="对比指标">
            <option v-for="o in metricOptions" :key="o.key" :value="o.key">{{ o.label }}</option>
          </select>
        </label>
      </div>
      <UChart
        v-if="curves"
        :x="curves.x"
        :series="curves.series"
        :height="260"
      />
      <div v-else class="empty">勾选实验后显示叠加曲线</div>
    </section>

    <section class="panel" style="margin-top: 16px">
      <h2 class="panel-title">同局对战</h2>
      <div class="battle-controls">
        <label>
          棋盘
          <input v-model.number="boardSize" type="number" min="5" max="32" />
        </label>
        <label>
          种子
          <input v-model.number="seed" type="number" />
        </label>
        <button
          type="button"
          class="btn btn-primary"
          :disabled="comparing || selected.length < 2"
          @click="runCompare"
        >
          {{ comparing ? '对战中…' : '开始同局对战' }}
        </button>
        <button
          v-if="trajectories.length"
          type="button"
          class="btn"
          @click="togglePlay"
        >
          {{ playing ? '暂停' : '播放' }}
        </button>
      </div>

      <div v-if="trajectories.length" class="battle-grid">
        <div v-for="(t, i) in trajectories" :key="i" class="battle-cell">
          <h3>{{ t.name ?? t.experiment_id }}</h3>
          <SnakeBoard
            v-if="t.steps[Math.min(playIdx, t.steps.length - 1)]"
            :size="t.board_size"
            :snake="t.steps[Math.min(playIdx, t.steps.length - 1)]!.snake.filter((p) => p[0] >= 0)"
            :food="t.steps[Math.min(playIdx, t.steps.length - 1)]!.food"
            :dir="t.steps[Math.min(playIdx, t.steps.length - 1)]!.dir"
            :max-css-size="280"
          />
          <p class="dim">
            得分
            {{ t.steps[Math.min(playIdx, t.steps.length - 1)]?.score ?? 0 }}
            · 终局 {{ formatNumber(t.result.score, 0) }}
          </p>
        </div>
      </div>
      <input
        v-if="trajectories.length"
        v-model.number="playIdx"
        type="range"
        min="0"
        :max="maxSteps - 1"
        aria-label="对战时间轴"
        style="width: 100%; margin-top: 12px"
      />
    </section>
  </div>
</template>

<style scoped>
.error-banner {
  padding: 12px 14px;
  border-radius: var(--radius);
  background: rgba(240, 113, 120, 0.12);
  color: #ffc4c8;
}
.pick-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.pick {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--radius);
  border: 1px solid var(--stroke);
  background: rgba(0, 0, 0, 0.15);
  cursor: pointer;
}
.pick .name {
  font-weight: 650;
  flex: 1;
}
.curve-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.curve-head select,
.battle-controls input {
  min-height: 34px;
  padding: 4px 8px;
  border-radius: 8px;
  border: 1px solid var(--stroke);
  background: var(--bg-0);
}
.battle-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}
.battle-controls label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85rem;
}
.battle-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}
.battle-cell h3 {
  margin: 0 0 8px;
  font-size: 0.9rem;
}
</style>
