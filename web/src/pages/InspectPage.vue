<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { api, type CheckpointInfo, type Trajectory } from '@/api'
import ProbBars from '@/components/ProbBars.vue'
import SnakeBoard from '@/components/SnakeBoard.vue'
import UChart from '@/components/UChart.vue'
import { causeLabel, formatNumber } from '@/utils/format'

const route = useRoute()
const id = computed(() => String(route.params.id))

const checkpoints = ref<CheckpointInfo[]>([])
const checkpoint = ref<'latest' | 'best'>('best')
const boardSize = ref(8)
const seed = ref(7)
const greedy = ref(true)
const traj = ref<Trajectory | null>(null)
const loading = ref(false)
const error = ref('')
const showHeat = ref(true)

const idx = ref(0)
const playing = ref(false)
const speed = ref(8)
let timer: ReturnType<typeof setInterval> | null = null

const step = computed(() => traj.value?.steps[idx.value] ?? null)
const heat = computed(() => {
  if (!traj.value?.saliency || !showHeat.value) return null
  return traj.value.saliency[idx.value] ?? null
})

const valueSeries = computed(() => traj.value?.steps.map((s) => s.value) ?? [])
const xs = computed(() => traj.value?.steps.map((_, i) => i) ?? [])

const componentLabels = ['食物', '死亡', '步进', '靠近', '饿死', '通关']

async function boot() {
  error.value = ''
  try {
    const [cps, detail] = await Promise.all([api.checkpoints(id.value), api.getExperiment(id.value)])
    checkpoints.value = cps
    boardSize.value = detail.experiment.config.env.max_size
  } catch (e) {
    error.value = e instanceof Error ? e.message : '加载失败'
  }
}

async function run() {
  loading.value = true
  error.value = ''
  playing.value = false
  try {
    traj.value = await api.inspect(id.value, {
      checkpoint: checkpoint.value,
      board_size: boardSize.value,
      seed: seed.value,
      greedy: greedy.value,
    })
    idx.value = 0
  } catch (e) {
    error.value = e instanceof Error ? e.message : '推演失败'
  } finally {
    loading.value = false
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
  if (!traj.value) return
  playing.value = true
  timer = setInterval(() => {
    if (!traj.value) return
    if (idx.value >= traj.value.steps.length - 1) {
      stopPlay()
      return
    }
    idx.value += 1
  }, Math.max(40, 1000 / speed.value))
}

function jumpBeforeDeath() {
  if (!traj.value || traj.value.steps.length < 2) return
  idx.value = Math.max(0, traj.value.steps.length - 2)
  stopPlay()
}

watch(speed, () => {
  if (playing.value) {
    stopPlay()
    togglePlay()
  }
})

onMounted(() => void boot())
onUnmounted(() => stopPlay())
</script>

<template>
  <div class="page">
    <header class="page-header">
      <div>
        <p class="crumb">
          <RouterLink to="/">实验室</RouterLink>
          <span>/</span>
          <RouterLink :to="`/exp/${id}`">训练实况</RouterLink>
          <span>/</span>
          <span>洞察</span>
        </p>
        <h1>AI 在想什么</h1>
        <p>回放一整局：动作概率、价值估计，以及「它在看棋盘哪儿」的热力图。</p>
      </div>
    </header>

    <section class="panel controls">
      <div class="row">
        <label>
          检查点
          <select v-model="checkpoint" aria-label="检查点">
            <option value="best">最佳</option>
            <option value="latest">最新</option>
          </select>
        </label>
        <label>
          棋盘
          <input v-model.number="boardSize" type="number" min="5" max="32" aria-label="棋盘大小" />
        </label>
        <label>
          种子
          <input v-model.number="seed" type="number" aria-label="随机种子" />
        </label>
        <label class="chk">
          <input v-model="greedy" type="checkbox" />
          贪心
        </label>
        <button type="button" class="btn btn-primary" :disabled="loading" @click="run">
          {{ loading ? '推演中…' : '开始推演' }}
        </button>
      </div>
      <p v-if="checkpoints.length" class="dim cps">
        可用：
        <span v-for="c in checkpoints" :key="c.name"> {{ c.name }}@{{ c.env_steps }} </span>
      </p>
      <p v-if="error" class="error-banner" role="alert">{{ error }}</p>
    </section>

    <div v-if="!traj && !loading" class="empty">选好检查点，点「开始推演」。</div>

    <div v-else-if="traj" class="inspect-grid">
      <section class="panel board-side">
        <SnakeBoard
          v-if="step"
          :size="traj.board_size"
          :snake="step.snake.filter((p) => p[0] >= 0)"
          :food="step.food"
          :dir="step.dir"
          :heatmap="heat"
          :show-heatmap="showHeat"
          :max-css-size="440"
        />
        <label class="chk heat-tog">
          <input v-model="showHeat" type="checkbox" />
          显示注意力热力图
        </label>

        <div class="timeline">
          <input
            v-model.number="idx"
            type="range"
            min="0"
            :max="Math.max(0, traj.steps.length - 1)"
            aria-label="时间轴"
          />
          <div class="tl-actions">
            <button type="button" class="btn btn-sm" @click="idx = Math.max(0, idx - 1)">上一步</button>
            <button type="button" class="btn btn-sm btn-primary" @click="togglePlay">
              {{ playing ? '暂停' : '播放' }}
            </button>
            <button
              type="button"
              class="btn btn-sm"
              @click="idx = Math.min(traj.steps.length - 1, idx + 1)"
            >
              下一步
            </button>
            <label>
              速度
              <input v-model.number="speed" type="range" min="1" max="30" />
            </label>
            <button type="button" class="btn btn-sm btn-ghost" @click="jumpBeforeDeath">跳到死亡前</button>
          </div>
          <p class="dim">
            第 {{ idx + 1 }} / {{ traj.steps.length }} 步 · 得分 {{ step?.score ?? 0 }} · 结局
            {{ causeLabel(traj.result.cause) }}
          </p>
        </div>
      </section>

      <aside class="panel info-side" v-if="step">
        <h2 class="panel-title">这一步</h2>
        <ProbBars :probs="step.probs" :chosen="step.action" />
        <p class="val">价值估计 <strong class="mono">{{ formatNumber(step.value, 3) }}</strong></p>

        <h3>奖励分量</h3>
        <ul class="comps">
          <li v-for="(v, i) in step.reward_components" :key="i">
            <span>{{ componentLabels[i] }}</span>
            <strong class="mono" :class="{ pos: v > 0, neg: v < 0 }">{{ formatNumber(v, 3) }}</strong>
          </li>
        </ul>

        <h3>价值随时间</h3>
        <UChart
          :x="xs"
          :series="[{ label: 'value', data: valueSeries }]"
          :cursor-x="idx"
          :height="160"
        />
      </aside>
    </div>
  </div>
</template>

<style scoped>
.crumb {
  display: flex;
  gap: 8px;
  font-size: 0.8rem;
  color: var(--text-dim);
  margin: 0 0 4px;
}
.crumb a {
  color: var(--text-muted);
}
.controls .row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
}
.controls label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85rem;
}
.controls select,
.controls input[type='number'] {
  min-height: 34px;
  padding: 4px 8px;
  border-radius: 8px;
  border: 1px solid var(--stroke);
  background: var(--bg-0);
  width: 88px;
}
.cps {
  margin: 8px 0 0;
  font-size: 0.78rem;
}
.error-banner {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: var(--radius);
  background: rgba(240, 113, 120, 0.12);
  color: #ffc4c8;
}
.inspect-grid {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 16px;
  margin-top: 16px;
}
@media (max-width: 900px) {
  .inspect-grid {
    grid-template-columns: 1fr;
  }
}
.heat-tog {
  display: inline-flex;
  gap: 8px;
  margin: 10px 0;
  font-size: 0.85rem;
}
.timeline input[type='range'] {
  width: 100%;
  accent-color: var(--accent);
}
.tl-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin: 8px 0;
}
.val {
  margin: 12px 0;
}
.comps {
  list-style: none;
  margin: 0 0 16px;
  padding: 0;
}
.comps li {
  display: flex;
  justify-content: space-between;
  padding: 6px 0;
  border-bottom: 1px solid var(--stroke);
  font-size: 0.85rem;
}
.pos {
  color: var(--ok);
}
.neg {
  color: var(--danger);
}
h3 {
  margin: 16px 0 8px;
  font-size: 0.9rem;
}
</style>
