<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  api,
  connectExperimentWs,
  connectWatchWs,
  type ConfigSchema,
  type ExperimentDetail,
  type ExperimentEvent,
  type FieldSchema,
  type MetricRow,
  type WatchFrame,
  type WatchGame,
  type WsStatus,
} from '@/api'
import FieldControl from '@/components/FieldControl.vue'
import ProbBars from '@/components/ProbBars.vue'
import SnakeBoard from '@/components/SnakeBoard.vue'
import StatusPill from '@/components/StatusPill.vue'
import UChart, { type ChartMarker } from '@/components/UChart.vue'
import { formatNumber, formatSteps, getByPath, setByPath } from '@/utils/format'

const route = useRoute()
const router = useRouter()
const id = computed(() => String(route.params.id))

const detail = ref<ExperimentDetail | null>(null)
const schema = ref<ConfigSchema | null>(null)
const metrics = ref<MetricRow[]>([])
const events = ref<ExperimentEvent[]>([])
const error = ref('')
const loading = ref(true)
const wsStatus = ref<WsStatus>('closed')
const frame = ref<WatchFrame | null>(null)
const watchInfo = ref('')

const watchCfg = reactive({
  games: 4 as 1 | 4 | 9,
  board_size: 8,
  speed: 10,
  greedy: true,
})

const liveDraft = ref<Record<string, number>>({})
const patching = ref(false)
const showLiveAdvanced = ref(false)

let expWs: ReturnType<typeof connectExperimentWs> | null = null
let watchWs: ReturnType<typeof connectWatchWs> | null = null

const status = computed(() => detail.value?.experiment.status ?? 'created')
const algo = computed(() => detail.value?.experiment.algo ?? 'ppo')

const liveFields = computed((): FieldSchema[] => {
  if (!schema.value) return []
  return schema.value.groups
    .flatMap((g) => g.fields)
    .filter((f) => f.live && (!f.algo || f.algo === algo.value) && !f.advanced)
})

const liveAdvancedFields = computed((): FieldSchema[] => {
  if (!schema.value || !showLiveAdvanced.value) return []
  return schema.value.groups
    .flatMap((g) => g.fields)
    .filter((f) => f.live && (!f.algo || f.algo === algo.value) && f.advanced)
})

const xs = computed(() => metrics.value.map((m) => m.env_steps ?? 0))

const markers = computed((): ChartMarker[] =>
  events.value
    .filter((e) => e.type === 'live_patch')
    .map((e) => {
      const changes = (e.data?.changes ?? {}) as Record<string, [unknown, unknown]>
      const keys = Object.keys(changes)
      const label = keys[0] ? keys[0].split('.').pop()! : '调参'
      return { x: e.env_steps, label: `调参·${label}` }
    }),
)

const games = computed((): WatchGame[] => frame.value?.games ?? [])

const gridClass = computed(() => {
  if (watchCfg.games === 1) return 'g1'
  if (watchCfg.games === 4) return 'g4'
  return 'g9'
})

async function load() {
  loading.value = true
  error.value = ''
  try {
    const [d, s] = await Promise.all([api.getExperiment(id.value), api.configSchema()])
    detail.value = d
    schema.value = s
    metrics.value = d.metrics
    events.value = d.events
    watchCfg.board_size = d.experiment.config.env.max_size
    // init live draft from config
    const draft: Record<string, number> = {}
    for (const f of s.groups.flatMap((g) => g.fields).filter((f) => f.live)) {
      const v = getByPath(d.experiment.config, f.key)
      if (typeof v === 'number') draft[f.key] = v
    }
    liveDraft.value = draft
  } catch (e) {
    error.value = e instanceof Error ? e.message : '加载失败'
  } finally {
    loading.value = false
  }
}

function connect() {
  expWs?.close()
  watchWs?.close()
  expWs = connectExperimentWs(
    id.value,
    (msg) => {
      if (msg.type === 'hello' && detail.value) {
        detail.value.experiment.status = msg.status
      } else if (msg.type === 'status' && detail.value) {
        detail.value.experiment.status = msg.status
        if (msg.error) error.value = msg.error
      } else if (msg.type === 'metrics') {
        metrics.value = [...metrics.value, msg.row]
      } else if (msg.type === 'event') {
        events.value = [...events.value, msg.event]
        if (msg.event.type === 'live_patch' && detail.value) {
          const changes = (msg.event.data?.changes ?? {}) as Record<string, [unknown, unknown]>
          for (const [k, [, neu]] of Object.entries(changes)) {
            setByPath(detail.value.experiment.config as unknown as Record<string, unknown>, k, neu)
            if (typeof neu === 'number') liveDraft.value[k] = neu
          }
        }
      }
    },
    (s) => {
      wsStatus.value = s
    },
  )

  watchWs = connectWatchWs(id.value, (msg) => {
    if (msg.type === 'frame') {
      frame.value = msg
      watchInfo.value = ''
    } else if (msg.type === 'info') {
      watchInfo.value = msg.message
    }
  })
  sendWatchConfig()
}

function sendWatchConfig() {
  watchWs?.send({
    type: 'config',
    games: watchCfg.games,
    board_size: watchCfg.board_size,
    speed: watchCfg.speed,
    greedy: watchCfg.greedy,
  })
}

watch(watchCfg, () => sendWatchConfig(), { deep: true })

async function doAction(act: 'start' | 'pause' | 'resume' | 'stop') {
  try {
    const s = await api.experimentAction(id.value, act)
    if (detail.value) detail.value.experiment.status = s.status
  } catch (e) {
    error.value = e instanceof Error ? e.message : '操作失败'
  }
}

async function applyLive() {
  patching.value = true
  try {
    const patch: Record<string, number> = {}
    const fields = [...liveFields.value, ...liveAdvancedFields.value]
    // Always include all live fields from draft (even if advanced panel collapsed)
    if (schema.value) {
      for (const f of schema.value.groups.flatMap((g) => g.fields)) {
        if (!f.live || (f.algo && f.algo !== algo.value)) continue
        const v = liveDraft.value[f.key]
        if (typeof v === 'number') patch[f.key] = v
      }
    } else {
      for (const f of fields) {
        const v = liveDraft.value[f.key]
        if (typeof v === 'number') patch[f.key] = v
      }
    }
    const res = await api.livePatch(id.value, patch)
    if (detail.value) detail.value.experiment.config = res.config
  } catch (e) {
    error.value = e instanceof Error ? e.message : '调参失败'
  } finally {
    patching.value = false
  }
}

const last = computed(() => metrics.value[metrics.value.length - 1])

onMounted(async () => {
  await load()
  connect()
})

onUnmounted(() => {
  expWs?.close()
  watchWs?.close()
})

watch(id, async () => {
  await load()
  connect()
})
</script>

<template>
  <div class="page live-page">
    <header class="page-header">
      <div>
        <p class="crumb">
          <RouterLink to="/">实验室</RouterLink>
          <span>/</span>
          <span>{{ detail?.experiment.name ?? '…' }}</span>
        </p>
        <h1>训练实况</h1>
        <p>左边看 AI 怎么玩，右边看分数怎么涨。下方可实时拧奖励旋钮。</p>
      </div>
      <div class="header-actions" v-if="detail">
        <StatusPill :status="status" />
        <button
          v-if="status === 'running'"
          type="button"
          class="btn btn-sm"
          @click="doAction('pause')"
        >
          暂停
        </button>
        <button
          v-else-if="status === 'paused'"
          type="button"
          class="btn btn-sm btn-primary"
          @click="doAction('resume')"
        >
          继续
        </button>
        <button
          v-else
          type="button"
          class="btn btn-sm btn-primary"
          @click="doAction('start')"
        >
          开始
        </button>
        <button
          v-if="status === 'running' || status === 'paused'"
          type="button"
          class="btn btn-sm"
          @click="doAction('stop')"
        >
          停止
        </button>
        <button type="button" class="btn btn-sm btn-ghost" @click="router.push(`/exp/${id}/inspect`)">
          AI 在想什么
        </button>
      </div>
    </header>

    <div v-if="loading" class="loading">加载实验…</div>
    <p v-else-if="error" class="error-banner" role="alert">{{ error }}</p>

    <template v-else>
      <div class="summary-row">
        <div class="stat"><span class="dim">环境步数</span><strong class="mono">{{ formatSteps(last?.env_steps) }}</strong></div>
        <div class="stat"><span class="dim">平均得分</span><strong>{{ formatNumber(last?.score_mean, 2) }}</strong></div>
        <div class="stat"><span class="dim">最高得分</span><strong>{{ formatNumber(last?.score_max, 1) }}</strong></div>
        <div class="stat"><span class="dim">吞吐</span><strong>{{ formatNumber(last?.sps, 0) }} 步/秒</strong></div>
        <div class="stat"><span class="dim">WS</span><strong>{{ wsStatus }}</strong></div>
      </div>

      <div class="live-grid">
        <section class="panel watch-panel">
          <div class="watch-toolbar">
            <label>
              画面
              <select v-model.number="watchCfg.games" aria-label="同时观看局数">
                <option :value="1">1 局</option>
                <option :value="4">4 局</option>
                <option :value="9">9 局</option>
              </select>
            </label>
            <label>
              盘面
              <input
                v-model.number="watchCfg.board_size"
                type="number"
                min="5"
                max="32"
                aria-label="观看棋盘大小"
              />
            </label>
            <label>
              速度
              <input
                v-model.number="watchCfg.speed"
                type="range"
                min="1"
                max="60"
                aria-label="播放速度"
              />
              <span class="mono">{{ watchCfg.speed }}</span>
            </label>
            <label class="chk">
              <input v-model="watchCfg.greedy" type="checkbox" />
              贪心（少随机）
            </label>
          </div>

          <p v-if="watchInfo" class="info-msg">{{ watchInfo }}</p>

          <div class="game-grid" :class="gridClass">
            <div v-for="(g, i) in games" :key="i" class="game-cell">
              <SnakeBoard
                :size="frame?.board_size ?? watchCfg.board_size"
                :snake="g.snake"
                :food="g.food"
                :dir="g.dir"
                :dead="g.dead"
                :max-css-size="watchCfg.games === 1 ? 420 : watchCfg.games === 4 ? 260 : 180"
              />
              <div class="game-foot">
                <span class="score">得分 {{ g.score }}</span>
                <span class="dim">V={{ formatNumber(g.value, 2) }}</span>
              </div>
              <ProbBars :probs="g.probs" :chosen="g.action" compact />
            </div>
          </div>
          <div v-if="games.length === 0" class="empty">等待模型画面…权重就绪后会自动播放。</div>
        </section>

        <aside class="side">
          <section class="panel">
            <h2 class="panel-title">实时调参</h2>
            <p class="dim tip">只改「实时」字段。应用后曲线上会出现竖线标记。</p>
            <FieldControl
              v-for="f in liveFields"
              :key="f.key"
              :field="f"
              :model-value="liveDraft[f.key]"
              @update:model-value="liveDraft[f.key] = $event as number"
            />
            <button
              type="button"
              class="btn btn-ghost btn-sm"
              style="margin-bottom: 8px"
              @click="showLiveAdvanced = !showLiveAdvanced"
            >
              {{ showLiveAdvanced ? '收起进阶实时参数' : '展开进阶实时参数' }}
            </button>
            <FieldControl
              v-for="f in liveAdvancedFields"
              :key="f.key"
              :field="f"
              :model-value="liveDraft[f.key]"
              @update:model-value="liveDraft[f.key] = $event as number"
            />
            <button
              type="button"
              class="btn btn-primary"
              :disabled="patching || status !== 'running'"
              @click="applyLive"
            >
              {{ patching ? '应用中…' : '应用调参' }}
            </button>
          </section>
        </aside>
      </div>

      <section class="charts panel">
        <h2 class="panel-title">训练曲线</h2>
        <div class="chart-grid">
          <UChart
            title="得分"
            :x="xs"
            :series="[
              { label: '平均得分', data: metrics.map((m) => m.score_mean) },
              { label: '最高得分', data: metrics.map((m) => m.score_max), color: '#f0c35a' },
            ]"
            :markers="markers"
          />
          <UChart
            title="评估得分"
            :x="xs"
            :series="[{ label: 'eval 均分', data: metrics.map((m) => m.eval_score_mean), color: '#6db3f2' }]"
            :markers="markers"
          />
          <UChart
            title="死亡原因"
            :x="xs"
            :series="[
              { label: '撞墙', data: metrics.map((m) => m.death_wall), color: '#f07178' },
              { label: '撞自己', data: metrics.map((m) => m.death_self), color: '#f0c35a' },
              { label: '饿死', data: metrics.map((m) => m.death_starve), color: '#b794f6' },
            ]"
            :markers="markers"
          />
          <UChart
            v-if="algo === 'ppo'"
            title="熵"
            :x="xs"
            :series="[{ label: 'entropy', data: metrics.map((m) => m.entropy) }]"
            :markers="markers"
          />
          <UChart
            v-else
            title="探索 ε"
            :x="xs"
            :series="[{ label: 'epsilon', data: metrics.map((m) => m.epsilon), color: '#f0c35a' }]"
            :markers="markers"
          />
          <UChart
            title="损失"
            :x="xs"
            :series="
              algo === 'ppo'
                ? [
                    { label: 'policy', data: metrics.map((m) => m.loss_policy) },
                    { label: 'value', data: metrics.map((m) => m.loss_value), color: '#6db3f2' },
                  ]
                : [{ label: 'Q loss', data: metrics.map((m) => m.loss_q), color: '#f07178' }]
            "
            :markers="markers"
          />
        </div>
      </section>
    </template>
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
.header-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.error-banner {
  padding: 12px 14px;
  border-radius: var(--radius);
  background: rgba(240, 113, 120, 0.12);
  border: 1px solid rgba(240, 113, 120, 0.35);
  color: #ffc4c8;
}
.summary-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}
@media (max-width: 800px) {
  .summary-row {
    grid-template-columns: repeat(2, 1fr);
  }
}
.stat {
  padding: 10px 12px;
  border-radius: var(--radius);
  border: 1px solid var(--stroke);
  background: rgba(0, 0, 0, 0.2);
}
.stat span {
  display: block;
  font-size: 0.72rem;
}
.stat strong {
  font-size: 1.05rem;
}
.live-grid {
  display: grid;
  grid-template-columns: 1fr 300px;
  gap: 16px;
  align-items: start;
}
@media (max-width: 960px) {
  .live-grid {
    grid-template-columns: 1fr;
  }
}
.watch-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
  font-size: 0.85rem;
}
.watch-toolbar label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.watch-toolbar select,
.watch-toolbar input[type='number'] {
  width: 72px;
  min-height: 32px;
  padding: 4px 8px;
  border-radius: 8px;
  border: 1px solid var(--stroke);
  background: var(--bg-0);
}
.watch-toolbar input[type='range'] {
  width: 100px;
  accent-color: var(--accent);
}
.info-msg {
  color: var(--warn);
  font-size: 0.85rem;
}
.game-grid {
  display: grid;
  gap: 12px;
}
.game-grid.g1 {
  grid-template-columns: 1fr;
}
.game-grid.g4 {
  grid-template-columns: 1fr 1fr;
}
.game-grid.g9 {
  grid-template-columns: repeat(3, 1fr);
}
@media (max-width: 700px) {
  .game-grid.g4,
  .game-grid.g9 {
    grid-template-columns: 1fr 1fr;
  }
}
@media (max-width: 420px) {
  .game-grid.g4,
  .game-grid.g9 {
    grid-template-columns: 1fr;
  }
}
.game-cell {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.game-foot {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
}
.score {
  font-weight: 700;
  color: var(--accent);
}
.tip {
  font-size: 0.8rem;
  margin: -4px 0 12px;
}
.charts {
  margin-top: 16px;
}
.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 800px) {
  .chart-grid {
    grid-template-columns: 1fr;
  }
}
</style>
