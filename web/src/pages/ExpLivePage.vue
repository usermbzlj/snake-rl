<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  api,
  type ConfigSchema,
  type ExperimentDetail,
  type ExperimentEvent,
  type MetricRow,
} from '@/api'
import LiveTuner from '@/components/LiveTuner.vue'
import MetricCharts from '@/components/MetricCharts.vue'
import StatusPill from '@/components/StatusPill.vue'
import WatchGrid from '@/components/WatchGrid.vue'
import { initLiveDraft, useExperimentStream } from '@/composables/useExperimentStream'
import { useWatch } from '@/composables/useWatch'
import { formatNumber, formatSteps } from '@/utils/format'
import { bestEvalScore, livePatchMarkers } from '@/utils/metrics'

const route = useRoute()
const router = useRouter()
const id = computed(() => String(route.params.id))

const detail = ref<ExperimentDetail | null>(null)
const schema = ref<ConfigSchema | null>(null)
const metrics = ref<MetricRow[]>([])
const events = ref<ExperimentEvent[]>([])
const error = ref('')
const loading = ref(true)
const liveDraft = ref<Record<string, number>>({})
const patching = ref(false)

const { wsStatus, connect: connectExp } = useExperimentStream(
  id,
  detail,
  metrics,
  events,
  liveDraft,
  error,
)
const { frame, watchInfo, watchCfg, connect: connectWatch } = useWatch(id)

const status = computed(() => detail.value?.experiment.status ?? 'created')
const algo = computed(() => detail.value?.experiment.algo ?? 'ppo')
const markers = computed(() => livePatchMarkers(events.value, schema.value))
const last = computed(() => metrics.value[metrics.value.length - 1])
const bestEval = computed(() => bestEvalScore(metrics.value))

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
    const liveKeys = s.groups.flatMap((g) => g.fields).filter((f) => f.live).map((f) => f.key)
    liveDraft.value = initLiveDraft(d.experiment.config, liveKeys)
  } catch (e) {
    error.value = e instanceof Error ? e.message : '加载失败'
  } finally {
    loading.value = false
  }
}

function connectAll() {
  connectExp()
  connectWatch()
}

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
    if (schema.value) {
      for (const f of schema.value.groups.flatMap((g) => g.fields)) {
        if (!f.live || (f.algo && f.algo !== algo.value)) continue
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

function onWatchCfg(partial: Partial<typeof watchCfg>) {
  Object.assign(watchCfg, partial)
}

function onLiveDraft(key: string, value: number) {
  liveDraft.value = { ...liveDraft.value, [key]: value }
}

onMounted(async () => {
  await load()
  connectAll()
})

watch(id, async () => {
  await load()
  connectAll()
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
        <p>看 AI 怎么玩、分数怎么涨；右侧可实时拧奖励旋钮，曲线上会留下标记。</p>
      </div>
      <div v-if="detail" class="header-actions">
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
        <button v-else type="button" class="btn btn-sm btn-primary" @click="doAction('start')">
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
        <div class="stat">
          <span class="dim">环境步数</span>
          <strong class="mono">{{ formatSteps(last?.env_steps) }}</strong>
        </div>
        <div class="stat">
          <span class="dim">平均得分</span>
          <strong>{{ formatNumber(last?.score_mean, 2) }}</strong>
        </div>
        <div class="stat">
          <span class="dim">最高得分</span>
          <strong>{{ formatNumber(last?.score_max, 1) }}</strong>
        </div>
        <div class="stat">
          <span class="dim">吞吐</span>
          <strong>{{ formatNumber(last?.sps, 0) }} 步/秒</strong>
        </div>
        <div class="stat" :title="wsStatus === 'open' ? '' : '与服务器的实时连接已断开，正在重连…'">
          <span class="dim">最佳评估得分</span>
          <strong>{{ formatNumber(bestEval, 2) }}</strong>
        </div>
      </div>

      <div class="live-grid">
        <WatchGrid
          :watch-cfg="watchCfg"
          :frame="frame"
          :watch-info="watchInfo"
          @update:watch-cfg="onWatchCfg"
        />
        <aside class="side">
          <LiveTuner
            :schema="schema"
            :algo="algo"
            :live-draft="liveDraft"
            :status="status"
            :patching="patching"
            @update:live-draft="onLiveDraft"
            @apply="applyLive"
          />
        </aside>
      </div>

      <MetricCharts :metrics="metrics" :markers="markers" :algo="algo" />
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
  grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.55fr);
  gap: 18px;
  align-items: start;
}
@media (max-width: 1100px) {
  .live-grid {
    grid-template-columns: 1fr;
  }
}
</style>
