<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  api,
  type Algo,
  type ConfigSchema,
  type ExperimentConfig,
  type ExperimentSummary,
  type FieldSchema,
  type Preset,
} from '@/api'
import FieldControl from '@/components/FieldControl.vue'
import StatusPill from '@/components/StatusPill.vue'
import { deepClone, formatNumber, formatSteps, getByPath, setByPath } from '@/utils/format'

const router = useRouter()
const schema = ref<ConfigSchema | null>(null)
const list = ref<ExperimentSummary[]>([])
const loading = ref(true)
const creating = ref(false)
const error = ref('')
const showAdvanced = ref(false)
const draft = ref<ExperimentConfig | null>(null)
const selectedPreset = ref<string | null>(null)

const algo = computed({
  get: () => draft.value?.algo ?? 'ppo',
  set: (v: Algo) => {
    if (draft.value) draft.value.algo = v
  },
})

function visibleFields(fields: FieldSchema[]): FieldSchema[] {
  return fields.filter((f) => {
    if (f.algo && f.algo !== algo.value) return false
    if (f.advanced && !showAdvanced.value) return false
    return true
  })
}

const essentialGroups = computed(() => {
  if (!schema.value) return []
  return schema.value.groups
    .map((g) => ({
      ...g,
      fields: visibleFields(g.fields).filter((f) => !f.advanced),
    }))
    .filter((g) => g.fields.length > 0)
})

const advancedGroups = computed(() => {
  if (!schema.value || !showAdvanced.value) return []
  return schema.value.groups
    .map((g) => ({
      ...g,
      fields: visibleFields(g.fields).filter((f) => f.advanced),
    }))
    .filter((g) => g.fields.length > 0)
})

function fieldValue(key: string): unknown {
  if (!draft.value) return undefined
  return getByPath(draft.value, key)
}

function setField(key: string, value: unknown) {
  if (!draft.value) return
  if (key === 'algo') {
    draft.value.algo = value as Algo
    return
  }
  setByPath(draft.value as unknown as Record<string, unknown>, key, value)
}

function applyPreset(p: Preset) {
  selectedPreset.value = p.id
  draft.value = deepClone(p.config)
}

async function refreshList() {
  list.value = await api.listExperiments()
}

async function boot() {
  loading.value = true
  error.value = ''
  try {
    const [s, exps] = await Promise.all([api.configSchema(), api.listExperiments()])
    schema.value = s
    list.value = exps
    if (s.presets[0]) applyPreset(s.presets[0])
    else {
      // build from defaults
      const cfg: Record<string, unknown> = {
        name: '新实验',
        algo: 'ppo',
        env: {},
        reward: {},
        model: {},
        ppo: {},
        dqn: {},
        run: {},
      }
      for (const g of s.groups) {
        for (const f of g.fields) {
          setByPath(cfg, f.key, f.default)
        }
      }
      draft.value = cfg as unknown as ExperimentConfig
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : '加载失败'
  } finally {
    loading.value = false
  }
}

async function createExp() {
  if (!draft.value) return
  creating.value = true
  error.value = ''
  try {
    const summary = await api.createExperiment(deepClone(draft.value), true)
    await refreshList()
    await router.push(`/exp/${summary.id}`)
  } catch (e) {
    error.value = e instanceof Error ? e.message : '创建失败'
  } finally {
    creating.value = false
  }
}

async function action(id: string, act: 'start' | 'pause' | 'resume' | 'stop') {
  try {
    await api.experimentAction(id, act)
    await refreshList()
  } catch (e) {
    error.value = e instanceof Error ? e.message : '操作失败'
  }
}

async function cloneExp(exp: ExperimentSummary) {
  try {
    const name = `${exp.name} 副本`
    await api.cloneExperiment(exp.id, name, true)
    await refreshList()
  } catch (e) {
    error.value = e instanceof Error ? e.message : '克隆失败'
  }
}

async function removeExp(exp: ExperimentSummary) {
  if (!confirm(`确定删除「${exp.name}」？此操作不可恢复。`)) return
  try {
    await api.deleteExperiment(exp.id)
    await refreshList()
  } catch (e) {
    error.value = e instanceof Error ? e.message : '删除失败'
  }
}

function sparkPoints(spark: number[]): string {
  if (!spark.length) return ''
  const w = 120
  const h = 28
  const min = Math.min(...spark)
  const max = Math.max(...spark)
  const span = max - min || 1
  return spark
    .map((v, i) => {
      const x = (i / Math.max(1, spark.length - 1)) * w
      const y = h - ((v - min) / span) * (h - 4) - 2
      return `${x},${y}`
    })
    .join(' ')
}

onMounted(() => void boot())
watch(algo, () => {
  /* recompute visible fields via computed */
})
</script>

<template>
  <div class="page">
    <header class="page-header">
      <div>
        <h1>实验室</h1>
        <p>选一个预设，拧几下旋钮，按下开始——看着 AI 一点点学会吃豆。</p>
      </div>
    </header>

    <div v-if="loading" class="loading">正在加载配置…</div>
    <p v-else-if="error" class="error-banner" role="alert">{{ error }}</p>

    <div v-else class="lab-grid">
      <section class="panel create-panel" aria-labelledby="new-exp-title">
        <h2 id="new-exp-title" class="panel-title">新建实验</h2>

        <div class="presets" role="list">
          <button
            v-for="p in schema?.presets ?? []"
            :key="p.id"
            type="button"
            class="preset-card"
            :class="{ active: selectedPreset === p.id }"
            role="listitem"
            @click="applyPreset(p)"
          >
            <strong>{{ p.name }}</strong>
            <span>{{ p.description }}</span>
          </button>
        </div>

        <div class="algo-toggle" role="radiogroup" aria-label="算法">
          <button
            type="button"
            role="radio"
            :aria-checked="algo === 'ppo'"
            :class="{ on: algo === 'ppo' }"
            @click="algo = 'ppo'"
          >
            <strong>PPO</strong>
            <span>稳、适合入门，边玩边学</span>
          </button>
          <button
            type="button"
            role="radio"
            :aria-checked="algo === 'dqn'"
            :class="{ on: algo === 'dqn' }"
            @click="algo = 'dqn'"
          >
            <strong>DQN</strong>
            <span>记经验再学，风格更「试错」</span>
          </button>
        </div>

        <template v-if="draft">
          <div v-for="g in essentialGroups" :key="g.key" class="group">
            <h3>{{ g.label }}</h3>
            <p class="dim group-desc">{{ g.description }}</p>
            <FieldControl
              v-for="f in g.fields"
              :key="f.key"
              :field="f"
              :model-value="fieldValue(f.key)"
              @update:model-value="setField(f.key, $event)"
            />
          </div>

          <button type="button" class="btn btn-ghost btn-sm" @click="showAdvanced = !showAdvanced">
            {{ showAdvanced ? '收起进阶参数' : '展开进阶参数' }}
          </button>

          <div v-for="g in advancedGroups" :key="'adv-' + g.key" class="group">
            <h3>{{ g.label }} · 进阶</h3>
            <FieldControl
              v-for="f in g.fields"
              :key="f.key"
              :field="f"
              :model-value="fieldValue(f.key)"
              @update:model-value="setField(f.key, $event)"
            />
          </div>

          <div class="btn-row" style="margin-top: 12px">
            <button type="button" class="btn btn-primary" :disabled="creating" @click="createExp">
              {{ creating ? '启动中…' : '开始训练' }}
            </button>
          </div>
        </template>
      </section>

      <section class="panel list-panel" aria-labelledby="list-title">
        <div class="list-head">
          <h2 id="list-title" class="panel-title">我的实验</h2>
          <button type="button" class="btn btn-sm btn-ghost" @click="refreshList">刷新</button>
        </div>

        <div v-if="list.length === 0" class="empty">还没有实验。左边选个预设，点「开始训练」吧。</div>

        <ul v-else class="exp-list">
          <li v-for="exp in list" :key="exp.id" class="exp-card">
            <div class="exp-top">
              <div>
                <RouterLink class="exp-name" :to="`/exp/${exp.id}`">{{ exp.name }}</RouterLink>
                <div class="exp-meta muted">
                  {{ exp.algo.toUpperCase() }} · {{ exp.board[0] }}–{{ exp.board[1] }} ·
                  {{ formatSteps(exp.env_steps) }} 步
                </div>
              </div>
              <StatusPill :status="exp.status" />
            </div>

            <div class="exp-mid">
              <svg class="spark" viewBox="0 0 120 28" aria-hidden="true">
                <polyline
                  v-if="exp.spark.length"
                  fill="none"
                  stroke="var(--accent)"
                  stroke-width="2"
                  stroke-linejoin="round"
                  stroke-linecap="round"
                  :points="sparkPoints(exp.spark)"
                />
              </svg>
              <div class="stats">
                <div>
                  <span class="dim">最佳评估</span>
                  <strong>{{ formatNumber(exp.best_eval_score, 1) }}</strong>
                </div>
                <div>
                  <span class="dim">最近均分</span>
                  <strong>{{ formatNumber(exp.last?.score_mean, 1) }}</strong>
                </div>
              </div>
            </div>

            <div class="btn-row">
              <RouterLink class="btn btn-sm btn-primary" :to="`/exp/${exp.id}`">观看</RouterLink>
              <button
                v-if="exp.status === 'paused'"
                type="button"
                class="btn btn-sm"
                @click="action(exp.id, 'resume')"
              >
                继续
              </button>
              <button
                v-else-if="exp.status === 'stopped' || exp.status === 'created' || exp.status === 'finished'"
                type="button"
                class="btn btn-sm"
                @click="action(exp.id, 'start')"
              >
                继续
              </button>
              <button
                v-if="exp.status === 'running'"
                type="button"
                class="btn btn-sm"
                @click="action(exp.id, 'pause')"
              >
                暂停
              </button>
              <button
                v-if="exp.status === 'running' || exp.status === 'paused'"
                type="button"
                class="btn btn-sm"
                @click="action(exp.id, 'stop')"
              >
                停止
              </button>
              <button type="button" class="btn btn-sm btn-ghost" @click="cloneExp(exp)">克隆</button>
              <button type="button" class="btn btn-sm btn-danger" @click="removeExp(exp)">删除</button>
            </div>
          </li>
        </ul>
      </section>
    </div>
  </div>
</template>

<style scoped>
.lab-grid {
  display: grid;
  grid-template-columns: 1.05fr 0.95fr;
  gap: var(--space-4);
  align-items: start;
}
@media (max-width: 960px) {
  .lab-grid {
    grid-template-columns: 1fr;
  }
}
.error-banner {
  padding: 12px 14px;
  border-radius: var(--radius);
  background: rgba(240, 113, 120, 0.12);
  border: 1px solid rgba(240, 113, 120, 0.35);
  color: #ffc4c8;
}
.presets {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 14px;
}
@media (max-width: 520px) {
  .presets {
    grid-template-columns: 1fr;
  }
}
.preset-card {
  text-align: left;
  padding: 12px;
  border-radius: var(--radius);
  border: 1px solid var(--stroke);
  background: rgba(0, 0, 0, 0.2);
  display: flex;
  flex-direction: column;
  gap: 4px;
  transition: border-color 0.15s var(--ease), background 0.15s var(--ease);
}
.preset-card span {
  font-size: 0.78rem;
  color: var(--text-dim);
}
.preset-card:hover,
.preset-card.active {
  border-color: rgba(61, 214, 198, 0.45);
  background: var(--accent-soft);
}
.algo-toggle {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 16px;
}
.algo-toggle button {
  text-align: left;
  padding: 12px;
  border-radius: var(--radius);
  border: 1px solid var(--stroke);
  background: var(--bg-0);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.algo-toggle button span {
  font-size: 0.75rem;
  color: var(--text-dim);
}
.algo-toggle button.on {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px rgba(61, 214, 198, 0.35);
  background: var(--accent-soft);
}
.group h3 {
  margin: 16px 0 4px;
  font-size: 0.92rem;
}
.group-desc {
  margin: 0 0 10px;
  font-size: 0.8rem;
}
.list-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.exp-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.exp-card {
  padding: 12px;
  border-radius: var(--radius);
  border: 1px solid var(--stroke);
  background: rgba(0, 0, 0, 0.18);
}
.exp-top {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: flex-start;
}
.exp-name {
  font-weight: 700;
  color: var(--text);
  text-decoration: none;
}
.exp-name:hover {
  color: var(--accent);
}
.exp-meta {
  font-size: 0.78rem;
  margin-top: 2px;
}
.exp-mid {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 10px 0;
}
.spark {
  width: 120px;
  height: 28px;
  flex-shrink: 0;
  background: rgba(61, 214, 198, 0.04);
  border-radius: 6px;
}
.stats {
  display: flex;
  gap: 16px;
  font-size: 0.8rem;
}
.stats strong {
  display: block;
  font-size: 1rem;
}
</style>
