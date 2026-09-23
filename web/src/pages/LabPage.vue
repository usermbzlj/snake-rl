<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  api,
  type ConfigSchema,
  type ExperimentConfig,
  type ExperimentSummary,
} from '@/api'
import ExperimentCard from '@/components/ExperimentCard.vue'
import NewExperimentForm from '@/components/NewExperimentForm.vue'
import { deepClone, setByPath } from '@/utils/format'

const router = useRouter()
const schema = ref<ConfigSchema | null>(null)
const list = ref<ExperimentSummary[]>([])
const loading = ref(true)
const creating = ref(false)
const error = ref('')
const draft = ref<ExperimentConfig | null>(null)

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
    if (s.presets[0]) {
      draft.value = deepClone(s.presets[0].config)
    } else {
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

onMounted(() => void boot())
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
      <NewExperimentForm
        v-if="schema && draft"
        :schema="schema"
        :draft="draft"
        :creating="creating"
        @update:draft="draft = $event"
        @create="createExp"
      />

      <section class="panel list-panel" aria-labelledby="list-title">
        <div class="list-head">
          <h2 id="list-title" class="panel-title">我的实验</h2>
          <button type="button" class="btn btn-sm btn-ghost" @click="refreshList">刷新</button>
        </div>

        <div v-if="list.length === 0" class="empty">还没有实验。左边选个预设，点「开始训练」吧。</div>

        <ul v-else class="exp-list">
          <ExperimentCard
            v-for="exp in list"
            :key="exp.id"
            :exp="exp"
            @action="action(exp.id, $event)"
            @clone="cloneExp(exp)"
            @remove="removeExp(exp)"
          />
        </ul>
      </section>
    </div>
  </div>
</template>

<style scoped>
.lab-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.85fr);
  gap: var(--space-5);
  align-items: start;
}
@media (max-width: 1100px) {
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
</style>
