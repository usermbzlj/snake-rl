<script setup lang="ts">
import { computed, ref } from 'vue'
import FieldControl from '@/components/FieldControl.vue'
import type { Algo, ConfigSchema, ExperimentConfig, FieldSchema, Preset } from '@/api'
import { deepClone, getByPath, setByPath } from '@/utils/format'

const props = defineProps<{
  schema: ConfigSchema
  draft: ExperimentConfig
  creating: boolean
}>()

const emit = defineEmits<{
  'update:draft': [cfg: ExperimentConfig]
  create: []
}>()

const showAdvanced = ref(false)
const selectedPreset = ref<string | null>(null)

const algo = computed({
  get: () => props.draft.algo,
  set: (v: Algo) => {
    const next = deepClone(props.draft)
    next.algo = v
    emit('update:draft', next)
  },
})

function visibleFields(fields: FieldSchema[]): FieldSchema[] {
  return fields.filter((f) => {
    if (f.key === 'algo') return false
    if (f.algo && f.algo !== algo.value) return false
    if (f.advanced && !showAdvanced.value) return false
    return true
  })
}

const essentialGroups = computed(() =>
  props.schema.groups
    .map((g) => ({
      ...g,
      fields: visibleFields(g.fields).filter((f) => !f.advanced),
    }))
    .filter((g) => g.fields.length > 0),
)

const advancedGroups = computed(() => {
  if (!showAdvanced.value) return []
  return props.schema.groups
    .map((g) => ({
      ...g,
      fields: visibleFields(g.fields).filter((f) => f.advanced),
    }))
    .filter((g) => g.fields.length > 0)
})

function fieldValue(key: string): unknown {
  return getByPath(props.draft, key)
}

function setField(key: string, value: unknown) {
  const next = deepClone(props.draft)
  if (key === 'algo') {
    next.algo = value as Algo
  } else {
    setByPath(next as unknown as Record<string, unknown>, key, value)
  }
  emit('update:draft', next)
}

function applyPreset(p: Preset) {
  selectedPreset.value = p.id
  emit('update:draft', deepClone(p.config))
}

// Select first preset visually if draft matches (parent seeds draft)
if (props.schema.presets[0]) {
  selectedPreset.value = props.schema.presets[0].id
}
</script>

<template>
  <section class="panel create-panel" aria-labelledby="new-exp-title">
    <h2 id="new-exp-title" class="panel-title">新建实验</h2>

    <div class="presets" role="group" aria-label="预设方案">
      <button
        v-for="p in schema.presets"
        :key="p.id"
        type="button"
        class="preset-card"
        :class="{ active: selectedPreset === p.id }"
        :aria-pressed="selectedPreset === p.id"
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

    <div v-for="g in essentialGroups" :key="g.key" class="group card-group">
      <h3>{{ g.label }}</h3>
      <p class="dim group-desc">{{ g.description }}</p>
      <div class="fields-grid">
        <FieldControl
          v-for="f in g.fields"
          :key="f.key"
          :field="f"
          :model-value="fieldValue(f.key)"
          @update:model-value="setField(f.key, $event)"
        />
      </div>
    </div>

    <button type="button" class="btn btn-ghost btn-sm" @click="showAdvanced = !showAdvanced">
      {{ showAdvanced ? '收起进阶参数' : '展开进阶参数' }}
    </button>

    <div v-for="g in advancedGroups" :key="'adv-' + g.key" class="group card-group">
      <h3>{{ g.label }} · 进阶</h3>
      <div class="fields-grid">
        <FieldControl
          v-for="f in g.fields"
          :key="f.key"
          :field="f"
          :model-value="fieldValue(f.key)"
          @update:model-value="setField(f.key, $event)"
        />
      </div>
    </div>

    <div class="btn-row" style="margin-top: 12px">
      <button type="button" class="btn btn-primary" :disabled="creating" @click="emit('create')">
        {{ creating ? '启动中…' : '开始训练' }}
      </button>
    </div>
  </section>
</template>

<style scoped>
.card-group {
  margin-top: 14px;
  padding: 14px;
  border-radius: var(--radius);
  border: 1px solid var(--stroke);
  background: rgba(0, 0, 0, 0.16);
}
.card-group h3 {
  margin: 0 0 4px;
}
.fields-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px 16px;
}
.fields-grid :deep(.field) {
  margin-bottom: 8px;
}
@media (max-width: 640px) {
  .fields-grid {
    grid-template-columns: 1fr;
  }
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
  transition:
    border-color 0.15s var(--ease),
    background 0.15s var(--ease);
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
  margin: 0 0 4px;
  font-size: 0.92rem;
}
.group-desc {
  margin: 0 0 10px;
  font-size: 0.8rem;
}
</style>
