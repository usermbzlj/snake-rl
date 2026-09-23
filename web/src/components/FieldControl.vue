<script setup lang="ts">
import { computed } from 'vue'
import type { FieldSchema } from '@/api'

const props = defineProps<{
  field: FieldSchema
  modelValue: unknown
}>()

const emit = defineEmits<{
  'update:modelValue': [value: unknown]
}>()

const id = computed(() => `field-${props.field.key.replace(/\./g, '-')}`)

const displayValue = computed(() => {
  const v = props.modelValue
  if (v == null || v === '') return '—'
  if (typeof v === 'number') {
    const s = props.field.step ?? 0.01
    const digits = s < 0.01 ? 5 : s < 0.1 ? 3 : s < 1 ? 2 : 0
    return Number(v).toFixed(digits).replace(/\.?0+$/, '') || String(v)
  }
  return String(v)
})

function onInput(ev: Event) {
  const el = ev.target as HTMLInputElement | HTMLSelectElement
  const t = props.field.type
  if (t === 'boolean') {
    emit('update:modelValue', (el as HTMLInputElement).checked)
    return
  }
  if (t === 'integer') {
    const n = parseInt(el.value, 10)
    emit('update:modelValue', Number.isFinite(n) ? n : props.field.default)
    return
  }
  if (t === 'number') {
    const n = parseFloat(el.value)
    emit('update:modelValue', Number.isFinite(n) ? n : props.field.default)
    return
  }
  if (t === 'select') {
    const raw = el.value
    const choice = props.field.choices?.find((c) => String(c.value) === raw)
    emit('update:modelValue', choice ? choice.value : raw)
    return
  }
  emit('update:modelValue', el.value)
}

const isSlider = computed(() => {
  const f = props.field
  return (
    (f.type === 'number' || f.type === 'integer') &&
    f.min != null &&
    f.max != null &&
    f.key !== 'run.seed' &&
    f.key !== 'run.max_env_steps'
  )
})
</script>

<template>
  <div class="field" :data-key="field.key">
    <div class="field-label">
      <label :for="id">
        {{ field.label }}
        <span v-if="field.live" class="live-tag" title="训练中可实时修改">实时</span>
      </label>
      <span class="field-value">{{ displayValue }}{{ field.unit ? ` ${field.unit}` : '' }}</span>
    </div>

    <select
      v-if="field.type === 'select'"
      :id="id"
      :value="String(modelValue ?? field.default)"
      :aria-describedby="`${id}-help`"
      @change="onInput"
    >
      <option v-for="c in field.choices ?? []" :key="String(c.value)" :value="String(c.value)">
        {{ c.label }}
      </option>
    </select>

    <input
      v-else-if="field.type === 'boolean'"
      :id="id"
      type="checkbox"
      :checked="Boolean(modelValue)"
      :aria-describedby="`${id}-help`"
      @change="onInput"
    />

    <template v-else-if="isSlider">
      <input
        :id="id"
        type="range"
        :min="field.min"
        :max="field.max"
        :step="field.step ?? (field.type === 'integer' ? 1 : 0.01)"
        :value="Number(modelValue ?? field.default)"
        :aria-describedby="`${id}-help`"
        :aria-valuetext="String(displayValue)"
        @input="onInput"
      />
    </template>

    <input
      v-else-if="field.type === 'integer' || field.type === 'number'"
      :id="id"
      type="number"
      :min="field.min"
      :max="field.max"
      :step="field.step ?? (field.type === 'integer' ? 1 : 'any')"
      :value="modelValue == null ? '' : Number(modelValue)"
      :aria-describedby="`${id}-help`"
      @change="onInput"
    />

    <input
      v-else
      :id="id"
      type="text"
      :value="String(modelValue ?? '')"
      :aria-describedby="`${id}-help`"
      @change="onInput"
    />

    <p :id="`${id}-help`" class="field-help">{{ field.help }}</p>
  </div>
</template>

<style scoped>
.live-tag {
  margin-left: 6px;
  font-size: 0.65rem;
  font-weight: 700;
  padding: 1px 6px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  vertical-align: middle;
}
</style>
