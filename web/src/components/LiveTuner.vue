<script setup lang="ts">
import { computed, ref } from 'vue'
import FieldControl from '@/components/FieldControl.vue'
import type { Algo, ConfigSchema, FieldSchema } from '@/api'

const props = defineProps<{
  schema: ConfigSchema | null
  algo: Algo
  liveDraft: Record<string, number>
  status: string
  patching: boolean
  note?: string
}>()

const emit = defineEmits<{
  'update:liveDraft': [key: string, value: number]
  apply: []
}>()

const showLiveAdvanced = ref(false)

const liveFields = computed((): FieldSchema[] => {
  if (!props.schema) return []
  return props.schema.groups
    .flatMap((g) => g.fields)
    .filter((f) => f.live && (!f.algo || f.algo === props.algo) && !f.advanced)
})

const liveAdvancedFields = computed((): FieldSchema[] => {
  if (!props.schema || !showLiveAdvanced.value) return []
  return props.schema.groups
    .flatMap((g) => g.fields)
    .filter((f) => f.live && (!f.algo || f.algo === props.algo) && f.advanced)
})
</script>

<template>
  <section class="panel">
    <h2 class="panel-title">实时调参</h2>
    <p class="dim tip">只改「实时」字段。应用后曲线上会出现竖线标记。</p>
    <FieldControl
      v-for="f in liveFields"
      :key="f.key"
      :field="f"
      :model-value="liveDraft[f.key]"
      @update:model-value="emit('update:liveDraft', f.key, $event as number)"
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
      @update:model-value="emit('update:liveDraft', f.key, $event as number)"
    />
    <button
      type="button"
      class="btn btn-primary"
      :disabled="patching"
      @click="emit('apply')"
    >
      {{ patching ? '应用中…' : '应用调参' }}
    </button>
    <p v-if="note" class="apply-note" role="status">{{ note }}</p>
    <p v-else-if="status !== 'running' && status !== 'paused'" class="dim tip">
      当前没在训练。应用后会写入这个实验，下次开始时生效。
    </p>
  </section>
</template>

<style scoped>
.tip {
  font-size: 0.8rem;
  margin: -4px 0 12px;
}
.apply-note {
  margin: 8px 0 0;
  font-size: 0.85rem;
  color: var(--accent);
}
</style>
