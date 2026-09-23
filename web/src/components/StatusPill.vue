<script setup lang="ts">
import { computed } from 'vue'
import type { ExpStatus } from '@/api'

const props = defineProps<{
  status: ExpStatus
}>()

const label = computed(() => {
  const map: Record<ExpStatus, string> = {
    created: '已创建',
    running: '训练中',
    paused: '已暂停',
    stopped: '已停止',
    finished: '已完成',
    error: '出错',
  }
  return map[props.status] ?? props.status
})

const tone = computed(() => {
  switch (props.status) {
    case 'running':
      return 'ok'
    case 'paused':
      return 'warn'
    case 'error':
      return 'danger'
    case 'finished':
      return 'info'
    default:
      return 'muted'
  }
})
</script>

<template>
  <span class="pill" :class="tone" :aria-label="`状态：${label}`">
    <span class="dot" aria-hidden="true" />
    {{ label }}
  </span>
</template>

<style scoped>
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 650;
  border: 1px solid transparent;
}
.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}
.ok {
  color: var(--ok);
  background: rgba(93, 222, 160, 0.12);
  border-color: rgba(93, 222, 160, 0.28);
}
.warn {
  color: var(--warn);
  background: rgba(240, 195, 90, 0.12);
  border-color: rgba(240, 195, 90, 0.28);
}
.danger {
  color: var(--danger);
  background: rgba(240, 113, 120, 0.12);
  border-color: rgba(240, 113, 120, 0.28);
}
.info {
  color: var(--info);
  background: rgba(109, 179, 242, 0.12);
  border-color: rgba(109, 179, 242, 0.28);
}
.muted {
  color: var(--text-muted);
  background: rgba(148, 178, 220, 0.08);
  border-color: var(--stroke);
}
</style>
