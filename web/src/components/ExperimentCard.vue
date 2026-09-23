<script setup lang="ts">
import StatusPill from '@/components/StatusPill.vue'
import type { ExperimentSummary } from '@/api'
import { formatNumber, formatSteps } from '@/utils/format'
import { sparkPoints } from '@/utils/metrics'

defineProps<{
  exp: ExperimentSummary
}>()

const emit = defineEmits<{
  action: [act: 'start' | 'pause' | 'resume' | 'stop']
  clone: []
  remove: []
}>()
</script>

<template>
  <li class="exp-card">
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
        @click="emit('action', 'resume')"
      >
        继续
      </button>
      <button
        v-else-if="exp.status === 'stopped' || exp.status === 'created' || exp.status === 'finished'"
        type="button"
        class="btn btn-sm"
        @click="emit('action', 'start')"
      >
        继续
      </button>
      <button
        v-if="exp.status === 'running'"
        type="button"
        class="btn btn-sm"
        @click="emit('action', 'pause')"
      >
        暂停
      </button>
      <button
        v-if="exp.status === 'running' || exp.status === 'paused'"
        type="button"
        class="btn btn-sm"
        @click="emit('action', 'stop')"
      >
        停止
      </button>
      <button type="button" class="btn btn-sm btn-ghost" @click="emit('clone')">克隆</button>
      <button type="button" class="btn btn-sm btn-danger" @click="emit('remove')">删除</button>
    </div>
  </li>
</template>

<style scoped>
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
