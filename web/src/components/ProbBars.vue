<script setup lang="ts">
defineProps<{
  probs: [number, number, number] | number[]
  chosen?: number | null
  compact?: boolean
}>()

const labels = ['直行', '左转', '右转']
</script>

<template>
  <div class="bars" :class="{ compact }" role="img" :aria-label="`动作概率 直行${(probs[0] * 100).toFixed(0)}% 左转${(probs[1] * 100).toFixed(0)}% 右转${(probs[2] * 100).toFixed(0)}%`">
    <div
      v-for="(p, i) in probs"
      :key="i"
      class="row"
      :class="{ chosen: chosen === i }"
    >
      <span class="lab">{{ labels[i] }}</span>
      <div class="track">
        <div class="fill" :style="{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }" />
      </div>
      <span class="pct mono">{{ (p * 100).toFixed(0) }}%</span>
    </div>
  </div>
</template>

<style scoped>
.bars {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 100%;
}
.row {
  display: grid;
  grid-template-columns: 36px 1fr 36px;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  color: var(--text-muted);
}
.row.chosen {
  color: var(--accent);
}
.row.chosen .fill {
  background: linear-gradient(90deg, var(--accent-2), var(--accent));
  box-shadow: 0 0 10px var(--accent-glow);
}
.lab {
  text-align: right;
}
.track {
  height: 7px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  overflow: hidden;
}
.fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #3a6d9e, #6db3f2);
  transition: width 0.2s var(--ease);
}
.pct {
  font-size: 0.7rem;
  text-align: right;
}
.compact .row {
  grid-template-columns: 28px 1fr 30px;
  font-size: 0.68rem;
}
.compact .track {
  height: 5px;
}
</style>
