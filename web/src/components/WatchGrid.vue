<script setup lang="ts">
import { computed } from 'vue'
import ProbBars from '@/components/ProbBars.vue'
import SnakeBoard from '@/components/SnakeBoard.vue'
import type { WatchFrame, WatchGame } from '@/api'
import type { WatchConfig } from '@/composables/useWatch'
import { formatNumber } from '@/utils/format'

const props = defineProps<{
  watchCfg: WatchConfig
  frame: WatchFrame | null
  watchInfo: string
}>()

const emit = defineEmits<{
  'update:watchCfg': [partial: Partial<WatchConfig>]
}>()

const boardMax = computed(() => {
  if (props.watchCfg.games === 1) return 720
  if (props.watchCfg.games === 4) return 420
  return 280
})

const games = computed((): WatchGame[] => props.frame?.games ?? [])

const gridClass = computed(() => {
  if (props.watchCfg.games === 1) return 'g1'
  if (props.watchCfg.games === 4) return 'g4'
  return 'g9'
})

function setGames(v: number) {
  const games = (v === 1 || v === 9 ? v : 4) as 1 | 4 | 9
  emit('update:watchCfg', { games })
}
</script>

<template>
  <section class="panel watch-panel">
    <div class="watch-toolbar">
      <label>
        画面
        <select
          :value="watchCfg.games"
          aria-label="同时观看局数"
          @change="setGames(Number(($event.target as HTMLSelectElement).value))"
        >
          <option :value="1">1 局</option>
          <option :value="4">4 局</option>
          <option :value="9">9 局</option>
        </select>
      </label>
      <label>
        盘面
        <input
          :value="watchCfg.board_size"
          type="number"
          min="5"
          max="32"
          aria-label="观看棋盘大小"
          @input="
            emit('update:watchCfg', {
              board_size: Number(($event.target as HTMLInputElement).value),
            })
          "
        />
      </label>
      <label>
        速度
        <input
          :value="watchCfg.speed"
          type="range"
          min="1"
          max="60"
          aria-label="播放速度"
          @input="
            emit('update:watchCfg', {
              speed: Number(($event.target as HTMLInputElement).value),
            })
          "
        />
        <span class="mono">{{ watchCfg.speed }}</span>
      </label>
      <label class="chk">
        <input
          :checked="watchCfg.greedy"
          type="checkbox"
          @change="
            emit('update:watchCfg', {
              greedy: ($event.target as HTMLInputElement).checked,
            })
          "
        />
        贪心（少随机）
      </label>
    </div>

    <div v-if="games.length === 0" class="waiting" role="status">
      <div class="waiting-visual" aria-hidden="true" />
      <strong>{{ watchInfo || '模型还在热身…' }}</strong>
      <p class="dim">
        {{
          watchInfo
            ? '权重一就绪就会自动开播，先喝口水。'
            : '正在等第一份权重；通常开训几十秒后这里就会动起来。'
        }}
      </p>
    </div>

    <div v-else class="game-grid" :class="gridClass">
      <div v-for="(g, i) in games" :key="i" class="game-cell">
        <SnakeBoard
          fit-container
          :size="frame?.board_size ?? watchCfg.board_size"
          :snake="g.snake"
          :food="g.food"
          :dir="g.dir"
          :dead="g.dead"
          :max-css-size="boardMax"
        />
        <div class="game-foot">
          <span class="score">得分 {{ g.score }}</span>
          <span class="dim">V={{ formatNumber(g.value, 2) }}</span>
        </div>
        <ProbBars :probs="g.probs" :chosen="g.action" compact />
      </div>
    </div>
  </section>
</template>

<style scoped>
.waiting {
  display: grid;
  place-items: center;
  gap: 8px;
  text-align: center;
  min-height: 280px;
  padding: 28px 16px;
  border-radius: var(--radius);
  border: 1px dashed var(--stroke-strong);
  background: rgba(0, 0, 0, 0.18);
}
.waiting-visual {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  background:
    radial-gradient(circle at 30% 30%, #9ffff0, transparent 50%),
    linear-gradient(135deg, var(--accent-soft), rgba(109, 179, 242, 0.15));
  border: 1px solid rgba(61, 214, 198, 0.35);
  animation: waitPulse 1.8s ease-in-out infinite;
}
@keyframes waitPulse {
  50% {
    transform: scale(1.06);
    opacity: 0.85;
  }
}
.waiting p {
  margin: 0;
  max-width: 36ch;
  font-size: 0.85rem;
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
.game-grid {
  display: grid;
  gap: 14px;
}
.game-grid.g1 {
  grid-template-columns: minmax(0, 720px);
  justify-content: center;
}
.game-grid.g4 {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.game-grid.g9 {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}
@media (max-width: 700px) {
  .game-grid.g9 {
    grid-template-columns: repeat(2, minmax(0, 1fr));
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
</style>
