<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api } from '@/api'
import SnakeBoard from '@/components/SnakeBoard.vue'
import { SnakeEngine, type Dir } from '@/game/engine'
import { causeLabel } from '@/utils/format'

const STORAGE_BEST = 'snake-lab.play.best'

const size = ref(12)
const engine = ref(new SnakeEngine({ size: size.value }))
const running = ref(false)
const score = ref(0)
const best = ref(Number(localStorage.getItem(STORAGE_BEST) || '0'))
const lanUrl = ref('')
const tickMs = computed(() => Math.max(60, 220 - size.value * 4))

let timer: ReturnType<typeof setInterval> | null = null
let touchStart: { x: number; y: number } | null = null

const state = computed(() => engine.value.snapshot())

function persistBest() {
  if (score.value > best.value) {
    best.value = score.value
    localStorage.setItem(STORAGE_BEST, String(best.value))
  }
}

function stop() {
  running.value = false
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

function tick() {
  const s = engine.value.stepHuman()
  score.value = s.score
  if (s.done) {
    persistBest()
    stop()
  }
}

function startLoop() {
  stop()
  running.value = true
  timer = setInterval(tick, tickMs.value)
}

function newGame() {
  engine.value.reset(size.value)
  score.value = 0
  startLoop()
}

function pauseToggle() {
  if (!running.value) {
    if (engine.value.done) {
      newGame()
      return
    }
    startLoop()
  } else {
    stop()
  }
}

function queue(dir: Dir) {
  engine.value.queueAbsolute(dir)
  if (!running.value && !engine.value.done) startLoop()
}

function onKey(ev: KeyboardEvent) {
  const t = ev.target as HTMLElement | null
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return
  const map: Record<string, Dir> = {
    ArrowUp: 0,
    KeyW: 0,
    ArrowRight: 1,
    KeyD: 1,
    ArrowDown: 2,
    KeyS: 2,
    ArrowLeft: 3,
    KeyA: 3,
  }
  if (ev.code in map) {
    ev.preventDefault()
    queue(map[ev.code]!)
  } else if (ev.code === 'Space') {
    ev.preventDefault()
    pauseToggle()
  } else if (ev.code === 'KeyR' || ev.code === 'Enter') {
    ev.preventDefault()
    newGame()
  }
}

function onTouchStart(ev: TouchEvent) {
  const touch = ev.changedTouches[0]
  if (!touch) return
  touchStart = { x: touch.clientX, y: touch.clientY }
}

function onTouchEnd(ev: TouchEvent) {
  if (!touchStart) return
  const touch = ev.changedTouches[0]
  if (!touch) return
  const dx = touch.clientX - touchStart.x
  const dy = touch.clientY - touchStart.y
  touchStart = null
  if (Math.hypot(dx, dy) < 24) return
  if (Math.abs(dx) > Math.abs(dy)) queue(dx > 0 ? 1 : 3)
  else queue(dy > 0 ? 2 : 0)
}

watch(size, () => {
  newGame()
})

onMounted(async () => {
  window.addEventListener('keydown', onKey)
  try {
    const meta = await api.meta()
    lanUrl.value = meta.lan_urls[0] ?? `http://本机:${meta.port}`
  } catch {
    lanUrl.value = ''
  }
  engine.value.reset(size.value)
  score.value = 0
  // Start paused so phones/desktop users see the board before it runs into a wall
  running.value = false
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  stop()
})
</script>

<template>
  <div class="page play-page">
    <header class="page-header">
      <div>
        <h1>自己玩</h1>
        <p>规则与 AI 训练环境一致。手机可打开下方局域网地址一起玩。</p>
      </div>
    </header>

    <div class="play-layout">
      <section
        class="panel board-panel"
        @touchstart.passive="onTouchStart"
        @touchend.passive="onTouchEnd"
      >
        <SnakeBoard
          :size="state.size"
          :snake="state.body"
          :food="state.food"
          :dir="state.dir"
          :dead="state.done"
          :max-css-size="480"
        />

        <div v-if="state.done" class="overlay-msg" role="status">
          <strong>{{ causeLabel(state.cause) }}</strong>
          <span>得分 {{ score }} · 按 R 或点「新局」再来</span>
        </div>
      </section>

      <aside class="panel side">
        <div class="hud">
          <div><span class="dim">得分</span><strong>{{ score }}</strong></div>
          <div><span class="dim">最佳</span><strong>{{ best }}</strong></div>
          <div><span class="dim">长度</span><strong>{{ state.body.length }}</strong></div>
        </div>

        <label class="size-field">
          棋盘大小
          <select v-model.number="size" aria-label="棋盘大小">
            <option v-for="n in [6, 8, 10, 12, 16, 20]" :key="n" :value="n">{{ n }}×{{ n }}</option>
          </select>
        </label>

        <div class="btn-row">
          <button type="button" class="btn btn-primary" @click="newGame">新局</button>
          <button type="button" class="btn" @click="pauseToggle">
            {{ running ? '暂停' : '继续' }}
          </button>
        </div>

        <div class="dpad" aria-label="方向键">
          <button type="button" class="btn" aria-label="上" @click="queue(0)">↑</button>
          <div class="mid">
            <button type="button" class="btn" aria-label="左" @click="queue(3)">←</button>
            <button type="button" class="btn" aria-label="下" @click="queue(2)">↓</button>
            <button type="button" class="btn" aria-label="右" @click="queue(1)">→</button>
          </div>
        </div>

        <p class="help dim">
          键盘：方向键 / WASD 转向 · 空格暂停 · R 重开<br />
          触屏：滑动转向
        </p>

        <div v-if="lanUrl" class="lan">
          <span class="dim">手机访问</span>
          <code>{{ lanUrl }}/play</code>
        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.play-layout {
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 16px;
  align-items: start;
}
@media (max-width: 800px) {
  .play-layout {
    grid-template-columns: 1fr;
  }
}
.board-panel {
  position: relative;
  touch-action: none;
}
.overlay-msg {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: center;
  padding: 12px;
  border-radius: var(--radius);
  background: rgba(240, 113, 120, 0.1);
  border: 1px solid rgba(240, 113, 120, 0.3);
}
.hud {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-bottom: 14px;
}
.hud span {
  display: block;
  font-size: 0.72rem;
}
.hud strong {
  font-size: 1.25rem;
}
.size-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 12px;
  font-size: 0.85rem;
}
.size-field select {
  min-height: 36px;
  padding: 6px 10px;
  border-radius: 8px;
  border: 1px solid var(--stroke);
  background: var(--bg-0);
}
.dpad {
  margin: 16px 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.dpad .mid {
  display: flex;
  gap: 8px;
}
.dpad .btn {
  width: 56px;
  height: 48px;
  font-size: 1.2rem;
}
.help {
  font-size: 0.8rem;
  line-height: 1.5;
}
.lan {
  margin-top: 14px;
  padding: 10px;
  border-radius: var(--radius);
  background: var(--accent-soft);
  border: 1px solid rgba(61, 214, 198, 0.25);
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.8rem;
}
.lan code {
  font-family: var(--font-mono);
  color: var(--accent);
  word-break: break-all;
}
</style>
