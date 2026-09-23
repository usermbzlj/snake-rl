<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api, type ExperimentSummary, type MetaResponse } from '@/api'

const meta = ref<MetaResponse | null>(null)
const experiments = ref<ExperimentSummary[]>([])
const loadError = ref('')

const runningCount = computed(
  () => experiments.value.filter((e) => e.status === 'running' || e.status === 'paused').length,
)

async function refresh() {
  try {
    const [m, list] = await Promise.all([api.meta(), api.listExperiments()])
    meta.value = m
    experiments.value = list
    loadError.value = ''
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : '无法连接后端'
  }
}

let timer: ReturnType<typeof setInterval> | null = null
onMounted(() => {
  void refresh()
  timer = setInterval(() => void refresh(), 8000)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})

watch(
  () => location.pathname,
  () => void refresh(),
)
</script>

<template>
  <div class="shell">
    <header class="topnav" role="banner">
      <RouterLink class="brand" to="/" aria-label="回到实验室">
        <span class="brand-mark" aria-hidden="true" />
        <span class="brand-text">
          <strong>贪吃蛇 AI</strong>
          <small>训练实验室</small>
        </span>
      </RouterLink>

      <nav class="nav-links" aria-label="主导航">
        <RouterLink to="/">实验室</RouterLink>
        <RouterLink to="/compare">对比</RouterLink>
        <RouterLink to="/play">自己玩</RouterLink>
      </nav>

      <div class="nav-right">
        <span
          v-if="runningCount > 0"
          class="run-pill"
          :title="`${runningCount} 个实验进行中`"
        >
          <span class="pulse" aria-hidden="true" />
          {{ runningCount }} 训练中
        </span>
        <span v-if="meta" class="device" :title="meta.device.name">
          {{ meta.device.cuda ? 'GPU' : 'CPU' }}
          · {{ meta.device.name }}
        </span>
        <span v-else-if="loadError" class="device warn" :title="loadError">离线</span>
      </div>
    </header>
    <main class="shell-main">
      <slot />
    </main>
  </div>
</template>

<style scoped>
.shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.topnav {
  position: sticky;
  top: 0;
  z-index: 40;
  height: var(--nav-h);
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 18px;
  background: rgba(7, 11, 20, 0.82);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--stroke);
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--text);
  text-decoration: none;
  min-width: 0;
}
.brand-mark {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background:
    radial-gradient(circle at 30% 30%, #9ffff0, transparent 45%),
    linear-gradient(135deg, var(--accent), #1a6b8a);
  box-shadow: 0 0 18px var(--accent-glow);
  flex-shrink: 0;
}
.brand-text {
  display: flex;
  flex-direction: column;
  line-height: 1.15;
}
.brand-text strong {
  font-size: 0.95rem;
  letter-spacing: -0.02em;
}
.brand-text small {
  font-size: 0.7rem;
  color: var(--text-dim);
}
.nav-links {
  display: flex;
  gap: 4px;
  margin-left: 8px;
}
.nav-links a {
  padding: 8px 12px;
  border-radius: 8px;
  color: var(--text-muted);
  text-decoration: none;
  font-weight: 600;
  font-size: 0.9rem;
  transition: background 0.15s var(--ease), color 0.15s var(--ease);
}
.nav-links a:hover {
  color: var(--text);
  background: rgba(255, 255, 255, 0.04);
}
.nav-links a.router-link-active {
  color: var(--accent);
  background: var(--accent-soft);
}
.nav-right {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.device {
  font-size: 0.75rem;
  color: var(--text-dim);
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.device.warn {
  color: var(--warn);
}
.run-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(93, 222, 160, 0.12);
  color: var(--ok);
  font-size: 0.75rem;
  font-weight: 650;
  border: 1px solid rgba(93, 222, 160, 0.28);
}
.pulse {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--ok);
  box-shadow: 0 0 0 0 rgba(93, 222, 160, 0.6);
  animation: pulse 1.6s ease-out infinite;
}
@keyframes pulse {
  70% {
    box-shadow: 0 0 0 8px rgba(93, 222, 160, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(93, 222, 160, 0);
  }
}
.shell-main {
  flex: 1;
}
@media (max-width: 640px) {
  .brand-text small {
    display: none;
  }
  .device {
    display: none;
  }
  .nav-links a {
    padding: 8px 8px;
    font-size: 0.85rem;
  }
}
</style>
