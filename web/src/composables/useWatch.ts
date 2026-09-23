import { onUnmounted, reactive, ref, watch, type Ref } from 'vue'
import { connectWatchWs, type WatchFrame } from '@/api'

export type WatchConfig = {
  games: 1 | 4 | 9
  board_size: number
  speed: number
  greedy: boolean
}

export function useWatch(id: Ref<string>, initialBoardSize = 8) {
  const frame = ref<WatchFrame | null>(null)
  const watchInfo = ref('')
  const watchCfg = reactive<WatchConfig>({
    games: 4,
    board_size: initialBoardSize,
    speed: 10,
    greedy: true,
  })

  let watchWs: ReturnType<typeof connectWatchWs> | null = null

  function sendWatchConfig() {
    watchWs?.send({
      type: 'config',
      games: watchCfg.games,
      board_size: watchCfg.board_size,
      speed: watchCfg.speed,
      greedy: watchCfg.greedy,
    })
  }

  function connect() {
    watchWs?.close()
    watchWs = connectWatchWs(id.value, (msg) => {
      if (msg.type === 'frame') {
        frame.value = msg
        watchInfo.value = ''
      } else if (msg.type === 'info') {
        watchInfo.value = msg.message
      }
    })
    sendWatchConfig()
  }

  function disconnect() {
    watchWs?.close()
    watchWs = null
  }

  watch(watchCfg, () => sendWatchConfig(), { deep: true })
  onUnmounted(disconnect)

  return { frame, watchInfo, watchCfg, connect, disconnect, sendWatchConfig }
}
