import { onUnmounted, ref, type Ref } from 'vue'
import {
  connectExperimentWs,
  type ExperimentDetail,
  type ExperimentEvent,
  type ExpWsMessage,
  type MetricRow,
  type WsStatus,
} from '@/api'
import { getByPath, setByPath } from '@/utils/format'

export function useExperimentStream(
  id: Ref<string>,
  detail: Ref<ExperimentDetail | null>,
  metrics: Ref<MetricRow[]>,
  events: Ref<ExperimentEvent[]>,
  liveDraft: Ref<Record<string, number>>,
  error: Ref<string>,
) {
  const wsStatus = ref<WsStatus>('closed')
  let expWs: ReturnType<typeof connectExperimentWs> | null = null

  function handleMessage(msg: ExpWsMessage) {
    if (msg.type === 'hello' && detail.value) {
      detail.value.experiment.status = msg.status
    } else if (msg.type === 'status' && detail.value) {
      detail.value.experiment.status = msg.status
      if (msg.error) error.value = msg.error
    } else if (msg.type === 'metrics') {
      metrics.value = [...metrics.value, msg.row]
    } else if (msg.type === 'event') {
      const dup = events.value.some((e) => e.t === msg.event.t && e.type === msg.event.type)
      if (!dup) events.value = [...events.value, msg.event]
      if (msg.event.type === 'live_patch' && detail.value) {
        const changes = (msg.event.data?.changes ?? {}) as Record<string, [unknown, unknown]>
        for (const [k, [, neu]] of Object.entries(changes)) {
          setByPath(detail.value.experiment.config as unknown as Record<string, unknown>, k, neu)
          if (typeof neu === 'number') liveDraft.value[k] = neu
        }
      }
    }
  }

  function connect() {
    expWs?.close()
    expWs = connectExperimentWs(
      id.value,
      handleMessage,
      (s) => {
        wsStatus.value = s
      },
    )
  }

  function disconnect() {
    expWs?.close()
    expWs = null
  }

  onUnmounted(disconnect)

  return { wsStatus, connect, disconnect }
}

/** Seed liveDraft from config + schema live fields. */
export function initLiveDraft(config: unknown, liveKeys: string[]): Record<string, number> {
  const draft: Record<string, number> = {}
  for (const key of liveKeys) {
    const v = getByPath(config, key)
    if (typeof v === 'number') draft[key] = v
  }
  return draft
}
