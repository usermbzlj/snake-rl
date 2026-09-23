/** Metric → chart series helpers and event markers. */

import type { ChartMarker } from '@/components/UChart.vue'
import type { ConfigSchema, ExperimentEvent, MetricRow } from '@/api'
import { downsampleEven, formatLivePatchLabel } from '@/utils/format'

export function metricXs(metrics: MetricRow[]): number[] {
  return metrics.map((m) => m.env_steps ?? 0)
}

export function metricSeries(metrics: MetricRow[], key: keyof MetricRow): (number | undefined)[] {
  return metrics.map((m) => {
    const v = m[key]
    return typeof v === 'number' ? v : undefined
  })
}

export function livePatchMarkers(
  events: ExperimentEvent[],
  schema?: ConfigSchema | null,
): ChartMarker[] {
  return events
    .filter((e) => e.type === 'live_patch')
    .map((e) => {
      const changes = (e.data?.changes ?? {}) as Record<string, [unknown, unknown]>
      return {
        x: e.env_steps,
        label: formatLivePatchLabel(changes, schema),
      }
    })
}

export function bestEvalScore(metrics: MetricRow[]): number | undefined {
  let best: number | undefined
  for (const m of metrics) {
    const v = m.eval_score_mean
    if (typeof v === 'number' && (best === undefined || v > best)) best = v
  }
  return best
}

export function sparkPoints(spark: number[], width = 120, height = 28): string {
  if (!spark.length) return ''
  const min = Math.min(...spark)
  const max = Math.max(...spark)
  const span = max - min || 1
  return spark
    .map((v, i) => {
      const x = (i / Math.max(1, spark.length - 1)) * width
      const y = height - ((v - min) / span) * (height - 4) - 2
      return `${x},${y}`
    })
    .join(' ')
}

export { downsampleEven }
