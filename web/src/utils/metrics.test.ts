import { describe, expect, it } from 'vitest'
import {
  bestEvalScore,
  downsampleEven,
  livePatchMarkers,
  metricSeries,
  metricXs,
  sparkPoints,
} from '@/utils/metrics'
import type { ExperimentEvent, MetricRow } from '@/api'

describe('metricXs / metricSeries', () => {
  const rows: MetricRow[] = [
    { env_steps: 100, score_mean: 1.5 },
    { env_steps: 200, score_mean: 2.0 },
  ]

  it('extracts x axis', () => {
    expect(metricXs(rows)).toEqual([100, 200])
  })

  it('extracts series values', () => {
    expect(metricSeries(rows, 'score_mean')).toEqual([1.5, 2.0])
  })
})

describe('livePatchMarkers', () => {
  it('builds markers from live_patch events', () => {
    const events: ExperimentEvent[] = [
      {
        t: 1,
        env_steps: 500,
        type: 'live_patch',
        data: { changes: { 'reward.food': [1, 2] } },
      },
      { t: 2, env_steps: 600, type: 'pause', data: {} },
    ]
    const markers = livePatchMarkers(events)
    expect(markers).toHaveLength(1)
    expect(markers[0]!.x).toBe(500)
    expect(markers[0]!.label).toContain('→')
  })
})

describe('bestEvalScore', () => {
  it('tracks max eval', () => {
    const rows: MetricRow[] = [
      { env_steps: 1, eval_score_mean: 1 },
      { env_steps: 2, eval_score_mean: 3 },
      { env_steps: 3, score_mean: 9 },
    ]
    expect(bestEvalScore(rows)).toBe(3)
  })
})

describe('sparkPoints', () => {
  it('returns empty for empty spark', () => {
    expect(sparkPoints([])).toBe('')
  })

  it('returns polyline points', () => {
    const pts = sparkPoints([0, 1, 0.5])
    expect(pts.split(' ')).toHaveLength(3)
  })
})

describe('downsampleEven', () => {
  it('keeps all when under limit', () => {
    expect(downsampleEven([1, 2, 3], 10)).toEqual([1, 2, 3])
  })

  it('downsamples evenly keeping ends', () => {
    const src = Array.from({ length: 100 }, (_, i) => i)
    const out = downsampleEven(src, 5)
    expect(out).toHaveLength(5)
    expect(out[0]).toBe(0)
    expect(out[4]).toBe(99)
  })
})
