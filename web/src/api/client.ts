import type {
  CheckpointInfo,
  CompareResponse,
  ConfigSchema,
  ExperimentConfig,
  ExperimentDetail,
  ExperimentSummary,
  MetaResponse,
  Trajectory,
} from './types'
import { ApiError } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  if (!res.ok) {
    let detail = `请求失败 (${res.status})`
    try {
      const body = (await res.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  meta: () => request<MetaResponse>('/api/meta'),
  configSchema: () => request<ConfigSchema>('/api/config-schema'),
  listExperiments: () => request<ExperimentSummary[]>('/api/experiments'),
  createExperiment: (config: ExperimentConfig, start = true) =>
    request<ExperimentSummary>('/api/experiments', {
      method: 'POST',
      body: JSON.stringify({ config, start }),
    }),
  getExperiment: (id: string) => request<ExperimentDetail>(`/api/experiments/${id}`),
  experimentAction: (id: string, action: 'start' | 'pause' | 'resume' | 'stop') =>
    request<ExperimentSummary>(`/api/experiments/${id}/${action}`, { method: 'POST' }),
  livePatch: (id: string, patch: Record<string, number>) =>
    request<{ config: ExperimentConfig; event: unknown }>(`/api/experiments/${id}/live`, {
      method: 'PATCH',
      body: JSON.stringify({ patch }),
    }),
  cloneExperiment: (id: string, name: string, with_weights: boolean) =>
    request<ExperimentSummary>(`/api/experiments/${id}/clone`, {
      method: 'POST',
      body: JSON.stringify({ name, with_weights }),
    }),
  deleteExperiment: (id: string) =>
    request<void>(`/api/experiments/${id}`, { method: 'DELETE' }),
  checkpoints: (id: string) =>
    request<CheckpointInfo[]>(`/api/experiments/${id}/checkpoints`),
  inspect: (
    id: string,
    body: {
      checkpoint: 'latest' | 'best'
      board_size: number
      seed?: number
      greedy?: boolean
    },
  ) =>
    request<Trajectory>(`/api/experiments/${id}/inspect`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  compare: (body: {
    entries: { experiment_id: string; checkpoint: 'latest' | 'best' }[]
    board_size: number
    seed?: number
  }) =>
    request<CompareResponse>('/api/compare', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}
