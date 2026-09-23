/** Shared API types mirroring the SPEC HTTP/WS contract. */

export type Algo = 'ppo' | 'dqn'
export type ExpStatus = 'created' | 'running' | 'paused' | 'stopped' | 'finished' | 'error'
export type Cause = 0 | 1 | 2 | 3 | 4

export interface MetaResponse {
  version: string
  device: { cuda: boolean; name: string }
  lan_urls: string[]
  port: number
}

export interface FieldSchema {
  key: string
  label: string
  help: string
  type: 'number' | 'integer' | 'boolean' | 'string' | 'select'
  min?: number
  max?: number
  step?: number
  default: unknown
  advanced: boolean
  live: boolean
  algo?: Algo | null
  choices?: { value: string | number | boolean; label: string }[]
  unit?: string
}

export interface SchemaGroup {
  key: string
  label: string
  description: string
  fields: FieldSchema[]
}

export interface Preset {
  id: string
  name: string
  description: string
  config: ExperimentConfig
}

export interface ConfigSchema {
  groups: SchemaGroup[]
  presets: Preset[]
}

export interface EnvConfig {
  min_size: number
  max_size: number
  hunger_factor: number
}

export interface RewardConfig {
  food: number
  death: number
  step: number
  approach: number
  starve: number
  win: number
}

export interface ModelConfig {
  width: number
  resize_obs: boolean
}

export interface PPOConfig {
  num_envs: number
  rollout: number
  epochs: number
  minibatches: number
  lr: number
  lr_end: number
  lr_anneal_steps: number
  gamma: number
  gae_lambda: number
  clip: number
  ent_coef: number
  ent_coef_end: number
  ent_anneal_steps: number
  vf_coef: number
  max_grad_norm: number
  normalize_returns: boolean
}

export interface DQNConfig {
  num_envs: number
  replay_size: number
  n_step: number
  batch_size: number
  lr: number
  gamma: number
  tau: number
  epsilon_start: number
  epsilon_end: number
  epsilon_decay_steps: number
  learning_starts: number
  train_freq: number
  steps_per_iteration: number
}

export interface RunConfig {
  max_env_steps: number
  seed: number | null
  device: 'auto' | 'cuda' | 'cpu'
  eval_every_s: number
  compile: boolean
}

export interface ExperimentConfig {
  name: string
  algo: Algo
  env: EnvConfig
  reward: RewardConfig
  model: ModelConfig
  ppo: PPOConfig
  dqn: DQNConfig
  run: RunConfig
}

export interface MetricRow {
  iter?: number
  env_steps?: number
  time_s?: number
  sps?: number
  episodes?: number
  score_mean?: number
  score_max?: number
  length_mean?: number
  return_mean?: number
  ep_steps_mean?: number
  death_wall?: number
  death_self?: number
  death_starve?: number
  win_rate?: number
  lr?: number
  entropy?: number
  loss_policy?: number
  loss_value?: number
  kl?: number
  clipfrac?: number
  loss_q?: number
  q_mean?: number
  epsilon?: number
  eval_score_mean?: number
  eval_score_max?: number
  eval_win_rate?: number
  [key: string]: number | undefined
}

export interface ExperimentEvent {
  t: number
  env_steps: number
  type: 'start' | 'pause' | 'resume' | 'stop' | 'live_patch' | 'finish' | 'error' | 'best'
  data?: Record<string, unknown>
}

export interface ExperimentSummary {
  id: string
  name: string
  algo: Algo
  status: ExpStatus
  created_at: string
  board: [number, number]
  env_steps: number
  elapsed_s: number
  best_eval_score: number | null
  last: { score_mean: number; score_max: number; sps: number } | null
  spark: number[]
}

export interface ExperimentDetail {
  experiment: {
    id: string
    name: string
    algo: Algo
    created_at: string
    status: ExpStatus
    config: ExperimentConfig
    parent_id?: string | null
    notes?: string | null
    error?: string | null
  }
  metrics: MetricRow[]
  events: ExperimentEvent[]
}

export interface CheckpointInfo {
  name: 'latest' | 'best'
  env_steps: number
  saved_at: string
}

export type Cell = [number, number]

export interface WatchGame {
  snake: Cell[]
  food: Cell
  dir: number
  score: number
  steps: number
  probs: [number, number, number]
  value: number
  action: number
  dead: boolean
  cause: Cause
}

export interface WatchFrame {
  type: 'frame'
  model_version: number
  env_steps: number
  games: WatchGame[]
  board_size: number
}

export interface TrajectoryStep {
  snake: Cell[]
  food: Cell
  dir: number
  action: number
  probs: [number, number, number]
  value: number
  q?: [number, number, number]
  reward_components: [number, number, number, number, number, number]
  score: number
}

export interface Trajectory {
  board_size: number
  seed: number
  steps: TrajectoryStep[]
  result: { score: number; cause: Cause; steps: number }
  saliency: number[][] | null
  experiment_id?: string
  name?: string
}

export interface CompareResponse {
  seed: number
  trajectories: Trajectory[]
}

export type ExpWsMessage =
  | { type: 'hello'; status: ExpStatus }
  | { type: 'metrics'; row: MetricRow }
  | { type: 'event'; event: ExperimentEvent }
  | { type: 'status'; status: ExpStatus; error?: string }

export type WatchWsServerMessage =
  | WatchFrame
  | { type: 'info'; message: string }

export interface WatchWsClientConfig {
  type: 'config'
  games: 1 | 4 | 9
  board_size: number
  speed: number
  greedy: boolean
}

export class ApiError extends Error {
  status: number
  detail: string
  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.detail = detail
  }
}
