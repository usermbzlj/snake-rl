# Parallel Rollout Benchmark (Smoke)

Environment:
- device: `cpu`
- model: `adaptive_cnn`
- episodes: `16`
- max_steps_per_episode: `40`
- board_size: `10`
- logging: tensorboard/csv/jsonl disabled

Measured with `final_global_step / elapsed_seconds`:

| mode | workers | elapsed_s | global_steps | steps_per_s |
|---|---:|---:|---:|---:|
| serial | 0 | 2.003 | 274 | 136.78 |
| parallel | 1 | 9.501 | 246 | 25.89 |
| parallel | 2 | 10.168 | 322 | 31.67 |
| parallel | 4 | 10.873 | 261 | 24.00 |

Notes:
- This is a smoke benchmark on short runs; Windows `spawn` startup and IPC overhead dominate.
- For stable throughput comparison, run longer benchmarks (larger `episodes`, fewer process restarts).
