/** Pure utilities — downsample & formatting */

export function downsampleEven<T>(items: T[], max: number): T[] {
  if (items.length <= max || max <= 0) return items.slice()
  if (max === 1) return [items[items.length - 1]!]
  const out: T[] = []
  const last = items.length - 1
  for (let i = 0; i < max; i++) {
    const idx = Math.round((i * last) / (max - 1))
    out.push(items[idx]!)
  }
  return out
}

export function formatNumber(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return '—'
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (Math.abs(n) >= 1e4) return `${(n / 1e3).toFixed(1)}k`
  if (Number.isInteger(n)) return String(n)
  return n.toFixed(digits)
}

export function formatSteps(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '—'
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(Math.round(n))
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  const s = Math.floor(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  if (h > 0) return `${h}小时${m}分`
  if (m > 0) return `${m}分${r}秒`
  return `${r}秒`
}

export function causeLabel(cause: number): string {
  switch (cause) {
    case 1:
      return '撞墙'
    case 2:
      return '撞自己'
    case 3:
      return '饿死'
    case 4:
      return '通关'
    default:
      return '进行中'
  }
}

export function setByPath(obj: Record<string, unknown>, path: string, value: unknown): void {
  const parts = path.split('.')
  let cur: Record<string, unknown> = obj
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i]!
    const next = cur[p]
    if (typeof next !== 'object' || next === null) {
      cur[p] = {}
    }
    cur = cur[p] as Record<string, unknown>
  }
  cur[parts[parts.length - 1]!] = value
}

export function getByPath(obj: unknown, path: string): unknown {
  const parts = path.split('.')
  let cur: unknown = obj
  for (const p of parts) {
    if (cur == null || typeof cur !== 'object') return undefined
    cur = (cur as Record<string, unknown>)[p]
  }
  return cur
}

export function deepClone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T
}
