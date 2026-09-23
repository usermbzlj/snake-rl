import type { ExpWsMessage, WatchWsClientConfig, WatchWsServerMessage } from './types'

export type WsStatus = 'connecting' | 'open' | 'closed' | 'error'

export interface ReconnectingWsOptions<TIn, TOut> {
  url: string
  onMessage: (msg: TIn) => void
  onStatus?: (status: WsStatus) => void
  encode?: (msg: TOut) => string
  decode?: (raw: string) => TIn
  baseDelayMs?: number
  maxDelayMs?: number
  createSocket?: (url: string) => WebSocket
}

export class ReconnectingWs<TIn, TOut = never> {
  private ws: WebSocket | null = null
  private closedByUser = false
  private attempt = 0
  private timer: ReturnType<typeof setTimeout> | null = null
  private readonly opts: Required<
    Pick<ReconnectingWsOptions<TIn, TOut>, 'encode' | 'decode' | 'baseDelayMs' | 'maxDelayMs'>
  > &
    ReconnectingWsOptions<TIn, TOut>

  constructor(opts: ReconnectingWsOptions<TIn, TOut>) {
    this.opts = {
      encode: (m) => JSON.stringify(m),
      decode: (raw) => JSON.parse(raw) as TIn,
      baseDelayMs: 600,
      maxDelayMs: 8000,
      ...opts,
    }
    this.connect()
  }

  private connect() {
    if (this.closedByUser) return
    this.opts.onStatus?.('connecting')
    const create = this.opts.createSocket ?? ((url: string) => new WebSocket(url))
    const ws = create(this.opts.url)
    this.ws = ws

    ws.onopen = () => {
      this.attempt = 0
      this.opts.onStatus?.('open')
    }
    ws.onmessage = (ev) => {
      try {
        const msg = this.opts.decode(String(ev.data))
        this.opts.onMessage(msg)
      } catch {
        /* ignore bad frames */
      }
    }
    ws.onerror = () => {
      this.opts.onStatus?.('error')
    }
    ws.onclose = () => {
      this.opts.onStatus?.('closed')
      this.ws = null
      if (!this.closedByUser) this.scheduleReconnect()
    }
  }

  private scheduleReconnect() {
    if (this.timer) clearTimeout(this.timer)
    const delay = Math.min(
      this.opts.maxDelayMs,
      this.opts.baseDelayMs * 2 ** Math.min(this.attempt, 5),
    )
    this.attempt += 1
    this.timer = setTimeout(() => this.connect(), delay)
  }

  send(msg: TOut) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(this.opts.encode(msg))
    }
  }

  close() {
    this.closedByUser = true
    if (this.timer) clearTimeout(this.timer)
    this.ws?.close()
    this.ws = null
  }
}

function wsUrl(path: string): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${location.host}${path}`
}

export function connectExperimentWs(
  id: string,
  onMessage: (msg: ExpWsMessage) => void,
  onStatus?: (s: WsStatus) => void,
) {
  return new ReconnectingWs<ExpWsMessage>({
    url: wsUrl(`/ws/experiments/${id}`),
    onMessage,
    onStatus,
  })
}

export function connectWatchWs(
  id: string,
  onMessage: (msg: WatchWsServerMessage) => void,
  onStatus?: (s: WsStatus) => void,
) {
  return new ReconnectingWs<WatchWsServerMessage, WatchWsClientConfig>({
    url: wsUrl(`/ws/experiments/${id}/watch`),
    onMessage,
    onStatus,
  })
}
