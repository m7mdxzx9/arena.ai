// Tiny typed-ish API client. All requests use relative URLs (/api/...) so the app works behind any proxy.
/* eslint-disable @typescript-eslint/no-explicit-any */
export type Any = any

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request(method: string, url: string, body?: unknown): Promise<Any> {
  const res = await fetch(url, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  const ct = res.headers.get('content-type') || ''
  const data = ct.includes('application/json') ? await res.json() : await res.text()
  if (!res.ok) {
    const msg = typeof data === 'object' ? data.error || JSON.stringify(data.detail || data) : String(data)
    throw new ApiError(msg, res.status)
  }
  return data
}

export const get = (url: string) => request('GET', url)
export const post = (url: string, body: unknown = {}) => request('POST', url, body)

export function qs(params: Record<string, unknown>): string {
  const p = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') p.set(k, String(v))
  })
  const s = p.toString()
  return s ? `?${s}` : ''
}

// player id is stored locally; all progress lives in the backend SQLite database
const KEY = 'neural-forge-player'
export const storedPlayer = (): number | null => {
  const v = localStorage.getItem(KEY)
  return v ? Number(v) : null
}
export const storePlayer = (id: number | null) => {
  if (id === null) localStorage.removeItem(KEY)
  else localStorage.setItem(KEY, String(id))
}
