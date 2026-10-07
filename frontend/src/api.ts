// Typed API client. Relative URLs keep browser traffic on the preview/deployment origin.
/* eslint-disable @typescript-eslint/no-explicit-any */
export type Any = any

export type StructuredApplicationError = {
  error?: string
  message?: string
  detail?: string | { msg?: string }[]
  code?: string
}

export class ApiError extends Error {
  status: number
  code?: string
  constructor(message: string, status: number, code?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

/** Format unknown failures without throwing or exposing arbitrary object internals. */
export function formatApiError(error: unknown, fallback = 'Something went wrong. Please try again.'): string {
  try {
    if (error instanceof ApiError || error instanceof Error) {
      const message = typeof error.message === 'string' ? error.message.trim() : ''
      return message ? message.slice(0, 600) : fallback
    }
    if (typeof error === 'string') return error.trim().slice(0, 600) || fallback
    if (typeof error === 'number' || typeof error === 'boolean' || typeof error === 'bigint') return String(error)
    if (!error || typeof error !== 'object') return fallback
    const value = error as StructuredApplicationError
    if (typeof value.error === 'string' && value.error.trim()) return value.error.trim().slice(0, 600)
    if (typeof value.message === 'string' && value.message.trim()) return value.message.trim().slice(0, 600)
    if (typeof value.detail === 'string' && value.detail.trim()) return value.detail.trim().slice(0, 600)
    if (Array.isArray(value.detail)) {
      const messages = value.detail.flatMap((item) => typeof item?.msg === 'string' ? [item.msg] : []).join('; ')
      if (messages) return messages.slice(0, 600)
    }
    return fallback
  } catch {
    // Objects with throwing getters/proxies must not break the error UI.
    return fallback
  }
}

async function decodeResponse(res: Response): Promise<unknown> {
  const type = res.headers.get('content-type') || ''
  const text = await res.text()
  if (!type.includes('application/json')) return text
  if (!text) return null
  try { return JSON.parse(text) } catch {
    if (!res.ok) return { error: `The server returned an invalid response (HTTP ${res.status}).` }
    throw new ApiError('The server returned malformed JSON.', res.status, 'malformed_response')
  }
}

async function request(method: string, url: string, body?: unknown): Promise<Any> {
  let res: Response
  try {
    res = await fetch(url, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (error) {
    throw new ApiError(`Network request failed: ${formatApiError(error)}`, 0, 'network_error')
  }
  const data = await decodeResponse(res)
  if (!res.ok) {
    const structured = data && typeof data === 'object' ? data as StructuredApplicationError : null
    throw new ApiError(formatApiError(structured || data, `Request failed (HTTP ${res.status}).`), res.status, structured?.code)
  }
  return data
}

export const get = (url: string) => request('GET', url)
export const post = (url: string, body: unknown = {}) => request('POST', url, body)
export const patch = (url: string, body: unknown = {}) => request('PATCH', url, body)
export const put = (url: string, body: unknown = {}) => request('PUT', url, body)
export const del = (url: string) => request('DELETE', url)

export async function upload(url: string, file: File, fields: Record<string, string | number> = {}): Promise<Any> {
  const body = new FormData()
  body.set('file', file, file.name)
  Object.entries(fields).forEach(([key, value]) => body.set(key, String(value)))
  let response: Response
  try { response = await fetch(url, { method: 'POST', body }) } catch (error) {
    throw new ApiError(`Upload failed: ${formatApiError(error)}`, 0, 'network_error')
  }
  const data = await decodeResponse(response)
  if (!response.ok) {
    const structured = data && typeof data === 'object' ? data as StructuredApplicationError : null
    throw new ApiError(formatApiError(structured || data, `Upload failed (HTTP ${response.status}).`), response.status, structured?.code)
  }
  return data
}

export function qs(params: Record<string, unknown>): string {
  const p = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') p.set(k, String(v))
  })
  const s = p.toString()
  return s ? `?${s}` : ''
}

// Player id is stored locally; learning progress itself lives in SQLite.
const KEY = 'neural-forge-player'
export const storedPlayer = (): number | null => {
  const v = localStorage.getItem(KEY)
  return v ? Number(v) : null
}
export const storePlayer = (id: number | null) => {
  if (id === null) localStorage.removeItem(KEY)
  else localStorage.setItem(KEY, String(id))
}
