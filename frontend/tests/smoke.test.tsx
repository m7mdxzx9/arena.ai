// @vitest-environment jsdom
/**
 * UI smoke test: renders every route of the real app against a running backend
 * (default http://127.0.0.1:8000, override with NF_API) and fails on any React
 * render error or console.error. Run: npm run test:ui (backend must be running).
 */
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import App from '../src/App'
import { Widget, WIDGETS } from '../src/widgets'
import { GameCtx } from '../src/ui'

const API = process.env.NF_API || 'http://127.0.0.1:8000'
const realFetch = globalThis.fetch
const errors: string[] = []
let pending = 0
let pid = 0

beforeAll(async () => {
  globalThis.fetch = (async (u: RequestInfo | URL, o?: RequestInit) => {
    pending++
    try { return await realFetch(new URL(String(u), API), o) } finally { pending-- }
  }) as typeof fetch
  window.scrollTo = () => {}
  const orig = console.error
  console.error = (...a: unknown[]) => { errors.push(a.map(String).join(' ').slice(0, 400)); orig(...a) }
  const p = await (await realFetch(`${API}/api/players`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: 'Smoke Tester', mode: 4 }) })).json()
  pid = p.player.id
  await realFetch(`${API}/api/p/${pid}/settings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ free_play: true }) })
  localStorage.setItem('neural-forge-player', String(pid))
})
afterAll(() => { globalThis.fetch = realFetch })

async function settle() {
  for (let i = 0; i < 400; i++) {
    await act(async () => { await new Promise((r) => setTimeout(r, 25)) })
    if (pending === 0) {
      await act(async () => { await new Promise((r) => setTimeout(r, 60)) })
      if (pending === 0) return
    }
  }
}

const ROUTES = [
  '/', '/tree', '/area/foundation_academy', '/area/data_district', '/area/ml_workshop', '/area/neural_tower', '/area/vision_lab', '/area/language_center',
  '/area/genai_facility', '/area/rag_archives', '/area/agent_arena', '/area/evaluation_chamber', '/area/research_institute',
  '/lesson/what_is_ai', '/lesson/overfitting', '/lesson/learning_rate', '/lesson/chunking', '/lesson/data_leakage',
  '/data/student_success', '/data/data_chaos', '/data/spam', '/data/customer_segments', '/workbench', '/workbench/house_prices', '/history', '/nn', '/predict', '/predict/lr_explosion',
  '/language', '/rag', '/agent', '/dojo', '/dojo/ex_variables', '/missions', '/mission/m_student', '/bosses',
  '/boss/overfitter', '/boss/leak', '/boss/imbalance', '/boss/chaos', '/boss/lr_beast', '/boss/hallucination', '/boss/retrieval', '/boss/injection',
  '/research', '/profile',
]

describe('every route renders without errors', () => {
  it('boots the app and visits all routes', async () => {
    window.location.hash = '#/'
    const { container } = render(<App />)
    await settle()
    expect(container.textContent).toContain('Smoke Tester')
    for (const r of ROUTES) {
      errors.length = 0
      await act(async () => { window.location.hash = '#' + r; window.dispatchEvent(new HashChangeEvent('hashchange')) })
      await settle()
      const text = container.textContent || ''
      if (process.env.NF_DUMP) console.log('ROUTE', r, '::', text.replace(/\s+/g, ' ').slice(0, 600))
      expect(text.length, `route ${r} rendered nothing`).toBeGreaterThan(50)
      expect(errors, `console errors on ${r}`).toEqual([])
      expect(text, `error box on ${r}`).not.toMatch(/⚠️ (Not Found|Internal|Unknown)/)
    }
    cleanup()
  }, 240_000)
})

describe('every concept visual renders', () => {
  it('renders all widgets', async () => {
    const ov = await (await realFetch(`${API}/api/p/${pid}`)).json()
    const meta = await (await realFetch(`${API}/api/meta`)).json()
    const game = { pid, ov, meta, refresh: async () => {}, toast: () => {}, reward: () => {}, setPid: () => {} }
    for (const name of Object.keys(WIDGETS)) {
      errors.length = 0
      const { container, unmount } = render(<GameCtx.Provider value={game}><Widget name={name} /></GameCtx.Provider>)
      await settle()
      await waitFor(() => expect(container.textContent!.length).toBeGreaterThan(5))
      expect(container.textContent, `widget ${name}`).not.toContain('Loading')
      expect(errors, `console errors in widget ${name}`).toEqual([])
      unmount()
    }
  }, 240_000)
})
