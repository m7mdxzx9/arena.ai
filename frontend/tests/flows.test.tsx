// @vitest-environment jsdom
/**
 * End-to-end gameplay flows through the real UI against a running backend:
 * a full lesson, a Workbench run, a full mission, and a full boss fight.
 */
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import App from '../src/App'

const API = process.env.NF_API || 'http://127.0.0.1:8000'
const realFetch = globalThis.fetch
const errors: string[] = []
let pending = 0
let pid = 0
const J = { 'Content-Type': 'application/json' }

beforeAll(async () => {
  globalThis.fetch = (async (u: RequestInfo | URL, o?: RequestInit) => {
    pending++
    try { return await realFetch(new URL(String(u), API), o) } finally { pending-- }
  }) as typeof fetch
  window.scrollTo = () => {}
  const orig = console.error
  console.error = (...a: unknown[]) => { errors.push(a.map(String).join(' ').slice(0, 300)); orig(...a) }
  const p = await (await realFetch(`${API}/api/players`, { method: 'POST', headers: J, body: JSON.stringify({ name: 'Flow Tester', mode: 2 }) })).json()
  pid = p.player.id
  await realFetch(`${API}/api/p/${pid}/settings`, { method: 'POST', headers: J, body: JSON.stringify({ free_play: true }) })
  localStorage.setItem('neural-forge-player', String(pid))
  window.location.hash = '#/'
  render(<App />)
  await settle()
})
afterAll(() => { globalThis.fetch = realFetch })

async function settle() {
  for (let i = 0; i < 1200; i++) {
    await act(async () => { await new Promise((r) => setTimeout(r, 25)) })
    if (pending === 0) {
      await act(async () => { await new Promise((r) => setTimeout(r, 60)) })
      if (pending === 0) return
    }
  }
}
async function nav(path: string) {
  await act(async () => { window.location.hash = '#' + path; window.dispatchEvent(new HashChangeEvent('hashchange')) })
  await settle()
}
async function click(el: Element) { await act(async () => { fireEvent.click(el) }); await settle() }
const btn = (re: RegExp) => screen.getAllByRole('button').find((b) => re.test(b.textContent || '') && !(b as HTMLButtonElement).disabled)
const option = (i: number) => document.querySelectorAll('button.option')[i] as HTMLElement
const body = () => document.body.textContent || ''

describe('gameplay flows', () => {
  it('completes a full lesson with adaptive questions', async () => {
    await nav('/lesson/what_is_ai')
    for (let guard = 0; guard < 30 && !body().includes('Lesson complete'); guard++) {
      const next = btn(/Got it|Continue|I'm ready|Skip/)
      if (btn(/Check answer/) || document.querySelector('button.option:not(:disabled)')) {
        const opt = option(0) || null
        if (opt) await click(opt)
        else { const inp = document.querySelector('input[inputmode=decimal]') as HTMLInputElement; await act(async () => { fireEvent.change(inp, { target: { value: '1' } }) }) }
        await click(btn(/Check answer/)!)
        expect(body()).toMatch(/Correct!|Not quite/)
        expect(body()).toContain('Mastery estimate')
        continue
      }
      if (next) { await click(next); continue }
      throw new Error('stuck in lesson: ' + body().slice(-400))
    }
    expect(body()).toContain('Lesson complete')
    expect(errors).toEqual([])
  }, 120_000)

  it('trains a real model in the Workbench and stores it in history', async () => {
    await nav('/workbench/student_success')
    await click(btn(/Train & evaluate/)!)
    expect(body()).toMatch(/Run #\d+/)
    expect(body()).toContain('Confusion matrix')
    await nav('/history')
    expect(body()).toMatch(/runs/)
    expect(errors).toEqual([])
  }, 120_000)

  it('plays a full mission: data check → hypothesis → run → reflect', async () => {
    await nav('/mission/m_student')
    await click(option(4))
    await click(btn(/^Submit$/)!)
    await click(btn(/form a hypothesis/)!)
    const ta = document.querySelector('textarea') as HTMLTextAreaElement
    await act(async () => { fireEvent.change(ta, { target: { value: 'Study hours and attendance matter; logistic regression first.' } }) })
    await click(option(2))
    await click(btn(/Lock in/)!)
    const rec = btn(/Use recommended setup/)
    if (rec) await click(rec) // only shown in Beginner mode
    await click(btn(/Train & evaluate/)!)
    expect(body()).toContain('Objectives met!')
    await click(btn(/Continue to reflection/)!)
    await click(option(1))
    await click(btn(/^Submit$/)!)
    await click(btn(/Finish mission/)!)
    expect(body()).toContain('Mission complete')
    expect(errors).toEqual([])
  }, 180_000)

  it('defeats THE OVERFITTER through the UI', async () => {
    await nav('/boss/overfitter')
    await click(option(1))
    await click(btn(/Strike/)!)
    expect(body()).toContain('Hit!')
    // phase 2: keep only the two real signals + one noise column, use logistic regression
    await click(screen.getAllByText('none').find((e) => e.tagName === 'BUTTON')!)
    for (const f of ['signal_a', 'signal_b', 'noise_00']) await click(screen.getAllByText(f).find((e) => e.classList.contains('chip'))!)
    const sel = screen.getAllByRole('combobox').find((s) => Array.from((s as HTMLSelectElement).options).some((o) => o.value === 'logistic_regression'))!
    await act(async () => { fireEvent.change(sel, { target: { value: 'logistic_regression' } }) })
    await click(btn(/Train & evaluate/)!)
    expect(body()).toMatch(/Phase 3/)
    await click(option(3))
    await click(btn(/Strike/)!)
    expect(body()).toContain('DEFEATED')
    expect(errors).toEqual([])
  }, 180_000)
})
