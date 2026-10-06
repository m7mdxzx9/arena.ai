import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { post, type Any } from '../api'
import { Btn, Card, CodeBlock, ErrorBox, Loading, Pill, Rich, go, highlight, useApi, useGame } from '../ui'
import { LoopTracer } from '../widgets/basic'

export function Editor({ value, onChange, rows = 14 }: { value: string; onChange: (v: string) => void; rows?: number }) {
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault()
      const t = e.currentTarget, s = t.selectionStart, end = t.selectionEnd
      const v = value.slice(0, s) + '    ' + value.slice(end)
      onChange(v)
      requestAnimationFrame(() => { t.selectionStart = t.selectionEnd = s + 4 })
    }
  }
  const lines = value.split('\n').length
  const pre = useRef<HTMLPreElement>(null)
  const gut = useRef<HTMLDivElement>(null)
  const n = Math.max(lines, rows)
  return (
    <div className="editor" style={{ height: `calc(${n * 1.55}em + 1.6rem)`, maxHeight: '70vh' }}>
      <pre ref={pre} aria-hidden="true" style={{ height: '100%' }}>{highlight(value)}{'\n'}</pre>
      <div ref={gut} className="gutter">{Array.from({ length: n }, (_, i) => <div key={i}>{i + 1}</div>)}</div>
      <textarea value={value} onChange={(e) => onChange(e.target.value)} onKeyDown={onKey} spellCheck={false} autoCapitalize="off" autoCorrect="off" aria-label="Python code editor"
        onScroll={(e) => { if (pre.current) { pre.current.scrollTop = e.currentTarget.scrollTop; pre.current.scrollLeft = e.currentTarget.scrollLeft } if (gut.current) gut.current.scrollTop = e.currentTarget.scrollTop }} />
    </div>
  )
}

function Exercise({ exId }: { exId: string }) {
  const { pid, reward } = useGame()
  const { data: ex, error, reload } = useApi(`/api/p/${pid}/exercises/${exId}`)
  const [code, setCode] = useState('')
  const [out, setOut] = useState<Any>(null)
  const [busy, setBusy] = useState(false)
  const [hints, setHints] = useState(0)
  const [trace, setTrace] = useState(false)
  useEffect(() => { if (ex) setCode(ex.state.code || ex.starter) }, [ex])
  if (error) return <ErrorBox error={error} />
  if (!ex) return <Loading />
  const runFree = async () => {
    setBusy(true)
    try { const r = await post('/api/code/run', { code }); setOut({ free: true, ...r }) } catch (e: Any) { setOut({ free: true, error: e.message }) } finally { setBusy(false) }
  }
  const check = async () => {
    setBusy(true)
    try { const r = await post(`/api/p/${pid}/exercises/${exId}`, { code }); setOut(r); reward(r); if (r.passed) reload() } catch (e: Any) { setOut({ error: e.message }) } finally { setBusy(false) }
  }
  return (
    <div className="grid g2">
      <div className="col">
        <Card title={ex.title} icon="📝" right={ex.state.completed ? <Pill kind="green">✓ solved</Pill> : <Pill>tier {ex.tier}</Pill>}>
          <Rich text={ex.prompt} />
          <div className="field-l" style={{ marginTop: 8 }}>Tests</div>
          <ul style={{ margin: 0, paddingLeft: '1.2rem' }}>{ex.tests.map((t: Any) => <li key={t.name}><small>{t.name}</small></li>)}</ul>
          {ex.hints.length > 0 && <div style={{ marginTop: 8 }}>
            {ex.hints.slice(0, hints).map((h: string, i: number) => <div key={i} className="info-box" style={{ marginBottom: 4 }}>💡 <Rich text={h} /></div>)}
            {hints < ex.hints.length && <button className="link" onClick={() => setHints(hints + 1)}>Show a hint ({hints}/{ex.hints.length})</button>}
          </div>}
        </Card>
        {ex.solution && <Card title={ex.state.completed ? 'Reference solution' : 'Solution (revealed after 5 attempts)'} icon="🔑"><CodeBlock code={ex.solution} /></Card>}
      </div>
      <div className="col">
        <Editor value={code} onChange={setCode} />
        <div className="row">
          <Btn onClick={check} disabled={busy}>✔ Run tests</Btn>
          <Btn kind="ghost" onClick={runFree} disabled={busy}>▶ Just run</Btn>
          <Btn kind="ghost" onClick={() => setTrace(!trace)}>{trace ? 'Hide' : '🔍 Step through'}</Btn>
          <button className="link" onClick={() => setCode(ex.starter)}>reset</button>
        </div>
        {busy && <Loading text="Running in the sandbox…" />}
        {out && (
          <div className="col">
            {!out.free && out.tests?.length > 0 && <div className={out.passed ? 'ok-box' : 'warn-box'}>
              {out.tests.map((t: Any) => <div key={t.name}>{t.passed ? '✅' : '❌'} {t.name}{t.error && <small className="mono"> — {t.error}</small>}</div>)}
              {out.passed && <b>All tests pass! {out.xp ? `+${out.xp} XP` : ''}</b>}
            </div>}
            {(out.stdout || out.error) && <pre className="console">{out.stdout}{out.error && <span style={{ color: 'var(--red)' }}>{out.error}</span>}</pre>}
          </div>
        )}
        {trace && <Card title="Line-by-line trace" icon="🔍"><LoopTracer key={code} initial={code} /></Card>}
      </div>
    </div>
  )
}

export default function Dojo({ exId }: { exId?: string }) {
  const { pid } = useGame()
  const { data } = useApi<Any[]>(exId ? null : `/api/p/${pid}/exercises`)
  return (
    <div className="stack">
      <div className="topbar">
        {exId && <Btn kind="ghost" small onClick={() => go('/dojo')}>← All exercises</Btn>}
        <div><div className="kicker">Foundation Academy · Code Terminal</div><h1 style={{ margin: 0 }}>Code Dojo</h1></div>
      </div>
      {exId ? <Exercise exId={exId} /> : (
        <>
          <p className="muted">Real Python, auto-graded by hidden tests in a locked-down sandbox. Tier 1: Python basics → tier 2: NumPy/pandas → tier 3: write ML pieces yourself (MSE, gradient descent, train/test split, accuracy, k-NN…).</p>
          {!data ? <Loading /> : [1, 2, 3].map((tier) => (
            <Card key={tier} title={['', 'Tier 1 · Python foundations', 'Tier 2 · Data with NumPy & pandas', 'Tier 3 · Build ML from scratch'][tier]} icon={['', '🐍', '🐼', '🛠️'][tier]}>
              <div className="grid g3">
                {data.filter((e) => e.tier === tier).map((e) => (
                  <button key={e.id} className={`option ${e.completed ? 'right' : ''}`} onClick={() => go(`/dojo/${e.id}`)}>
                    <span style={{ flex: 1, textAlign: 'left' }}><b>{e.title}</b><br /><small className="muted">{e.concept_name}{e.attempts ? ` · ${e.attempts} attempts` : ''}</small></span>{e.completed && '✓'}
                  </button>
                ))}
              </div>
            </Card>
          ))}
        </>
      )}
    </div>
  )
}
