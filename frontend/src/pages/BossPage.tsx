import { useState } from 'react'
import { post, type Any } from '../api'
import { AgentResult } from '../labs/AgentLab'
import AgentLab from '../labs/AgentLab'
import RagLab, { AnswersResult, RetrievalResult } from '../labs/RagLab'
import Workbench, { RunResult } from '../labs/Workbench'
import { Bar, Btn, Card, ErrorBox, Loading, Pill, Rich, go, useApi, useGame } from '../ui'
import { DataProfile } from './DataLab'
import { NNLab } from './NNLabPage'
import { LineChart, PALETTE } from '../charts'

function IntroView({ intro }: { intro: Any }) {
  const r = intro.run
  if (intro.kind === 'profile') return <DataProfile ds={intro.dataset} compact />
  if (intro.kind === 'ml') return <RunResult res={r} config={{}} showCode={false} />
  if (intro.kind === 'nn') return (
    <div className="grid g2">
      <div><div className="field-l">Loss</div><LineChart logY series={[{ name: 'train loss', values: r.history.train_loss }, { name: 'val loss', values: r.history.val_loss, color: PALETTE[1] }]} x={r.history.epoch} xLabel="epoch" /></div>
      <div><div className="field-l">Accuracy</div><LineChart yDomain={[0, 1]} series={[{ name: 'train acc', values: r.history.train_acc }, { name: 'val acc', values: r.history.val_acc, color: PALETTE[1] }]} x={r.history.epoch} xLabel="epoch" /></div>
    </div>
  )
  if (intro.kind === 'answers') return <AnswersResult res={r} />
  if (intro.kind === 'rag') return <RetrievalResult res={r} />
  if (intro.kind === 'agent') return <AgentResult res={r} />
  return null
}

export default function BossPage({ id }: { id: string }) {
  const { pid, reward, refresh } = useGame()
  const { data: b, error, setData } = useApi(`/api/p/${pid}/bosses/${id}`)
  const [choice, setChoice] = useState<number | null>(null)
  const [sel, setSel] = useState<number[]>([])
  const [ev, setEv] = useState<Any>(null)
  const [hit, setHit] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  if (error) return <ErrorBox error={error} />
  if (!b) return <Loading />
  const st = b.state
  const phase = b.phases[st.phase]
  const won = st.phase >= b.phases.length

  const send = async (payload: Any) => {
    setErr(null); setBusy(true)
    try {
      const r = await post(`/api/p/${pid}/bosses/${id}`, payload)
      setEv({ ...r.evaluation, phase: st.phase, damage: r.damage, lesson: r.lesson })
      if (r.evaluation?.passed) { setHit(true); setTimeout(() => setHit(false), 700) }
      setData({ ...b, state: { ...b.state, ...r.state, defeated: b.state.defeated || !!r.defeated } })
      setChoice(null); setSel([])
      if (r.xp || r.achievements?.length) reward(r); else refresh()
    } catch (e: Any) { setErr(e.message) } finally { setBusy(false) }
  }
  const restart = async () => { await post(`/api/p/${pid}/bosses/${id}`, { restart: true }); setEv(null); setData({ ...b, state: { ...b.state, phase: 0, hp: 100, mistakes: 0, log: [] } }) }
  const onRun = (r: Any) => { if (r.ok) send({ run_id: r.run_id }) }
  const intro = b.intro

  return (
    <div className="stack">
      <div className="topbar"><Btn kind="ghost" small onClick={() => go('/bosses')}>← Bosses</Btn><div className="spacer" />{st.defeated && <Pill kind="green">defeated before</Pill>}<Btn small kind="ghost" onClick={restart}>↺ Restart fight</Btn></div>
      <div className={`boss-hero ${hit ? 'hit' : ''} ${won ? 'won' : ''}`}>
        <div className="boss-icon big">{won ? '💥' : b.icon}</div>
        <div style={{ flex: 1 }}>
          <h1 style={{ margin: 0 }}>{b.name}</h1>
          <p style={{ fontStyle: 'italic', margin: '0.3rem 0 0.6rem' }}>“{b.taunt}”</p>
          <Bar value={st.hp} max={100} color="var(--red)" label={`HP ${st.hp}/100`} />
          <div className="row" style={{ marginTop: 8 }}>{b.phases.map((p: Any, i: number) => <Pill key={i} kind={i < st.phase ? 'green' : i === st.phase ? 'amber' : ''}>{i + 1}. {p.title}</Pill>)}<small className="muted">mistakes: {st.mistakes}</small></div>
        </div>
      </div>
      <ErrorBox error={err} />
      {ev && (
        <div className={ev.passed ? 'ok-box' : 'error-box'}>
          <b>{ev.passed ? `⚔️ Hit! ${ev.damage} damage.` : '🛡️ The boss shrugs it off.'}</b>
          {ev.checks?.length > 0 && <div style={{ marginTop: 4 }}>{ev.checks.map((c: Any, i: number) => <div key={i}>{c.passed ? '✅' : '❌'} {c.label}{c.value !== undefined ? <span className="mono muted"> ({Array.isArray(c.value) ? c.value.join(', ') : String(c.value)})</span> : ''}</div>)}</div>}
          {ev.explanation && <Rich text={ev.explanation} />}
        </div>
      )}
      {won ? (
        <Card title={`${b.name} DEFEATED`} icon="🏆" className="glow">
          <p><b>The lesson:</b> {b.lesson}</p>
          <div className="row"><Btn onClick={() => go('/bosses')}>Next boss</Btn><Btn kind="ghost" onClick={() => go('/')}>Campus</Btn></div>
        </Card>
      ) : (
        <>
          {intro && st.phase === 0 && <Card title={intro.title} icon="👁️"><IntroView intro={intro} /></Card>}
          <Card title={`Phase ${st.phase + 1}: ${phase.title}`} icon="⚔️">
            {phase.kind === 'mcq' && (
              <div className="col">
                <Rich text={phase.q.prompt} />
                {phase.q.options.map((o: string, i: number) => <button key={i} className={`option ${choice === i ? 'sel' : ''}`} onClick={() => setChoice(i)}><span className="key">{i + 1}</span>{o}</button>)}
                <div><Btn kind="danger" disabled={choice === null || busy} onClick={() => send({ answer: choice })}>⚔️ Strike</Btn></div>
              </div>
            )}
            {phase.kind === 'select' && (
              <div className="col">
                <p><b>{phase.prompt}</b></p>
                {phase.options.map((o: string, i: number) => <label key={i} className={`option ${sel.includes(i) ? 'sel' : ''}`}><input type="checkbox" checked={sel.includes(i)} onChange={() => setSel(sel.includes(i) ? sel.filter((x) => x !== i) : [...sel, i])} /> <span className="mono">{o}</span></label>)}
                <div><Btn kind="danger" disabled={sel.length === 0 || busy} onClick={() => send({ selected: sel })}>⚔️ Strike</Btn></div>
              </div>
            )}
            {!['mcq', 'select'].includes(phase.kind) && (
              <div className="col">
                <div className="info-box"><b>Brief:</b> {phase.brief}</div>
                {phase.criteria && <div className="row">{phase.criteria.map((c: string) => <Pill key={c} kind="amber">{c}</Pill>)}</div>}
                <small className="muted">Every run you make here is automatically submitted as an attack.</small>
              </div>
            )}
          </Card>
          {phase.kind === 'ml' && <Workbench key={st.phase} dataset={phase.dataset} lockDataset context={`boss:${id}`} initial={intro?.kind === 'ml' ? { dataset: phase.dataset, features: intro.run.features_used, model: id === 'imbalance' ? 'logistic_regression' : id === 'leak' ? 'random_forest' : 'decision_tree', params: {}, preprocessing: {} } : undefined} onRun={onRun} />}
          {phase.kind === 'nn' && <NNLab key={st.phase} embedded initial={{ ...(intro?.run ? { dataset: phase.dataset, hidden: intro.run.sizes.slice(1, -1), activation: 'relu', optimizer: 'sgd', lr: 8, batch_size: 32, epochs: 60, dropout: 0, l2: 0, seed: 0 } : {}), dataset: phase.dataset }} context={`boss:${id}`} onRun={onRun} />}
          {phase.kind === 'rag' && <RagLab key={st.phase} initial={intro?.run?.config} fixed={phase.fixed} context={`boss:${id}`} onRun={onRun} />}
          {phase.kind === 'answers' && <RagLab key={st.phase} answers context={`boss:${id}`} onRun={onRun} />}
          {phase.kind === 'agent' && <AgentLab key={st.phase} context={`boss:${id}`} onRun={onRun} />}
        </>
      )}
      {st.log.length > 0 && <Card title="Battle log" icon="📜">{st.log.slice().reverse().map((l: Any, i: number) => <div key={i}><small>{l.text}</small></div>)}</Card>}
    </div>
  )
}
