import { useState } from 'react'
import { post, type Any } from '../api'
import Workbench from '../labs/Workbench'
import { Btn, Card, ErrorBox, Loading, Mentor, Pill, go, useApi, useGame } from '../ui'
import { DataProfile } from './DataLab'

const STEPS = ['Inspect the data', 'Hypothesis & prediction', 'Build & run', 'Reflect', 'Complete']

function Mcq({ prompt, options, onAnswer, result }: { prompt: string; options: string[]; onAnswer: (i: number) => void; result: Any }) {
  const [choice, setChoice] = useState<number | null>(null)
  return (
    <div className="col">
      <p><b>{prompt}</b></p>
      {options.map((o, i) => <button key={i} className={`option ${result ? (i === result.answer ? 'right' : i === choice ? 'wrong' : '') : choice === i ? 'sel' : ''}`} disabled={!!result} onClick={() => setChoice(i)}><span className="key">{i + 1}</span>{o}</button>)}
      {!result && <div><Btn disabled={choice === null} onClick={() => choice !== null && onAnswer(choice)}>Submit</Btn></div>}
      {result && <div className={result.correct ? 'ok-box' : 'warn-box'}>{result.correct ? '✅ ' : '❌ '}{result.explanation}</div>}
    </div>
  )
}

export default function MissionPage({ id }: { id: string }) {
  const { pid, meta, reward, refresh } = useGame()
  const { data: m, error, setData } = useApi(`/api/p/${pid}/missions/${id}`)
  const [dataRes, setDataRes] = useState<Any>(null)
  const [hyp, setHyp] = useState('')
  const [bucket, setBucket] = useState<number | null>(null)
  const [check, setCheck] = useState<Any>(null)
  const [refl, setRefl] = useState<Any>(null)
  const [err, setErr] = useState<string | null>(null)
  const [peek, setPeek] = useState(true)
  if (error) return <ErrorBox error={error} />
  if (!m) return <Loading />
  const st = m.state || {}
  const step = Math.min(st.step || 0, 4)
  const act = async (action: string, payload: Any) => {
    setErr(null)
    try {
      const r = await post(`/api/p/${pid}/missions/${id}`, { action, payload })
      setData({ ...m, state: r.state })
      if (r.xp || r.achievements?.length) reward(r)
      else refresh()
      return r
    } catch (e: Any) { setErr(e.message); return null }
  }
  const area = meta.areas[m.area]
  return (
    <div className="stack">
      <div className="topbar">
        <Btn kind="ghost" small onClick={() => go('/missions')}>← Missions</Btn>
        <div><div className="kicker">{area?.name} · mission</div><h1 style={{ margin: 0 }}>{m.title}</h1></div>
        <div className="spacer" /><Pill kind="violet">{m.xp} XP</Pill>
      </div>
      <div className="mission-steps">{STEPS.map((s, i) => <div key={s} className={`mstep ${i < step ? 'done' : i === step ? 'cur' : ''}`}><span>{i < step ? '✓' : i + 1}</span>{s}</div>)}</div>
      <Mentor name={m.mentor} icon={area?.icon} color={area?.color}>
        <p>{m.briefing}</p>
        <p className="muted"><small>Objectives: {m.objectives.join(' · ')}{m.bonus ? ` · (mode bonus: +${m.bonus} required)` : ''} · Concepts: {m.concepts.map((c: Any) => c.name).join(', ')}</small></p>
        {m.mentor_tip && <p><small>💡 {m.mentor_tip}</small></p>}
      </Mentor>
      <ErrorBox error={err} />

      {step === 0 && (
        <div className="grid g2">
          <Card title="Step 1 · Inspect the data before modelling" icon="🔍"><DataProfile ds={m.dataset} compact /></Card>
          <Card title="Data check" icon="❓"><Mcq prompt={m.data_question.prompt} options={m.data_question.options} result={dataRes} onAnswer={async (i) => { const r = await act('data_answer', { answer: i }); if (r) { setDataRes(r); setData({ ...m, state: { ...r.state, step: 0 } }) } }} />
            {dataRes && <div style={{ marginTop: 10 }}><Btn onClick={() => setData({ ...m, state: { ...st, step: 1 } })}>Next: form a hypothesis →</Btn></div>}
          </Card>
        </div>
      )}

      {step === 1 && (
        <Card title="Step 2 · Hypothesis & prediction" icon="🧠">
          <div className="col">
            <label className="field-l">Which features do you think matter, and which model will you try first? Why?</label>
            <textarea rows={3} value={hyp} onChange={(e) => setHyp(e.target.value)} placeholder="e.g. Study hours and attendance will matter most; I'll start with logistic regression because it's simple and interpretable." />
            <label className="field-l">Predict your first model's main test score ({m.task === 'regression' ? 'R²' : 'F1'}):</label>
            <div className="row">{m.metric_buckets.map((b: string, i: number) => <button key={i} className={`option ${bucket === i ? 'sel' : ''}`} style={{ flex: 1 }} onClick={() => setBucket(i)}>{b}</button>)}</div>
            <div><Btn disabled={bucket === null || hyp.trim().length < 5} onClick={() => act('hypothesis', { text: hyp, bucket })}>Lock in & open the Workbench →</Btn></div>
          </div>
        </Card>
      )}

      {step === 2 && (
        <div className="stack">
          {st.hypothesis && <div className="info-box"><b>Your hypothesis:</b> {st.hypothesis} · <b>predicted score:</b> {m.metric_buckets[st.prediction]}</div>}
          {m.banned?.length > 0 && <div className="warn-box">🚫 Not allowed in this mission: <span className="mono">{m.banned.join(', ')}</span></div>}
          {check && (
            <Card title={check.check.passed ? 'Objectives met!' : 'Not yet — diagnose and improve'} icon={check.check.passed ? '🏆' : '🔧'} className={check.check.passed ? 'glow' : ''}>
              {check.check.checks.map((c: Any, i: number) => <div key={i} className="row"><span>{c.passed ? '✅' : '❌'}</span><span>{c.label}</span>{c.value !== undefined && <span className="mono muted">({String(c.value)})</span>}{c.why && <small className="muted">— {c.why}</small>}</div>)}
              {check.predicted_bucket != null && <p>{check.prediction_right ? '🎯 Your score prediction was right!' : `🔮 You predicted ${m.metric_buckets[check.predicted_bucket]}; the real result was ${m.metric_buckets[check.actual_bucket]}.`}</p>}
              {check.check.passed && <Btn onClick={() => setData({ ...m, state: { ...st, step: 3 } })}>Continue to reflection →</Btn>}
            </Card>
          )}
          {peek && <div className="row"><small className="muted">Tip: every run is checked automatically against the objectives.</small><button className="link" onClick={() => setPeek(false)}>hide</button></div>}
          <Workbench dataset={m.dataset} lockDataset context={`mission:${id}`} recommended={m.recommended} banned={m.banned} onRun={async (r) => { if (r.ok) { const c = await act('submit_run', { run_id: r.run_id }); if (c) { setCheck(c); setData({ ...m, state: { ...c.state, step: 2 } }) } } }} />
        </div>
      )}

      {step === 3 && m.reflection && (
        <Card title="Step 4 · Reflect" icon="🪞">
          <Mcq prompt={m.reflection.prompt} options={m.reflection.options} result={refl} onAnswer={async (i) => { const r = await act('reflect', { answer: i }); if (r) { setRefl(r); setData({ ...m, state: { ...r.state, step: 3 } }) } }} />
          {refl && <div style={{ marginTop: 10 }}><Btn onClick={() => setData({ ...m, state: { ...st, step: 4 } })}>Finish mission →</Btn></div>}
        </Card>
      )}

      {step >= 4 && (
        <Card title="Mission complete" icon="🏆" className="glow">
          <p>You went through the whole ML loop on real data: inspected it, predicted, built, evaluated against objectives and reflected on the trade-offs.</p>
          <div className="row"><Btn onClick={() => go('/missions')}>More missions</Btn><Btn kind="ghost" onClick={() => go('/history')}>Compare your runs</Btn></div>
        </Card>
      )}
    </div>
  )
}
