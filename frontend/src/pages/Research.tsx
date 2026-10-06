import { useState } from 'react'
import { post, type Any } from '../api'
import Workbench from '../labs/Workbench'
import { Btn, Card, ErrorBox, Loading, Pill, fmt, go, useApi, useGame } from '../ui'

const CH_DATA: Record<string, string> = { rc_fraud_cost: 'fraud', rc_house_budget: 'house_prices', rc_stable_student: 'student_success' }

export default function Research() {
  const { pid, meta, reward, ov } = useGame()
  const { data, reload } = useApi<Any[]>(`/api/p/${pid}/challenges`)
  const [active, setActive] = useState<string | null>(null)
  const [score, setScore] = useState<Any>(null)
  const [err, setErr] = useState<string | null>(null)
  const ch = data?.find((c) => c.id === active)
  const submit = async (r: Any) => {
    if (!r.ok || !active) return
    setErr(null)
    try { const s = await post(`/api/p/${pid}/challenges/${active}`, { run_id: r.run_id }); setScore(s); reward(s); reload() } catch (e: Any) { setErr(e.message) }
  }
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">Research Institute</div><h1 style={{ margin: 0 }}>Research Challenges</h1></div><div className="spacer" /><Btn kind="ghost" onClick={() => go('/predict')}>🔮 Prediction Lab</Btn></div>
      <p className="muted">Open-ended problems with no single right answer. Your score combines the metric with engineering quality (cost, latency, interpretability, stability). Iterate, keep notes in your experiment history, and beat your personal best.{ov.player.mode < 5 && ' Switch to Research mode in your profile for the full no-hints experience.'}</p>
      {!data ? <Loading /> : (
        <div className="grid g3">
          {data.map((c) => {
            const locked = c.requires.some((r: string) => !ov.player.settings.free_play && (meta.concepts[r] && false))
            return (
              <Card key={c.id} title={c.title} icon="🔬" className={active === c.id ? 'glow' : ''}>
                <p style={{ fontSize: '0.9rem' }}>{c.brief}</p>
                <div className="row between">
                  <span>{c.best != null ? <Pill kind="green">best {fmt(c.best, 1)}</Pill> : <Pill>no entry yet</Pill>} <small className="muted">{c.entries.length} attempts</small></span>
                  <Btn small disabled={locked} onClick={() => { setActive(c.id); setScore(null) }}>{active === c.id ? 'Active' : 'Attempt'}</Btn>
                </div>
                {c.entries.length > 1 && <div className="spark">{c.entries.map((e: Any, i: number) => <span key={i} title={`run #${e.run_id}: ${e.score}`} style={{ height: `${Math.max(4, Math.min(100, e.score))}%` }} />)}</div>}
              </Card>
            )
          })}
        </div>
      )}
      <ErrorBox error={err} />
      {score && (
        <div className={score.valid === false ? 'warn-box' : score.improved ? 'ok-box' : 'info-box'}>
          {score.valid === false ? <>⚠️ Not a valid entry: {score.reason}</> : <>
            <b>Score: {fmt(score.score, 1)}</b> {score.improved ? `🎉 new personal best! +${score.xp} XP` : `(best: ${fmt(score.best, 1)})`}
            <div className="mono" style={{ fontSize: '0.8rem', marginTop: 4 }}>{Object.entries(score.details || {}).map(([k, v]) => `${k}=${fmt(v, 3)}`).join(' · ')}</div>
            <div><small>{score.explanation}</small></div>
          </>}
        </div>
      )}
      {ch && <Workbench key={ch.id} dataset={CH_DATA[ch.id]} lockDataset context={`challenge:${ch.id}`} onRun={submit} />}
    </div>
  )
}
