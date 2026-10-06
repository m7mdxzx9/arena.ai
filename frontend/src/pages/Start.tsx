import { useEffect, useState } from 'react'
import { get, post, type Any } from '../api'
import { Btn, Card, ErrorBox } from '../ui'

const MODE_ICONS: Record<number, string> = { 1: '🌱', 2: '🧭', 3: '🏋️', 4: '🛠️', 5: '🔭' }

export default function Start({ meta, onPlayer }: { meta: Any; onPlayer: (id: number) => void }) {
  const [players, setPlayers] = useState<Any[]>([])
  const [name, setName] = useState('')
  const [mode, setMode] = useState(1)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    get('/api/players').then(setPlayers).catch(() => {})
  }, [])
  const modes: Any = meta.modes
  const create = async () => {
    setBusy(true)
    setErr(null)
    try {
      const ov = await post('/api/players', { name: name || 'Apprentice', mode })
      onPlayer(ov.player.id)
    } catch (e: Any) {
      setErr(e.message)
    } finally {
      setBusy(false)
    }
  }
  const real = players
  return (
    <div className="main" style={{ margin: '0 auto' }}>
      <div className="hero">
        <div className="kicker">An AI education game with real machine learning inside</div>
        <h1><span>NEURAL FORGE</span></h1>
        <p style={{ fontSize: '1.1rem', maxWidth: 760 }}>
          Arrive at the campus as an <b>AI Beginner</b>. Learn by building: inspect real datasets, form hypotheses, predict outcomes,
          train <b>real models</b>, read honest metrics, and defeat the eight classic failures of AI — from <i>The Overfitter</i> to <i>Prompt Injection</i>.
          Leave as an <b>AI Research Scientist</b>.
        </p>
        <div className="row" style={{ gap: '1.2rem', marginTop: '0.8rem', flexWrap: 'wrap' }}>
          <span className="pill cyan">{Object.keys(meta.concepts).length} concepts</span>
          <span className="pill green">{meta.datasets.filter((d: Any) => !d.toy).length} real-data missions & labs</span>
          <span className="pill red">8 boss battles</span>
          <span className="pill amber">Real scikit-learn & NumPy computation</span>
          <span className="pill violet">Python Code Dojo</span>
        </div>
      </div>
      <div className="grid g2" style={{ marginTop: '1.2rem' }}>
        <Card title="New researcher" icon="🧑‍🔬">
          <label className="field">
            <span className="field-l">Your name</span>
            <input value={name} maxLength={40} placeholder="Apprentice" onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && create()} />
          </label>
          <div className="field-l" style={{ margin: '0.9rem 0 0.4rem' }}>Difficulty mode (change any time)</div>
          <div className="col">
            {modes && Object.entries(modes).map(([k, m]: [string, Any]) => (
              <div key={k} className={`mode-card ${mode === Number(k) ? 'on' : ''}`} onClick={() => setMode(Number(k))} role="radio" aria-checked={mode === Number(k)} tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && setMode(Number(k))}>
                <b>{MODE_ICONS[Number(k)]} {m.name}</b>
                <div className="muted"><small>{m.desc}</small></div>
              </div>
            ))}
          </div>
          <ErrorBox error={err} />
          <div style={{ marginTop: '1rem' }}><Btn onClick={create} disabled={busy}>Enter the campus →</Btn></div>
        </Card>
        <div className="col">
          <Card title="Continue" icon="💾">
            {real.length === 0 && <p className="muted">No saved researchers yet.</p>}
            {real.map((p) => (
              <div key={p.id} className="row between" style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--line)' }}>
                <span><b>{p.name}</b> <small className="muted">· {p.xp} XP</small></span>
                <Btn small kind="ghost" onClick={() => onPlayer(p.id)}>Play</Btn>
              </div>
            ))}
          </Card>
          <Card title="How you learn here" icon="🧪">
            <ol style={{ margin: 0, paddingLeft: '1.2rem', lineHeight: 1.8 }}>
              <li><b>Teach</b> → a short idea with an analogy, never a wall of text.</li>
              <li><b>Visualise</b> → an interactive widget computed from real data.</li>
              <li><b>Predict</b> → commit to a guess <i>before</i> the experiment runs.</li>
              <li><b>Experiment</b> → real models, real metrics, honest results.</li>
              <li><b>Explain & reflect</b> → why it happened, in your own words.</li>
              <li><b>Spaced review</b> → concepts come back just before you forget them.</li>
            </ol>
            <p className="muted" style={{ marginTop: '0.6rem' }}><small>The game estimates your mastery of every concept with Bayesian Knowledge Tracing and adapts difficulty, explanations and reviews — see docs/ADAPTIVE_LEARNING.md.</small></p>
          </Card>
        </div>
      </div>
    </div>
  )
}
