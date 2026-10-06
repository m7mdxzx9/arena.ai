import type { Any } from '../api'
import { PredictionCard } from '../labs/Prediction'
import { Card, Pill, go, useGame } from '../ui'

export default function PredictionLab({ focus }: { focus?: string }) {
  const { meta } = useGame()
  const { ov } = useGame()
  const done: Record<string, Any> = ov.predictions || {}
  const exps: Any[] = meta.predictions
  const cur = exps.find((e) => e.id === focus)
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">Research Institute</div><h1 style={{ margin: 0 }}>Prediction Lab</h1></div></div>
      <p className="muted">Commit to a prediction, then run the real experiment. Being surprised is the fastest way to update your intuition — wrong predictions still earn XP.</p>
      <div className="grid g-side">
        <Card title="Experiments" icon="🔮">
          <div className="col" style={{ gap: 4 }}>
            {exps.map((e) => (
              <button key={e.id} className={`option ${e.id === focus ? 'sel' : ''}`} onClick={() => go(`/predict/${e.id}`)}>
                <span style={{ flex: 1 }}>{e.title}</span>{done[e.id] && <Pill kind={done[e.id].was_right ? 'green' : 'amber'}>{done[e.id].was_right ? 'called it' : 'surprised'}</Pill>}
              </button>
            ))}
          </div>
        </Card>
        <div>{cur ? <Card key={cur.id} title={cur.title} icon="🧪"><PredictionCard exp={cur} /></Card> : <Card><div className="empty">Choose an experiment on the left.</div></Card>}</div>
      </div>
    </div>
  )
}
