import type { Any } from '../api'
import { Card, ErrorBox, Loading, Pill, go, useApi, useGame } from '../ui'

export default function Missions() {
  const { pid, meta } = useGame()
  const { data, error } = useApi<Any[]>(`/api/p/${pid}/missions`)
  const ds = (id: string) => meta.datasets.find((d: Any) => d.id === id)
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">Campus jobs</div><h1 style={{ margin: 0 }}>Missions</h1></div></div>
      <p className="muted">Each mission is a full ML project: inspect the data → form a hypothesis and predict the result → build and run a real model → check objectives → reflect. Missions unlock when you're proficient in the required concepts.</p>
      <ErrorBox error={error} />
      {!data ? <Loading /> : (
        <div className="grid g3">
          {data.map((m) => {
            const d = ds(m.dataset)
            return (
              <div key={m.id} className={`card mission-card ${m.unlocked ? 'clickable' : 'locked'} ${m.completed ? 'done' : ''}`} onClick={() => m.unlocked && go(`/mission/${m.id}`)}>
                <div className="row between"><span style={{ fontSize: '1.8rem' }}>{d?.icon}</span>{m.completed ? <Pill kind="green">✓ complete</Pill> : m.unlocked ? (m.step > 0 ? <Pill kind="cyan">in progress · step {m.step}/4</Pill> : <Pill kind="violet">available</Pill>) : <Pill>🔒 locked</Pill>}</div>
                <h3 style={{ margin: '0.4rem 0 0.2rem' }}>{m.title}</h3>
                <small className="muted">{meta.areas[m.area]?.name} · {d?.title} · {m.xp} XP</small>
                {!m.unlocked && <div style={{ marginTop: 8 }}><small>Requires: {m.requires.map((r: Any) => <a key={r.id} href={`#/lesson/${r.id}`} onClick={(e) => e.stopPropagation()} style={{ marginRight: 6 }}>{r.name}</a>)}</small></div>}
              </div>
            )
          })}
        </div>
      )}
      <Card title="Want something open-ended?" icon="🔬"><p>The <a href="#/research">Research Institute</a> has challenges with no single right answer, scored on metrics <i>and</i> engineering trade-offs.</p></Card>
    </div>
  )
}
