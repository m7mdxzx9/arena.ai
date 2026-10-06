import type { Any } from '../api'
import { Bar, Btn, Card, Loading, Mentor, Pill, go, useApi, useGame } from '../ui'
import { StatusPill } from './Tree'

const AREA_LABS: Record<string, { path: string; label: string }[]> = {
  foundation_academy: [{ path: '/dojo', label: '⌨️ Code Dojo' }],
  data_district: [{ path: '/data', label: '🔍 Data Lab' }],
  ml_workshop: [{ path: '/workbench', label: '⚙️ ML Workbench' }, { path: '/history', label: '📓 Experiments' }],
  evaluation_chamber: [{ path: '/history', label: '⚖️ Compare runs' }, { path: '/predict', label: '🔮 Prediction Lab' }],
  neural_tower: [{ path: '/nn', label: '🧠 Neural Net Lab' }],
  vision_lab: [{ path: '/nn', label: '🧠 Train on digits' }],
  language_center: [{ path: '/language', label: '💬 Language Lab' }],
  genai_facility: [{ path: '/language', label: '💬 Language Lab' }, { path: '/rag', label: '👻 Grounded answers' }],
  rag_archives: [{ path: '/rag', label: '📚 RAG Archives' }],
  agent_arena: [{ path: '/agent', label: '🤖 Agent Arena' }],
  research_institute: [{ path: '/research', label: '🔭 Research Challenges' }],
}

export default function AreaPage({ id }: { id: string }) {
  const { pid, ov } = useGame()
  const tree = useApi(`/api/p/${pid}/tree`)
  const missions = useApi(`/api/p/${pid}/missions`)
  const bosses = useApi(`/api/p/${pid}/bosses`)
  const a = ov.areas[id]
  if (!a) return <div className="empty">Unknown area</div>
  if (!tree.data) return <Loading />
  const concepts = tree.data.nodes.filter((n: Any) => n.area === id)
  const ms = (missions.data || []).filter((m: Any) => m.area === id)
  const bs = (bosses.data || []).filter((b: Any) => b.area === id)
  return (
    <div className="stack">
      <div className="topbar">
        <Btn kind="ghost" small onClick={() => go('/')}>← Campus</Btn>
        <h1 style={{ margin: 0 }}>{a.icon} {a.name}</h1>
        <Pill kind="cyan">Building level {a.level} / 3</Pill>
        <div className="spacer" />
        {(AREA_LABS[id] || []).map((l) => <Btn key={l.path} kind="ghost" small onClick={() => go(l.path)}>{l.label}</Btn>)}
      </div>
      <Mentor name={a.mentor} icon={a.icon} color={a.color}>
        <p style={{ margin: 0 }}>{a.greeting}</p>
        <small className="muted">{a.blurb}</small>
      </Mentor>
      <div><Bar value={a.proficient} max={a.total} label={`${a.proficient} / ${a.total} concepts proficient — the building powers up at 15%, 50% and 100%`} /></div>
      <div className="grid g-side">
        <div className="col">
          {bs.length > 0 && (
            <Card title="Boss" icon="⚔️" className="danger">
              {bs.map((b: Any) => (
                <div key={b.id} className="col" style={{ gap: '0.3rem' }}>
                  <b>{b.icon} {b.name}</b>
                  <small className="muted">“{b.taunt}”</small>
                  <div>{b.defeated ? <Pill kind="green">Defeated</Pill> : b.unlocked ? <Btn small kind="danger" onClick={() => go(`/boss/${b.id}`)}>Challenge</Btn> : <small className="muted">Requires: {b.requires.map((r: Any) => r.name).join(', ')}</small>}</div>
                </div>
              ))}
            </Card>
          )}
          <Card title="Missions" icon="🎯">
            {ms.length === 0 && <p className="muted">No missions in this building — its concepts feed missions elsewhere.</p>}
            {ms.map((m: Any) => (
              <div key={m.id} className="row between" style={{ padding: '0.3rem 0' }}>
                <span>{m.completed ? '✅' : m.unlocked ? '🎯' : '🔒'} {m.title}</span>
                {m.unlocked && <Btn small kind="ghost" onClick={() => go(`/mission/${m.id}`)}>{m.completed ? 'Replay' : 'Start'}</Btn>}
              </div>
            ))}
          </Card>
        </div>
        <Card title="Concepts taught here" icon="📘">
          <div className="tree-nodes">
            {concepts.map((n: Any) => (
              <div key={n.id} className={`node ${n.status}`} onClick={() => n.status !== 'locked' && go(`/lesson/${n.id}`)} title={n.status === 'locked' ? 'Locked: master the prerequisites first' : ''}>
                <div className="node-h"><b>{n.name}</b>{n.due && <span title="Review due">🔁</span>}</div>
                <StatusPill status={n.status} />
                <Bar value={n.p} thin />
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}
