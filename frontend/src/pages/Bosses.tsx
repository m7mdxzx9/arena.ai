import type { Any } from '../api'
import { ErrorBox, Loading, Pill, go, useApi, useGame } from '../ui'

export default function Bosses() {
  const { pid, meta } = useGame()
  const { data, error } = useApi<Any[]>(`/api/p/${pid}/bosses`)
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">Campus threats</div><h1 style={{ margin: 0 }}>Boss Battles</h1></div></div>
      <p className="muted">Each boss is a classic, real-world AI failure. You don't beat it by clicking fast — you beat it by diagnosing the problem, running an experiment that actually fixes it, and explaining why.</p>
      <ErrorBox error={error} />
      {!data ? <Loading /> : (
        <div className="grid g2">
          {data.map((b) => (
            <div key={b.id} className={`card boss-card ${b.unlocked ? 'clickable' : 'locked'} ${b.defeated ? 'done' : ''}`} onClick={() => b.unlocked && go(`/boss/${b.id}`)}>
              <div className="row between">
                <span className="boss-icon">{b.icon}</span>
                {b.defeated ? <Pill kind="green">defeated</Pill> : b.unlocked ? <Pill kind="red">HP {b.hp}</Pill> : <Pill>🔒 locked</Pill>}
              </div>
              <h3 style={{ margin: '0.3rem 0' }}>{b.name}</h3>
              <small className="muted">{meta.areas[b.area]?.name} · {b.xp} XP</small>
              <p style={{ fontStyle: 'italic', fontSize: '0.88rem' }}>“{b.taunt}”</p>
              {!b.unlocked && <small>Become proficient in: {b.requires.map((r: Any) => <a key={r.id} href={`#/lesson/${r.id}`} onClick={(e) => e.stopPropagation()} style={{ marginRight: 6 }}>{r.name} ({Math.round(r.p * 100)}%)</a>)}</small>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
