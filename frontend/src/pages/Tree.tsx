import { useState } from 'react'
import type { Any } from '../api'
import { Bar, Card, Loading, Pill, go, useApi, useGame } from '../ui'

const STATUS: Record<string, [string, string]> = {
  locked: ['🔒 locked', ''], new: ['✨ new', 'cyan'], learning: ['📖 learning', 'violet'], struggling: ['🧩 needs review', 'amber'],
  proficient: ['✅ proficient', 'green'], mastered: ['🏆 mastered', 'amber'],
}
export const StatusPill = ({ status }: { status: string }) => <Pill kind={STATUS[status]?.[1]}>{STATUS[status]?.[0] || status}</Pill>

export default function Tree({ focus }: { focus?: string }) {
  const { pid, meta } = useGame()
  const { data } = useApi(`/api/p/${pid}/tree`)
  const [hover, setHover] = useState<string | null>(focus || null)
  const [filter, setFilter] = useState('')
  if (!data) return <Loading />
  const byId: Record<string, Any> = Object.fromEntries(data.nodes.map((n: Any) => [n.id, n]))
  const prereqSet = new Set<string>()
  const walk = (id: string) => byId[id]?.prereqs.forEach((p: string) => { if (!prereqSet.has(p)) { prereqSet.add(p); walk(p) } })
  if (hover) walk(hover)
  const counts = data.nodes.reduce((acc: Any, n: Any) => ({ ...acc, [n.status]: (acc[n.status] || 0) + 1 }), {})
  return (
    <div className="stack">
      <div className="topbar">
        <h1 style={{ margin: 0 }}>🌳 Knowledge Tree</h1>
        <div className="spacer" />
        <input placeholder="Search concepts…" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: 220 }} />
      </div>
      <Card>
        <div className="row" style={{ gap: '0.5rem' }}>
          {Object.keys(STATUS).map((s) => <span key={s}><StatusPill status={s} /> <small className="muted">{counts[s] || 0}</small></span>)}
        </div>
        <p className="muted" style={{ margin: '0.6rem 0 0' }}><small>
          A concept unlocks when every prerequisite reaches {Math.round(data.thresholds.unlock * 100)}% estimated mastery. Proficient = {Math.round(data.thresholds.proficient * 100)}%, mastered = {Math.round(data.thresholds.mastered * 100)}% (plus at least 3 correct answers).
          Hover a concept to highlight everything it builds on.
        </small></p>
      </Card>
      {data.branches.map((b: string) => {
        const nodes = data.nodes.filter((n: Any) => n.branch === b && (!filter || n.name.toLowerCase().includes(filter.toLowerCase()))).sort((x: Any, y: Any) => x.depth - y.depth)
        if (!nodes.length) return null
        const prof = nodes.filter((n: Any) => n.status === 'proficient' || n.status === 'mastered').length
        return (
          <div key={b} className="tree-branch">
            <div className="section-title"><h2>{b}</h2><small className="muted">{prof} / {nodes.length} proficient</small></div>
            <div className="tree-nodes">
              {nodes.map((n: Any) => (
                <div key={n.id} className={`node ${n.status} ${hover && prereqSet.has(n.id) ? 'hl' : ''}`} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(focus || null)}
                  onClick={() => n.status !== 'locked' && go(`/lesson/${n.id}`)} role="button" tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && n.status !== 'locked' && go(`/lesson/${n.id}`)}>
                  <div className="node-h"><b>{n.name}</b>{n.due && <span title="Spaced review due">🔁</span>}</div>
                  <div className="row between"><StatusPill status={n.status} /><small className="muted">{meta.areas[n.area]?.icon}</small></div>
                  <Bar value={n.p} thin />
                  {n.status === 'locked' && <small className="muted">Needs: {n.prereqs.filter((p: string) => byId[p].p < data.thresholds.unlock).map((p: string) => byId[p].name).join(', ')}</small>}
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}
