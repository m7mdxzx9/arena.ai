import { useState } from 'react'
import { post, type Any } from '../api'
import { Bar, Btn, Card, Pill, Toggle, go, useApi, useGame } from '../ui'

export default function Profile() {
  const { pid, ov, meta, refresh, setPid, toast } = useGame()
  const review = useApi<Any[]>(`/api/p/${pid}/review`)
  const refl = useApi<Any[]>(`/api/p/${pid}/reflections`)
  const [busy, setBusy] = useState(false)
  const save = async (body: Any) => {
    setBusy(true)
    try { await post(`/api/p/${pid}/settings`, body); await refresh(); toast({ kind: 'info', text: 'Settings saved' }) } finally { setBusy(false) }
  }
  const r = ov.rank
  const earned = ov.achievements.filter((a: Any) => a.earned_at)
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">Player profile</div><h1 style={{ margin: 0 }}>{ov.player.name}</h1></div><div className="spacer" /><Btn kind="ghost" small onClick={() => setPid(null)}>Switch player</Btn></div>
      <div className="grid g2">
        <Card title="Career" icon="🎖️">
          <div className="rank-ladder">
            {r.all.map((t: string, i: number) => <div key={t} className={`rung ${i < r.index ? 'done' : i === r.index ? 'cur' : ''}`}><span>{i + 1}</span>{t}</div>)}
          </div>
          {r.next && (
            <div className="col" style={{ marginTop: 10 }}>
              <b>Next rank: {r.next.title}</b>
              <Bar value={ov.player.xp} max={r.next.xp} label={`XP ${ov.player.xp} / ${r.next.xp}`} />
              <Bar value={ov.counts.proficient} max={Math.max(1, r.next.proficient)} label={`proficient concepts ${ov.counts.proficient} / ${r.next.proficient}`} />
              {r.next.bosses > 0 && <Bar value={ov.counts.bosses_defeated} max={r.next.bosses} label={`bosses ${ov.counts.bosses_defeated} / ${r.next.bosses}`} />}
              <small className="muted">Ranks need XP <i>and</i> demonstrated competence — you can't grind XP alone.</small>
            </div>
          )}
        </Card>
        <Card title="Difficulty mode" icon="🎚️">
          <div className="col">
            {Object.entries(meta.modes).map(([id, m]: [string, Any]) => (
              <button key={id} className={`option ${Number(id) === ov.player.mode ? 'sel' : ''}`} disabled={busy} onClick={() => save({ mode: Number(id) })}>
                <span className="key">{id}</span><span style={{ flex: 1 }}><b>{m.name}</b><br /><small className="muted">{m.desc}</small></span>
              </button>
            ))}
            <Toggle label="Show generated code in labs (Code Mode)" checked={!!ov.player.settings.show_code} onChange={(v) => save({ show_code: v })} />
            <Toggle label="Free Play — unlock every area, lab and lesson (progress still tracked honestly)" checked={!!ov.player.settings.free_play} onChange={(v) => save({ free_play: v })} />
          </div>
        </Card>
      </div>
      <div className="grid g2">
        <Card title={`Spaced review (${review.data?.length ?? 0} due)`} icon="🔁">
          {review.data?.length ? review.data.map((c) => <div key={c.id} className="row between"><a href={`#/lesson/${c.id}`}>{c.name}</a><Pill>box {c.box}</Pill></div>) : <p className="muted">Nothing due. Concepts you've learned return here after 1, 3, 7, 16 and 35 days — just before you'd forget them.</p>}
        </Card>
        <Card title="Your reflections" icon="🪞">
          {refl.data?.length ? <div className="col" style={{ maxHeight: 260, overflowY: 'auto' }}>{refl.data.map((x, i) => <div key={i}><small className="muted">{meta.concepts[x.concept_id]?.name} · {new Date(x.ts * 1000).toLocaleDateString()}</small><div>{x.text}</div></div>)}</div> : <p className="muted">Reflections you write at the end of lessons collect here.</p>}
        </Card>
      </div>
      <Card title={`Achievements (${earned.length}/${ov.achievements.length})`} icon="🏅">
        <div className="grid g4">
          {ov.achievements.map((a: Any) => (
            <div key={a.id} className={`ach ${a.earned_at ? 'on' : ''}`} title={a.desc}><span className="ach-i">{a.earned_at ? a.icon : '🔒'}</span><div><b>{a.name}</b><br /><small className="muted">{a.desc}</small></div></div>
          ))}
        </div>
      </Card>
      <Card title="Recent activity" icon="📈">
        {ov.events.length ? ov.events.map((e: Any, i: number) => <div key={i} className="row between"><small>{e.reason}</small><Pill kind="violet">+{e.xp} XP</Pill></div>) : <p className="muted">No XP earned yet. <a href="#/" onClick={() => go('/')}>Start on the campus →</a></p>}
      </Card>
    </div>
  )
}
