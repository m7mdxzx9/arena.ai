import type { Any } from '../api'
import { Ring } from '../charts'
import { useI18n } from '../i18n'
import { Bar, Btn, Card, Pill, go, useGame } from '../ui'

const ROUTES: [string, string][] = [
  ['foundation_academy', 'data_district'], ['data_district', 'ml_workshop'], ['ml_workshop', 'evaluation_chamber'], ['data_district', 'neural_tower'],
  ['neural_tower', 'vision_lab'], ['neural_tower', 'language_center'], ['language_center', 'genai_facility'], ['genai_facility', 'rag_archives'],
  ['rag_archives', 'agent_arena'], ['evaluation_chamber', 'research_institute'], ['agent_arena', 'research_institute'], ['evaluation_chamber', 'rag_archives'],
]
const REC_ICON: Record<string, string> = { review: '🔁', lesson: '📘', mission: '🎯', boss: '⚔️' }

export function recLink(r: Any) {
  return r.kind === 'mission' ? `/mission/${r.id}` : r.kind === 'boss' ? `/boss/${r.id}` : `/lesson/${r.id}`
}

export default function Campus() {
  const { t } = useI18n()
  const { ov } = useGame()
  const areas: Record<string, Any> = ov.areas
  const c = ov.counts
  return (
    <div className="stack">
      <div className="topbar">
        <div>
          <div className="kicker">{t('campus.welcome', { name: ov.player.name })}</div>
          <h1 style={{ margin: 0 }}>{t('campus.title')}</h1>
        </div>
        <div className="spacer" />
        <div className="xpbox"><span className="rank">🎖️ {ov.rank.title}</span><span>{ov.player.xp} XP</span></div>
      </div>
      <div className="campus" role="img" aria-label="Campus map: buildings glow brighter as you master their concepts">
        <div className="stars" />
        <svg className="paths" viewBox="0 0 100 100" preserveAspectRatio="none">
          {ROUTES.map(([a, b]) => {
            const A = areas[a], B = areas[b]
            const lit = A.level > 0 && B.level > 0
            return <line key={a + b} x1={A.x} y1={A.y} x2={B.x} y2={B.y} stroke={lit ? '#22d3ee' : '#334155'} strokeOpacity={lit ? 0.6 : 0.5} strokeWidth={0.35} strokeDasharray={lit ? undefined : '1 1'} vectorEffect="non-scaling-stroke" />
          })}
        </svg>
        {Object.entries(areas).map(([id, a]) => (
          <div key={id} className={`building lv${a.level} ${a.unlocked ? '' : 'locked'}`} style={{ left: `${a.x}%`, top: `${a.y}%`, ['--c' as string]: a.color }} onClick={() => go(`/area/${id}`)} title={a.blurb}>
            <div className="building-icon" style={{ borderColor: a.color }}>
              {a.icon}
              {a.bosses.length > a.bosses_defeated.length && a.unlocked && <span className="boss-flag" title="A boss lurks here">👾</span>}
              {a.bosses.length > 0 && a.bosses.length === a.bosses_defeated.length && <span className="boss-flag" style={{ borderColor: '#059669' }} title="All bosses defeated">🛡️</span>}
            </div>
            <div className="building-name">{a.name}</div>
            <div className="building-lv">{[1, 2, 3].map((l) => <i key={l} className={a.level >= l ? 'on' : ''} />)}</div>
          </div>
        ))}
      </div>
      <div className="grid g3">
        <Card title={t('campus.nextUp')} icon="🧭" className="glow">
          {ov.recommendations.length === 0 && <p className="muted">{t('campus.allReady')}</p>}
          <div className="col">
            {ov.recommendations.map((r: Any) => (
              <div key={r.kind + r.id} className="row between" style={{ gap: '0.5rem' }}>
                <div style={{ minWidth: 0 }}>
                  <div><span style={{ marginRight: 6 }}>{REC_ICON[r.kind]}</span><b>{r.title}</b></div>
                  <small className="muted">{r.why}</small>
                </div>
                <Btn small onClick={() => go(recLink(r))}>{t('campus.go')}</Btn>
              </div>
            ))}
          </div>
        </Card>
        <Card title={t('campus.progress')} icon="📈">
          <div className="row" style={{ gap: '1rem', marginBottom: '0.8rem' }}>
            <Ring value={c.proficient / c.concepts} size={64} label={`${c.proficient}`} />
            <div>{t('campus.conceptsProficient', { done: c.proficient, total: c.concepts })}<br /><small className="muted">{t('campus.masteredProgress', { mastered: c.mastered, learning: c.learning })}</small></div>
          </div>
          <div className="kv">
            <dt>Missions</dt><dd>{c.missions_done} / {c.missions}</dd>
            <dt>Bosses</dt><dd>{c.bosses_defeated} / {c.bosses}</dd>
            <dt>Code Dojo</dt><dd>{c.exercises_done} / {c.exercises}</dd>
          </div>
          {ov.rank.next && (
            <div style={{ marginTop: '0.8rem' }}>
              <small className="muted">{t('campus.nextRank')}: <b className="rank">{ov.rank.next.title}</b></small>
              <div className="col" style={{ gap: '0.3rem', marginTop: '0.3rem' }}>
                <Bar value={ov.player.xp} max={ov.rank.next.xp} label={`${ov.player.xp} / ${ov.rank.next.xp} XP`} />
                <Bar value={c.proficient} max={ov.rank.next.proficient || 1} label={`${c.proficient} / ${ov.rank.next.proficient} proficient`} />
                {ov.rank.next.bosses > 0 && <Bar value={c.bosses_defeated} max={ov.rank.next.bosses} label={`${c.bosses_defeated} / ${ov.rank.next.bosses} bosses`} />}
              </div>
            </div>
          )}
        </Card>
        <Card title={t('campus.recent')} icon="📜">
          {ov.events.length === 0 && <p className="muted">{t('campus.xpEmpty')}</p>}
          {ov.events.map((e: Any, i: number) => (
            <div key={i} className="row between" style={{ fontSize: '0.85rem', padding: '0.2rem 0' }}>
              <span>{e.reason}</span><Pill kind="amber">+{e.xp}</Pill>
            </div>
          ))}
        </Card>
      </div>
    </div>
  )
}
