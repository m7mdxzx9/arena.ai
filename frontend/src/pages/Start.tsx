import { useEffect, useState } from 'react'
import { get, post, type Any } from '../api'
import { useI18n } from '../i18n'
import { Btn, Card, ErrorBox } from '../ui'

const MODE_ICONS: Record<number, string> = { 1: '🌱', 2: '🧭', 3: '🏋️', 4: '🛠️', 5: '🔭' }

export default function Start({ meta, onPlayer }: { meta: Any; onPlayer: (id: number) => void }) {
  const { t } = useI18n()
  const [players, setPlayers] = useState<Any[]>([])
  const [name, setName] = useState('')
  const [mode, setMode] = useState(1)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => { get('/api/players').then(setPlayers).catch(() => {}) }, [])
  const modes: Any = meta.modes
  const create = async () => {
    setBusy(true); setErr(null)
    try { const overview = await post('/api/players', { name: name || t('start.apprentice'), mode }); onPlayer(overview.player.id) } catch (error: Any) { setErr(error.message) } finally { setBusy(false) }
  }
  return (
    <div className="main" style={{ margin: '0 auto' }}>
      <div className="hero">
        <div className="kicker">{t('start.kicker')}</div>
        <h1><span>NEURAL FORGE</span></h1>
        <p style={{ fontSize: '1.1rem', maxWidth: 760 }}>{t('start.intro')}</p>
        <div className="row" style={{ gap: '1.2rem', marginTop: '0.8rem', flexWrap: 'wrap' }}>
          <span className="pill cyan">{t('start.concepts', { count: Object.keys(meta.concepts).length })}</span>
          <span className="pill green">{t('start.dataLabs', { count: meta.datasets.filter((dataset: Any) => !dataset.toy).length })}</span>
          <span className="pill red">{t('start.bosses')}</span><span className="pill amber">{t('start.computation')}</span><span className="pill violet">{t('start.dojo')}</span>
        </div>
      </div>
      <div className="grid g2" style={{ marginTop: '1.2rem' }}>
        <Card title={t('start.newResearcher')} icon="🧑‍🔬">
          <label className="field"><span className="field-l">{t('start.yourName')}</span><input value={name} maxLength={40} placeholder={t('start.apprentice')} onChange={(event) => setName(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && create()} /></label>
          <div className="field-l" style={{ margin: '0.9rem 0 0.4rem' }}>{t('start.difficulty')}</div>
          <div className="col">{modes && Object.entries(modes).map(([key, item]: [string, Any]) => <div key={key} className={`mode-card ${mode === Number(key) ? 'on' : ''}`} onClick={() => setMode(Number(key))} role="radio" aria-checked={mode === Number(key)} tabIndex={0} onKeyDown={(event) => event.key === 'Enter' && setMode(Number(key))}><b>{MODE_ICONS[Number(key)]} {item.name}</b><div className="muted"><small>{item.desc}</small></div></div>)}</div>
          <ErrorBox error={err} />
          <div style={{ marginTop: '1rem' }}><Btn onClick={create} disabled={busy}>{t('start.enter')}</Btn></div>
        </Card>
        <div className="col">
          <Card title={t('start.continue')} icon="💾">
            {players.length === 0 && <p className="muted">{t('start.noSaved')}</p>}
            {players.map((player) => <div key={player.id} className="row between" style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--line)' }}><span><b>{player.name}</b> <small className="muted" dir="ltr">· {player.xp} XP</small></span><Btn small kind="ghost" onClick={() => onPlayer(player.id)}>{t('start.play')}</Btn></div>)}
          </Card>
          <Card title={t('start.how')} icon="🧪">
            <ol style={{ margin: 0, paddingInlineStart: '1.2rem', lineHeight: 1.8 }}><li>{t('start.teach')}</li><li>{t('start.visualize')}</li><li>{t('start.predict')}</li><li>{t('start.experiment')}</li><li>{t('start.reflect')}</li><li>{t('start.review')}</li></ol>
          </Card>
        </div>
      </div>
    </div>
  )
}
