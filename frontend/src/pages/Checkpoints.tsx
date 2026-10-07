import { useState } from 'react'
import { del, patch, type Any } from '../api'
import { Ltr, useI18n } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, useApi, useGame } from '../ui'

export default function CheckpointsPage() {
  const { t, locale } = useI18n()
  const { pid } = useGame()
  const checkpoints = useApi<Any[]>(`/api/p/${pid}/checkpoints`)
  const [selected, setSelected] = useState<string | null>(null)
  const [editing, setEditing] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const rename = async (id: string) => {
    setBusy(true); setError(null)
    try { await patch(`/api/p/${pid}/checkpoints/${id}`, { name }); setEditing(null); checkpoints.reload() } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const remove = async (id: string) => {
    if (!window.confirm(t('checkpoints.deleteConfirm'))) return
    setBusy(true); setError(null)
    try { await del(`/api/p/${pid}/checkpoints/${id}`); if (selected === id) setSelected(null); checkpoints.reload() } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  return <div className="stack">
    <div className="topbar"><div><div className="kicker">{t('checkpoints.kicker')}</div><h1>{t('checkpoints.title')}</h1></div></div>
    <p className="muted">{t('checkpoints.subtitle')}</p>
    <div className="info-box">{t('checkpoints.security')}</div>
    <ErrorBox error={error || checkpoints.error} />
    {checkpoints.loading && <Loading />}
    {checkpoints.data?.length === 0 && <Card><div className="empty">{t('checkpoints.empty')}</div></Card>}
    <div className="checkpoint-grid">{checkpoints.data?.map((checkpoint) => <Card key={checkpoint.id} title={checkpoint.name} icon={checkpoint.kind === 'cnn' ? '👁️' : '🔥'} right={<Pill kind={checkpoint.available ? 'green' : 'red'}>{checkpoint.available ? t('common.available') : t('common.unavailable')}</Pill>}>
      <div className="row"><Pill><Ltr>{checkpoint.kind}</Ltr></Pill><Pill><Ltr>run #{checkpoint.run_id}</Ltr></Pill><small>{new Date(checkpoint.created_at * 1000).toLocaleString(locale)}</small></div>
      <div className="metrics">
        <div className="metric"><div className="metric-l">{t('models.size')}</div><div className="metric-v">{(checkpoint.size_bytes / 1024).toFixed(1)} KiB</div></div>
        <div className="metric"><div className="metric-l">SHA-256</div><div className="metric-v hash-short"><Ltr>{checkpoint.sha256.slice(0, 12)}…</Ltr></div></div>
      </div>
      {editing === checkpoint.id ? <div className="row"><input value={name} maxLength={160} onChange={(event) => setName(event.target.value)} /><Btn small onClick={() => rename(checkpoint.id)} disabled={busy || !name.trim()}>{t('common.save')}</Btn><Btn small kind="ghost" onClick={() => setEditing(null)}>{t('common.cancel')}</Btn></div> : null}
      {selected === checkpoint.id && <pre className="json-view" dir="ltr">{JSON.stringify(checkpoint.metadata, null, 2)}</pre>}
      <div className="row">
        <Btn small kind="ghost" onClick={() => setSelected(selected === checkpoint.id ? null : checkpoint.id)}>{t('checkpoints.metadata')}</Btn>
        <Btn small kind="ghost" onClick={() => { setEditing(checkpoint.id); setName(checkpoint.name) }}>{t('checkpoints.rename')}</Btn>
        <a className={`btn ghost small ${checkpoint.available ? '' : 'disabled'}`} href={checkpoint.available ? `/api/p/${pid}/checkpoints/${checkpoint.id}/download` : undefined}>{t('checkpoints.download')}</a>
        <Btn small kind="danger" onClick={() => remove(checkpoint.id)} disabled={busy}>{t('common.delete')}</Btn>
      </div>
    </Card>)}</div>
  </div>
}
