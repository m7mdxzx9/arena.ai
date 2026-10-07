import { useState } from 'react'
import { del, post, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, useApi, useGame } from '../ui'

export default function ModelHub() {
  const { t } = useI18n()
  const { pid, ov, refresh, toast } = useGame()
  const models = useApi<Any>('/api/models')
  const capabilities = useApi<Any>('/api/system/capabilities')
  const [detail, setDetail] = useState<Any>(null)
  const [pullName, setPullName] = useState('')
  const [operationError, setOperationError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const inspect = async (name: string) => {
    setBusy(name)
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(name)}`)
      const body = await response.json()
      if (!response.ok) throw new Error(body.error || 'Model information is unavailable.')
      setDetail(body)
    } catch (error: Any) { toast({ kind: 'error', text: error.message }) } finally { setBusy(null) }
  }
  const setDefault = async (name: string) => {
    await post(`/api/p/${pid}/settings`, { default_model: name })
    await refresh()
    toast({ kind: 'info', text: t('common.settingsSaved') })
  }
  const pull = async () => {
    setBusy('pull'); setOperationError(null)
    try { await post('/api/models/pull', { model: pullName, provider: 'ollama' }); setPullName(''); models.reload(); toast({ kind: 'info', text: t('models.pullComplete') }) } catch (error: Any) { setOperationError(error.message) } finally { setBusy(null) }
  }
  const remove = async (name: string) => {
    if (!window.confirm(t('models.deleteConfirm', { name }))) return
    setBusy(`delete:${name}`); setOperationError(null)
    try { await del(`/api/models/${encodeURIComponent(name)}`); if (detail?.name === name) setDetail(null); models.reload() } catch (error: Any) { setOperationError(error.message) } finally { setBusy(null) }
  }
  const list = models.data?.models || []
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('models.kicker')}</div><h1>{t('models.title')}</h1></div><div className="spacer" /><Btn kind="ghost" onClick={() => { models.reload(); capabilities.reload() }}>{t('models.refresh')}</Btn></div>
      <p className="muted">{t('models.subtitle')}</p>
      <Card title={t('models.manage')} icon="📦">
        <p className="muted">{t('models.manageHelp')}</p>
        <div className="row"><input className="technical-ltr" dir="ltr" value={pullName} onChange={(event) => setPullName(event.target.value)} placeholder={t('models.pullPlaceholder')} /><Btn onClick={pull} disabled={busy !== null || !pullName.trim() || !models.data?.reachable}>{busy === 'pull' ? t('models.pulling') : t('models.pull')}</Btn></div>
        <ErrorBox error={operationError} />
      </Card>
      <ErrorBox error={models.error || capabilities.error} />
      {models.loading && <Loading />}
      {models.data && !models.data.reachable && <div className="warn-box">🦙 {t('models.ollamaUnavailable')}<br /><small>{models.data.error}</small></div>}
      {models.data?.reachable && list.length === 0 && <Card><div className="empty">{t('models.noModels')}</div></Card>}
      {list.length > 0 && <div className="grid g2">{list.map((model: Any) => (
        <Card key={model.name} title={<Ltr>{model.name}</Ltr>} icon="🧠" right={ov.player.settings.default_model === model.name ? <Pill kind="green">{t('models.defaultModel')}</Pill> : undefined}>
          <div className="kv">
            <dt>{t('models.provider')}</dt><dd><Ltr>{model.provider}</Ltr></dd>
            <dt>{t('models.family')}</dt><dd><Ltr>{model.family || t('models.unknown')}</Ltr></dd>
            <dt>{t('models.size')}</dt><dd>{model.size_bytes ? `${(model.size_bytes / 1024 ** 3).toFixed(2)} GB` : t('models.unknown')}</dd>
            <dt>{t('models.quantization')}</dt><dd><Ltr>{model.quantization || t('models.unknown')}</Ltr></dd>
            <dt>{t('models.capabilities')}</dt><dd>{model.capabilities ? model.capabilities.join(', ') : t('models.unknown')}</dd>
          </div>
          <div className="row" style={{ marginTop: 10 }}><Btn small kind="ghost" onClick={() => inspect(model.name)} disabled={busy === model.name}>{busy === model.name ? t('common.loading') : t('datasets.inspect')}</Btn><Btn small onClick={() => setDefault(model.name)}>{t('models.selectDefault')}</Btn><Btn small kind="danger" onClick={() => remove(model.name)} disabled={busy === `delete:${model.name}`}>{t('common.delete')}</Btn></div>
        </Card>
      ))}</div>}
      {detail && <Card title={<><Ltr>{detail.name}</Ltr> · {t('datasets.inspect')}</>} icon="🔬">
        <div className="metrics"><div className="metric"><div className="metric-l">context</div><div className="metric-v">{detail.context_length?.toLocaleString() || '—'}</div></div><div className="metric"><div className="metric-l">{t('models.capabilities')}</div><div className="metric-v" style={{ fontSize: '.9rem' }}>{detail.capabilities?.join(', ') || t('models.unknown')}</div></div></div>
        <pre className="json-view" dir="ltr">{JSON.stringify(detail.details, null, 2)}</pre>
      </Card>}
    </div>
  )
}
