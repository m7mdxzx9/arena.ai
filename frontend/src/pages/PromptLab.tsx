import { useEffect, useState } from 'react'
import { del, patch, post, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Select, Tabs, useApi, useGame } from '../ui'

function parseObject(value: string): Record<string, Any> {
  const parsed = JSON.parse(value || '{}')
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Variables must be a JSON object.')
  return parsed
}

export default function PromptLabPage() {
  const { t, language } = useI18n()
  const { pid, ov } = useGame()
  const prompts = useApi<Any[]>(`/api/p/${pid}/prompts`)
  const models = useApi<Any>('/api/models')
  const [selectedId, setSelectedId] = useState('')
  const [creating, setCreating] = useState(false)
  const activeSelectedId = creating ? '' : selectedId || prompts.data?.[0]?.id || ''
  const detail = useApi<Any>(activeSelectedId ? `/api/p/${pid}/prompts/${activeSelectedId}` : null, [activeSelectedId])
  const [name, setName] = useState('')
  const [system, setSystem] = useState('You are a precise AI tutor. Answer only the requested question.')
  const [user, setUser] = useState('Explain {{concept}} with one example.')
  const [variables, setVariables] = useState('{"concept":"overfitting"}')
  const [note, setNote] = useState('')
  const [version, setVersion] = useState('1')
  const [model, setModel] = useState(ov.player.settings.default_model || '')
  const [busy, setBusy] = useState(false)
  const [output, setOutput] = useState<Any>(null)
  const [error, setError] = useState<string | null>(null)
  const modelList = models.data?.models || []
  const activeModel = model || modelList[0]?.name || ''
  // Server data initializes the editable draft; later edits remain event-driven.
  // oxlint-disable-next-line react/set-state-in-effect
  useEffect(() => { if (detail.data?.versions?.length) { const latest = detail.data.versions.at(-1); setName(detail.data.name || ''); setVersion(String(latest.version)); setSystem(latest.system); setUser(latest.user); setVariables(JSON.stringify(latest.variables || {}, null, 2)) } }, [detail.data])
  const create = async () => {
    setBusy(true); setError(null)
    try { const result = await post(`/api/p/${pid}/prompts`, { name, system, user, variables: parseObject(variables) }); prompts.reload(); setCreating(false); setSelectedId(result.id) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const startNew = () => {
    setCreating(true); setSelectedId(''); setName(''); setSystem('You are a precise AI tutor.'); setUser('Explain {{concept}} with one example.'); setVariables('{"concept":"overfitting"}'); setNote(''); setOutput(null)
  }
  const rename = async () => {
    if (!activeSelectedId) return
    setBusy(true); setError(null)
    try { await patch(`/api/p/${pid}/prompts/${activeSelectedId}`, { name }); prompts.reload(); detail.reload() } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const remove = async () => {
    if (!activeSelectedId || !window.confirm(t('promptLab.deleteConfirm'))) return
    setBusy(true); setError(null)
    try { await del(`/api/p/${pid}/prompts/${activeSelectedId}`); setSelectedId(''); setCreating(false); setOutput(null); prompts.reload() } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const saveVersion = async () => {
    if (!activeSelectedId) return
    setBusy(true); setError(null)
    try { await post(`/api/p/${pid}/prompts/${activeSelectedId}/versions`, { system, user, variables: parseObject(variables), change_note: note }); detail.reload(); prompts.reload(); setNote('') } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const execute = async () => {
    setBusy(true); setError(null); setOutput(null)
    try { setOutput(await post(`/api/p/${pid}/prompts/${activeSelectedId}/execute`, { version: Number(version), variables: parseObject(variables), model: activeModel })) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const versions = detail.data?.versions || []
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('promptLab.kicker')}</div><h1>{t('promptLab.title')}</h1></div></div>
      <p className="muted">{t('promptLab.subtitle')}</p>
      <div className="grid g-side">
        <div className="col">
          <Card title={creating || !activeSelectedId ? t('promptLab.create') : t('promptLab.edit')} icon="✍️" right={activeSelectedId ? <Btn small kind="ghost" onClick={startNew}>{t('promptLab.newPrompt')}</Btn> : undefined}>
            <label className="field"><span className="field-l">{t('promptLab.name')}</span><input value={name} onChange={(event) => setName(event.target.value)} /></label>
            <label className="field"><span className="field-l">{t('promptLab.system')}</span><textarea rows={5} value={system} onChange={(event) => setSystem(event.target.value)} /></label>
            <label className="field"><span className="field-l">{t('promptLab.user')}</span><textarea rows={6} value={user} onChange={(event) => setUser(event.target.value)} /></label>
            <label className="field"><span className="field-l">{t('promptLab.variables')}</span><textarea className="technical-ltr" dir="ltr" rows={4} value={variables} onChange={(event) => setVariables(event.target.value)} /></label>
            {!activeSelectedId ? <Btn onClick={create} disabled={busy || !name.trim() || !user.trim()}>{t('promptLab.create')}</Btn> : <><label className="field"><span className="field-l">{t('promptLab.changeNote')}</span><input value={note} onChange={(event) => setNote(event.target.value)} /></label><div className="row"><Btn onClick={saveVersion} disabled={busy}>{t('promptLab.saveVersion')}</Btn><Btn kind="ghost" onClick={rename} disabled={busy || !name.trim()}>{t('promptLab.rename')}</Btn><Btn kind="danger" onClick={remove} disabled={busy}>{t('common.delete')}</Btn></div></>}
          </Card>
        </div>
        <div className="col">
          <Card title={t('promptLab.versions')} icon="🗂️">
            {!prompts.data?.length && <p className="muted">{t('promptLab.noPrompts')}</p>}
            {prompts.data?.length && !creating ? <Select value={activeSelectedId} onChange={(value) => { setCreating(false); setSelectedId(value) }} options={prompts.data.map((prompt) => ({ value: prompt.id, label: `${prompt.name} · V${prompt.latest_version}` }))} /> : null}
            {versions.length > 0 && <>
              <Tabs value={version} onChange={(value) => { setVersion(value); const item = versions.find((entry: Any) => String(entry.version) === value); if (item) { setSystem(item.system); setUser(item.user); setVariables(JSON.stringify(item.variables || {}, null, 2)) } }} tabs={versions.map((item: Any) => ({ id: String(item.version), label: `V${item.version}` }))} />
              <div className="version-compare">{versions.slice(-3).map((item: Any) => <div className="card inner" key={item.version}><b>V{item.version}</b> <small>{new Date(item.created_at * 1000).toLocaleString(language === 'ar' ? 'ar-SA' : 'en')}</small><p>{item.change_note}</p><pre className="json-view" dir="ltr">SYSTEM\n{item.system}\n\nUSER\n{item.user}</pre></div>)}</div>
              <Select label={t('common.model')} value={activeModel} onChange={setModel} options={modelList.map((item: Any) => ({ value: item.name, label: item.name }))} />
              {!models.data?.reachable && <div className="warn-box">{t('models.ollamaUnavailable')}</div>}
              <Btn onClick={execute} disabled={busy || !activeModel || !models.data?.reachable}>{busy ? t('common.loading') : t('promptLab.execute')}</Btn>
            </>}
          </Card>
          <ErrorBox error={error || prompts.error || detail.error} />
          {busy && <Loading />}
          {output && <Card title={t('promptLab.output')} icon="💬" right={<><Pill><Ltr>{output.model}</Ltr></Pill><small>{output.duration_ms} ms</small></>}><p>{output.output}</p><details><summary>{t('promptLab.compare')}</summary><pre className="json-view" dir="ltr">{JSON.stringify(output.rendered, null, 2)}</pre></details></Card>}
        </div>
      </div>
    </div>
  )
}
