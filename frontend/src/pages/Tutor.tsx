import { useState } from 'react'
import { post, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Rich, Select, useApi, useGame } from '../ui'

export default function TutorPage() {
  const { t, language } = useI18n()
  const { pid, ov } = useGame()
  const modelData = useApi<Any>('/api/models')
  const [source, setSource] = useState('offline')
  const [mode, setMode] = useState('simple')
  const [model, setModel] = useState(ov.player.settings.default_model || '')
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [answer, setAnswer] = useState<Any>(null)
  const [error, setError] = useState<string | null>(null)
  const models = modelData.data?.models || []
  const activeModel = model || models[0]?.name || ''
  const ask = async () => {
    if (!question.trim()) return
    setBusy(true); setError(null); setAnswer(null)
    try { setAnswer(await post(`/api/p/${pid}/tutor`, { question, mode, source, model: source === 'ollama' ? activeModel : null, language })) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const modes = [
    { value: 'simple', label: t('tutor.simple') }, { value: 'example', label: t('tutor.example') },
    { value: 'visual', label: t('tutor.visual') }, { value: 'mathematical', label: t('tutor.mathematical') },
    { value: 'code', label: t('tutor.code') }, { value: 'hint', label: t('tutor.hint') },
    { value: 'question', label: t('tutor.askQuestion') }, { value: 'no_answer', label: t('tutor.noAnswer') },
  ]
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('tutor.kicker')}</div><h1>{t('tutor.title')}</h1></div></div>
      <p className="muted">{t('tutor.subtitle')}</p>
      <div className="grid g-side">
        <Card title={t('tutor.mode')} icon="🧑‍🏫">
          <Select label={t('tutor.source')} value={source} onChange={setSource} options={[{ value: 'offline', label: t('tutor.offline') }, { value: 'ollama', label: t('tutor.local') }]} />
          {source === 'ollama' && <Select label={t('common.model')} value={activeModel} onChange={setModel} options={models.map((item: Any) => ({ value: item.name, label: item.name }))} />}
          {source === 'ollama' && !modelData.data?.reachable && <div className="warn-box">{t('models.ollamaUnavailable')}</div>}
          <Select label={t('tutor.mode')} value={mode} onChange={setMode} options={modes} />
          <label className="field"><span className="field-l">{t('tutor.question')}</span><textarea rows={7} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') ask() }} /></label>
          <Btn onClick={ask} disabled={busy || !question.trim() || (source === 'ollama' && !activeModel)}>{busy ? t('tutor.thinking') : t('tutor.ask')}</Btn>
          <small className="muted">Ctrl/⌘ + Enter</small>
        </Card>
        <div className="col">
          {busy && <Loading text={t('tutor.thinking')} />}
          <ErrorBox error={error} />
          {answer && <Card title={answer.source === 'curated_offline' ? t('tutor.offline') : t('tutor.local')} icon="💬" right={<Pill kind={answer.source === 'curated_offline' ? 'cyan' : 'violet'}>{answer.source}</Pill>}>
            <Rich text={answer.answer} />
            <hr />
            <div className="field-l">{t('tutor.contextUsed')}</div>
            {answer.player_context ? <div className="info-box"><Ltr>Run #{answer.player_context.run_id} · {answer.player_context.kind}</Ltr>{answer.player_context.diagnosis?.length ? <div><Ltr>{answer.player_context.diagnosis.join(', ')}</Ltr></div> : null}</div> : <small className="muted">{t('tutor.noContext')}</small>}
          </Card>}
          {!answer && !busy && <Card><div className="empty">{t('tutor.question')}</div></Card>}
        </div>
      </div>
    </div>
  )
}
