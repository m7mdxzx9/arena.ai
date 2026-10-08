import { useState, type ReactNode } from 'react'
import { post, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Rich, Select, Slider, useApi, useGame } from '../ui'

export default function TutorPage() {
  const { t, language } = useI18n()
  const { pid, ov } = useGame()
  const modelData = useApi<Any>('/api/models')
  const runs = useApi<Any[]>(`/api/p/${pid}/runs`)
  const documents = useApi<Any[]>(`/api/p/${pid}/documents`)
  const analytics = useApi<Any>(`/api/p/${pid}/tutor/analytics`)
  const [source, setSource] = useState('offline')
  const [mode, setMode] = useState('simple')
  const [level, setLevel] = useState('auto')
  const [hintLevel, setHintLevel] = useState(1)
  const [runId, setRunId] = useState('auto')
  const [useDocuments, setUseDocuments] = useState(false)
  const [model, setModel] = useState(ov.player.settings.default_model || '')
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [feedbackBusy, setFeedbackBusy] = useState(false)
  const [feedback, setFeedback] = useState('')
  const [answer, setAnswer] = useState<Any>(null)
  const [error, setError] = useState<unknown | null>(null)
  const models = modelData.data?.models || []
  const activeModel = model || models[0]?.name || ''
  const modeOptions = [
    { value: 'simple', label: t('tutor.simple') }, { value: 'example', label: t('tutor.example') },
    { value: 'visual', label: t('tutor.visual') }, { value: 'mathematical', label: t('tutor.mathematical') },
    { value: 'code', label: t('tutor.code') }, { value: 'hint', label: t('tutor.hint') },
    { value: 'question', label: t('tutor.askQuestion') }, { value: 'no_answer', label: t('tutor.noAnswer') },
  ]
  const runOptions = [
    { value: 'auto', label: t('tutor.latestRelevantRun') },
    ...(runs.data || []).map((item: Any) => ({ value: String(item.id), label: `#${item.id} · ${item.name || item.kind} · ${item.summary?.model || item.config?.model || ''}` })),
  ]

  const ask = async () => {
    if (!question.trim()) return
    setBusy(true); setError(null); setAnswer(null); setFeedback('')
    try {
      setAnswer(await post(`/api/p/${pid}/tutor`, {
        question, mode, source, model: source === 'ollama' ? activeModel : null,
        language, level, hint_level: hintLevel,
        run_id: runId === 'auto' ? null : Number(runId),
        use_documents: useDocuments,
      }))
    } catch (e) { setError(e) } finally { setBusy(false) }
  }

  const submitFeedback = async (value: 'helpful' | 'unclear' | 'answered') => {
    if (!answer?.interaction_id) return
    setFeedbackBusy(true); setError(null)
    try {
      await post(`/api/p/${pid}/tutor/${answer.interaction_id}/feedback`, { feedback: value })
      setFeedback(value); analytics.reload()
    } catch (e) { setError(e) } finally { setFeedbackBusy(false) }
  }

  const context = answer?.player_context
  const run = context?.current_experiment
  const mastery = context?.mastery?.focus
  const runsSummary = run ? [
    run.model && `${t('common.model')}: ${run.model}`,
    run.dataset && `${t('datasets.title')}: ${run.dataset}`,
    ...Object.entries(run.train_metrics || {}).slice(0, 2).map(([key, value]) => `train ${key}: ${formatMetric(value)}`),
    ...Object.entries(run.validation_or_test_metrics || {}).slice(0, 2).map(([key, value]) => `test ${key}: ${formatMetric(value)}`),
  ].filter(Boolean) : []

  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('tutor.kicker')}</div><h1>{t('tutor.title')}</h1></div></div>
      <p className="muted">{t('tutor.subtitle')}</p>
      <div className="info-box">🔐 {t('tutor.privacyNote')}</div>
      <div className="grid g-side">
        <Card title={t('tutor.mode')} icon="🧑‍🏫">
          <Select label={t('tutor.source')} value={source} onChange={setSource} options={[{ value: 'offline', label: t('tutor.offline') }, { value: 'ollama', label: t('tutor.local') }]} />
          {source === 'ollama' && <Select label={t('common.model')} value={activeModel} onChange={setModel} options={models.map((item: Any) => ({ value: item.name, label: item.name }))} />}
          {source === 'ollama' && !modelData.data?.reachable && <div className="warn-box">{t('models.ollamaUnavailable')}</div>}
          <Select label={t('tutor.mode')} value={mode} onChange={setMode} options={modeOptions} />
          {mode === 'hint' && <Slider label={t('tutor.hintOf', { level: hintLevel })} value={hintLevel} min={1} max={3} onChange={setHintLevel} />}
          <Select label={t('tutor.level')} value={level} onChange={setLevel} options={[
            { value: 'auto', label: t('tutor.automatic') }, { value: 'beginner', label: t('tutor.beginner') }, { value: 'engineer', label: t('tutor.engineer') },
          ]} />
          <Select label={t('tutor.runContext')} value={runId} onChange={setRunId} options={runOptions} />
          <label className={`tutor-document-toggle ${!documents.data?.length ? 'disabled' : ''}`}>
            <input type="checkbox" checked={useDocuments} disabled={!documents.data?.length} onChange={(event) => setUseDocuments(event.target.checked)} />
            <span>{t('tutor.useDocuments')}</span>
            <Pill>{documents.data?.length || 0}</Pill>
          </label>
          {useDocuments && ['hint', 'no_answer', 'question'].includes(mode) && <small className="muted">{t('tutor.documentsSuppressed')}</small>}
          <small className="muted">{t('tutor.personalizationNote')}</small>
          <label className="field"><span className="field-l">{t('tutor.question')}</span><textarea rows={7} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') void ask() }} /></label>
          <Btn onClick={ask} disabled={busy || !question.trim() || (source === 'ollama' && !activeModel)}>{busy ? t('tutor.thinking') : t('tutor.ask')}</Btn>
          <small className="muted">Ctrl/⌘ + Enter</small>
        </Card>
        <div className="col">
          {busy && <Loading text={t('tutor.thinking')} />}
          <ErrorBox error={error || runs.error || documents.error || analytics.error} />
          {answer && <Card title={answer.source === 'curated_offline' ? t('tutor.offline') : t('tutor.local')} icon="💬" right={<Pill kind={answer.source === 'curated_offline' ? 'cyan' : 'violet'}><Ltr>{answer.source}</Ltr></Pill>}>
            <Rich text={answer.answer} />
            {answer.citation_details?.length > 0 && <div className="citation-list"><h4>{t('tutor.citations')}</h4>{answer.citation_details.map((citation: Any) => <div className="citation-card" key={citation.chunk_id}>
              <div className="row between"><b>{citation.document}{citation.page ? ` · ${t('advancedRag.page')} ${citation.page}` : ''}</b><Pill><Ltr>{citation.chunk_id}</Ltr></Pill></div><p>{citation.excerpt}</p>
            </div>)}</div>}
            <div className="tutor-feedback">
              <strong>{t('tutor.quickCheck')}</strong>
              <div className="row wrap">
                <Btn small kind={feedback === 'helpful' ? 'success' : 'ghost'} disabled={feedbackBusy || Boolean(feedback)} onClick={() => void submitFeedback('helpful')}>{t('tutor.clear')}</Btn>
                <Btn small kind={feedback === 'unclear' ? 'warn' : 'ghost'} disabled={feedbackBusy || Boolean(feedback)} onClick={() => void submitFeedback('unclear')}>{t('tutor.stillUnclear')}</Btn>
                <Btn small kind={feedback === 'answered' ? 'success' : 'ghost'} disabled={feedbackBusy || Boolean(feedback)} onClick={() => void submitFeedback('answered')}>{t('tutor.answerReceived')}</Btn>
              </div>
              {feedback && <small className="success-text">{t('tutor.noted')}</small>}
            </div>
          </Card>}
          {answer && <Card title={t('tutor.contextUsed')} icon="🧭">
            <div className="tutor-context-grid">
              <ContextValue label={t('tutor.runContext')}>
                {run ? <><Ltr>#{run.run_id} · {run.name || run.kind}</Ltr><small>{runsSummary.join(' · ')}</small></> : <span className="muted">{t('tutor.noContext')}</span>}
              </ContextValue>
              <ContextValue label={t('tutor.focusConcept')}>
                {mastery ? <><b>{conceptName(t, mastery.concept_id)}</b><span>{t('tutor.mastery')}: {Math.round(mastery.probability * 100)}% · {t('tutor.attempts')}: {mastery.attempts} · {t(`campaign.statuses.${mastery.status}`)}</span></> : <span className="muted">{t('tutor.noConceptMatch')}</span>}
              </ContextValue>
              <ContextValue label={t('tutor.topicsToReview')}>
                {context?.concepts_due_for_review?.length ? context.concepts_due_for_review.map((item: Any) => <Pill key={item.concept_id}>{conceptName(t, item.concept_id)}</Pill>) : <span className="muted">{t('tutor.noReviewTopics')}</span>}
              </ContextValue>
              <ContextValue label={t('tutor.repeatedMistakes')}>
                {context?.repeated_mistake_patterns?.length ? context.repeated_mistake_patterns.slice(0, 3).map((item: Any) => <Pill key={`${item.concept}-${item.mistake_type}`}>{item.mistake_type} × {item.count}</Pill>) : <span className="muted">{t('tutor.noMistakes')}</span>}
              </ContextValue>
              {context?.current_mission && <ContextValue label={t('tutor.activeMission')}><Ltr>{context.current_mission.id}</Ltr><span>{t('tutor.step')} {context.current_mission.step}</span></ContextValue>}
            </div>
          </Card>}
          {!answer && !busy && <Card><div className="empty">{t('tutor.question')}</div></Card>}
          <Card title={t('tutor.analytics')} icon="📈">
            <div className="row wrap"><Pill>{t('tutor.totalInteractions')}: {analytics.data?.interactions ?? 0}</Pill><Pill>{t('tutor.groundedRequests')}: {analytics.data?.rag_grounded_requests ?? 0}</Pill><Pill>{t('tutor.rawQuestionsNotStored')}</Pill></div>
            <small className="muted">{t('tutor.feedbackDoesNotChangeMastery')}</small>
          </Card>
        </div>
      </div>
    </div>
  )
}

function ContextValue({ label, children }: { label: string; children: ReactNode }) {
  return <section className="tutor-context-item"><div className="field-l">{label}</div><div>{children}</div></section>
}

function conceptName(t: (key: string) => string, conceptId: string) {
  const value = t(`campaign.concepts.${conceptId}`)
  return value === `campaign.concepts.${conceptId}` ? conceptId.replaceAll('_', ' ') : value
}

function formatMetric(value: unknown) {
  if (typeof value !== 'number') return String(value)
  return value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
}
