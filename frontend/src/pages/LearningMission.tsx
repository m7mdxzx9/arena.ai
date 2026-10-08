import { useState } from 'react'
import { post, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Select, useApi, useGame } from '../ui'

export default function LearningMissionPage({ id }: { id: string }) {
  const { t } = useI18n()
  const { pid, reward, refresh } = useGame()
  const { data, error, setData } = useApi<Any>(`/api/p/${pid}/learning-missions/${id}`)
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [selectedGroundingId, setSelectedGroundingId] = useState('')
  const [check, setCheck] = useState<Any>(null)
  const [actionError, setActionError] = useState<unknown | null>(null)
  const [busy, setBusy] = useState(false)
  if (error) return <ErrorBox error={error} />
  if (!data) return <Loading />

  const title = t(`learningMissions.titles.${id}`)
  const description = t(`learningMissions.descriptions.${id}`)
  const experiments: Any[] = data.experiments || []
  const runOptions = experiments.map((item) => ({ value: item.id, label: `#${item.id.slice(0, 8)} · ${item.method} · ${item.query.slice(0, 90)}` }))
  const ladder = data.hint_ladder
  const completeMission = async () => {
    setBusy(true); setActionError(null); setCheck(null)
    const payload = id === 'rag_compare' ? { experiment_ids: selectedIds } : id === 'rag_grounding' ? { experiment_id: selectedGroundingId } : {}
    try {
      const result = await post(`/api/p/${pid}/learning-missions/${id}`, { action: 'complete', payload })
      setCheck(result)
      if (result.passed) {
        setData({ ...data, state: result.state, completed: true })
        if (result.xp || result.achievements?.length) reward(result)
        else refresh()
      }
    } catch (e) { setActionError(e) } finally { setBusy(false) }
  }

  const toggle = (experimentId: string) => setSelectedIds((current) => {
    if (current.includes(experimentId)) return current.filter((value) => value !== experimentId)
    if (current.length >= 2) return [current[1], experimentId]
    return [...current, experimentId]
  })

  return (
    <div className="stack">
      <div className="topbar">
        <a className="btn btn-ghost btn-sm" href="#/missions">← {t('learningMissions.openMissions')}</a>
        <div><div className="kicker">{t('learningMissions.kicker')}</div><h1 style={{ margin: 0 }}>{title}</h1></div>
        <div className="spacer" /><Pill kind="violet">{data.xp} XP</Pill>
      </div>
      <p className="muted">{description}</p>
      <div className="info-box">🔎 {t('learningMissions.instructions')}<br /><small>{t('learningMissions.rewards')}</small></div>
      <ErrorBox error={actionError} />

      {id === 'rag_compare' && <Card title={t('learningMissions.selectRuns')} icon="🧪">
        {experiments.length === 0 ? <p className="muted">{t('learningMissions.noRuns')}</p> : <>
          <p className="muted">{t('learningMissions.compareSameQuestion')} · {t('learningMissions.compareSameSources')} · {t('learningMissions.compareDifferentMethods')}</p>
          <div className="stack-sm">{experiments.map((item) => <label key={item.id} className={`mission-evidence-row ${selectedIds.includes(item.id) ? 'selected' : ''}`}>
            <input type="checkbox" checked={selectedIds.includes(item.id)} onChange={() => toggle(item.id)} />
            <div className="mission-evidence-content"><b>{item.query}</b><div className="row wrap"><Pill><Ltr>{item.method}</Ltr></Pill><Pill><Ltr>{item.embedding_provider}</Ltr></Pill><Pill>{t('learningMissions.retrieved')}: {item.retrieved_count}</Pill><Pill>{t('learningMissions.citations')}: {item.citation_count}</Pill></div></div>
          </label>)}</div>
          <div className="row"><Pill>{t('learningMissions.selectRuns')}: {selectedIds.length}/2</Pill><Btn onClick={completeMission} disabled={busy || selectedIds.length !== 2 || data.completed}>{busy ? t('common.loading') : t('learningMissions.complete')}</Btn></div>
        </>}
      </Card>}

      {id === 'rag_grounding' && <Card title={t('learningMissions.selectRun')} icon="📑">
        {experiments.length === 0 ? <p className="muted">{t('learningMissions.noRuns')}</p> : <>
          <Select label={t('learningMissions.selectRun')} value={selectedGroundingId} onChange={setSelectedGroundingId} options={[{ value: '', label: t('learningMissions.selectRun') }, ...runOptions]} />
          {selectedGroundingId && (() => {
            const selected = experiments.find((item) => item.id === selectedGroundingId)
            return selected ? <div className="retrieved-chunk"><b>{selected.query}</b><div className="row wrap"><Pill><Ltr>{selected.method}</Ltr></Pill><Pill>{t('learningMissions.retrieved')}: {selected.retrieved_count}</Pill><Pill>{t('learningMissions.citations')}: {selected.citation_count}</Pill></div></div> : null
          })()}
          <Btn onClick={completeMission} disabled={busy || !selectedGroundingId || data.completed}>{busy ? t('common.loading') : t('learningMissions.complete')}</Btn>
        </>}
      </Card>}

      {id === 'tutor_hint_ladder' && <Card title={t('learningMissions.hintLevels')} icon="🧑‍🏫">
        <p>{t('learningMissions.savedRunRequired')}</p>
        <div className="row wrap">{[1, 2, 3].map((level) => <Pill key={level} kind={ladder?.context_levels?.includes(level) ? 'green' : ''}>{level}: {ladder?.context_levels?.includes(level) ? '✓' : '—'}</Pill>)}</div>
        {ladder?.concept_id && <p>{t('learningMissions.concept')}: <b>{conceptName(t, ladder.concept_id)}</b></p>}
        {ladder?.run_id && <p>{t('tutor.runContext')}: <Ltr>#{ladder.run_id}</Ltr>{ladder.mastery_probability !== null && <> · {t('tutor.mastery')}: {(ladder.mastery_probability * 100).toFixed(0)}%</>}</p>}
        {!ladder?.context_levels?.length && <p className="muted">{t('learningMissions.noMatchingHints')}</p>}
        <div className="row wrap"><a className="btn btn-ghost btn-sm" href="#/tutor">{t('learningMissions.openTutor')}</a><Btn onClick={completeMission} disabled={busy || data.completed}>{busy ? t('common.loading') : t('learningMissions.complete')}</Btn></div>
      </Card>}

      {check && <Card title={check.passed ? t('learningMissions.completed') : t('learningMissions.progress')} icon={check.passed ? '🏆' : '🔍'} className={check.passed ? 'glow' : ''}>
        <div className="stack-sm">{check.checks?.map((item: Any) => <div className="row" key={item.key}><span>{item.passed ? '✅' : '❌'}</span><span>{checkTitle(t, id, item.key)}</span></div>)}</div>
        {check.passed && <p className="success-text">{t('learningMissions.completed')}</p>}
      </Card>}

      {data.completed && !check && <Card title={t('learningMissions.completed')} icon="🏆" className="glow"><p>{t('learningMissions.completed')} · {t('learningMissions.rewards')}</p></Card>}
      <div className="row wrap"><a className="btn btn-ghost btn-sm" href="#/personal-rag">{t('learningMissions.openAdvancedRag')}</a><a className="btn btn-ghost btn-sm" href="#/tutor">{t('learningMissions.openTutor')}</a></div>
    </div>
  )
}

function conceptName(t: (key: string) => string, id: string) {
  const key = `campaign.concepts.${id}`
  const name = t(key)
  return name === key ? id.replaceAll('_', ' ') : name
}

function checkTitle(t: (key: string) => string, missionId: string, key: string) {
  const map: Record<string, Record<string, string>> = {
    rag_compare: { same_question: 'compareSameQuestion', same_sources: 'compareSameSources', different_methods: 'compareDifferentMethods', retrieved_evidence: 'compareResults' },
    rag_grounding: { retrieved_evidence: 'hasRetrievedChunk', citation_matches_retrieved_chunk: 'matchingCitation' },
    tutor_hint_ladder: { same_concept_progressive_hints: 'sameConceptHints', actual_saved_run_context: 'actualRunContext' },
  }
  const lookup = map[missionId]?.[key]
  return lookup ? t(`learningMissions.${lookup}`) : key.replaceAll('_', ' ')
}
