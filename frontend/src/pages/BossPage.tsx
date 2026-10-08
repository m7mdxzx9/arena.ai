import { useState } from 'react'
import { useI18n } from '../i18n'
import { post, type Any } from '../api'
import { AgentResult } from '../labs/AgentLab'
import AgentLab from '../labs/AgentLab'
import RagLab, { AnswersResult, RetrievalResult } from '../labs/RagLab'
import Workbench, { RunResult } from '../labs/Workbench'
import { Bar, Btn, Card, ErrorBox, Loading, Pill, Rich, Select, go, useApi, useGame } from '../ui'
import { DataProfile } from './DataLab'
import { NNLab } from './NNLabPage'
import { LineChart, PALETTE } from '../charts'

function IntroView({ intro }: { intro: Any }) {
  const r = intro.run
  if (intro.kind === 'profile') return <DataProfile ds={intro.dataset} compact />
  if (intro.kind === 'ml') return <RunResult res={r} config={{}} showCode={false} />
  if (intro.kind === 'nn') return (
    <div className="grid g2">
      <div><div className="field-l">Loss</div><LineChart logY series={[{ name: 'train loss', values: r.history.train_loss }, { name: 'val loss', values: r.history.val_loss, color: PALETTE[1] }]} x={r.history.epoch} xLabel="epoch" /></div>
      <div><div className="field-l">Accuracy</div><LineChart yDomain={[0, 1]} series={[{ name: 'train acc', values: r.history.train_acc }, { name: 'val acc', values: r.history.val_acc, color: PALETTE[1] }]} x={r.history.epoch} xLabel="epoch" /></div>
    </div>
  )
  if (intro.kind === 'answers') return <AnswersResult res={r} />
  if (intro.kind === 'rag') return <RetrievalResult res={r} />
  if (intro.kind === 'agent') return <AgentResult res={r} />
  return null
}

export default function BossPage({ id }: { id: string }) {
  const { t } = useI18n()
  const { pid, reward, refresh } = useGame()
  const { data: b, error, setData } = useApi(`/api/p/${pid}/bosses/${id}`)
  const ragExperiments = useApi<Any[]>(id === 'retrieval' ? `/api/p/${pid}/rag/experiments` : null)
  const [choice, setChoice] = useState<number | null>(null)
  const [sel, setSel] = useState<number[]>([])
  const [ev, setEv] = useState<Any>(null)
  const [hit, setHit] = useState(false)
  const [err, setErr] = useState<unknown | null>(null)
  const [busy, setBusy] = useState(false)
  const [selectedRagExperiment, setSelectedRagExperiment] = useState('')
  if (error) return <ErrorBox error={error} />
  if (!b) return <Loading />
  const st = b.state
  const phase = b.phases[st.phase]
  const won = st.phase >= b.phases.length

  const send = async (payload: Any) => {
    setErr(null); setBusy(true)
    try {
      const r = await post(`/api/p/${pid}/bosses/${id}`, payload)
      setEv({ ...r.evaluation, phase: st.phase, phaseKey: phase.i18n_key, damage: r.damage, lesson: r.lesson })
      if (r.evaluation?.passed) { setHit(true); setTimeout(() => setHit(false), 700) }
      setData({ ...b, state: { ...b.state, ...r.state, defeated: b.state.defeated || !!r.defeated } })
      setChoice(null); setSel([])
      if (r.xp || r.achievements?.length) reward(r); else refresh()
    } catch (e: Any) { setErr(e) } finally { setBusy(false) }
  }
  const restart = async () => { await post(`/api/p/${pid}/bosses/${id}`, { restart: true }); setEv(null); setSelectedRagExperiment(''); setData({ ...b, state: { ...b.state, phase: 0, hp: 100, mistakes: 0, log: [] } }) }
  const onRun = (r: Any) => { if (r.ok) send({ run_id: r.run_id }) }
  const intro = b.intro
  const retrievalBoss = id === 'retrieval'
  const phasePrefix = retrievalBoss && phase?.i18n_key ? `boss.retrieval.phases.${phase.i18n_key}` : ''
  const phaseText = (key: string, fallback: string) => {
    if (!phasePrefix) return fallback
    const lookup = `${phasePrefix}.${key}`
    const translated = t(lookup)
    return translated === lookup ? fallback : translated
  }
  const bossName = retrievalBoss ? t('boss.retrieval.name') : b.name
  const bossTaunt = retrievalBoss ? t('boss.retrieval.taunt') : b.taunt
  const bossLesson = retrievalBoss ? t('boss.retrieval.lesson') : b.lesson
  const phaseTitle = phase ? phaseText('title', phase.title) : ''
  const phasePrompt = phase?.q ? phaseText('prompt', phase.q.prompt) : ''
  const phaseOptions: string[] = phase?.q?.options?.map((option: string, index: number) => phaseText(`options.${index}`, option)) || []
  const phaseBrief = phase ? phaseText('brief', phase.brief || '') : ''
  const phaseCriteria: string[] = phase?.criteria?.map((criterion: string, index: number) => phaseText(`criteria.${index}`, criterion)) || []
  const checkLabel = (item: Any, index: number) => {
    if (retrievalBoss && ev?.phaseKey === 'real_personal') {
      const keys: Record<string, string> = {
        'Uses a document in your workspace': 'workspace',
        'Retrieves at least one chunk': 'retrieved',
        'Citation matches a retrieved chunk': 'citation',
        'Uses a documented retrieval strategy': 'strategy',
      }
      const key = keys[item.label]
      return key ? t(`boss.retrieval.checks.${key}`) : item.label
    }
    if (retrievalBoss && ev?.phaseKey === 'repair') return phaseText(`criteria.${index}`, item.label)
    if (item.label === 'Correct answer') return t('boss.correctAnswer')
    return item.label
  }
  const phaseExplanation = retrievalBoss && ev?.phaseKey ? t(`boss.retrieval.phases.${ev.phaseKey}.explanation`) : ev?.explanation

  return (
    <div className="stack">
      <div className="topbar"><Btn kind="ghost" small onClick={() => go('/bosses')}>← {t('boss.backToBosses')}</Btn><div className="spacer" />{st.defeated && <Pill kind="green">{t('boss.defeatedBefore')}</Pill>}<Btn small kind="ghost" onClick={restart}>↺ {t('boss.restart')}</Btn></div>
      <div className={`boss-hero ${hit ? 'hit' : ''} ${won ? 'won' : ''}`}>
        <div className="boss-icon big">{won ? '💥' : b.icon}</div>
        <div style={{ flex: 1 }}>
          <h1 style={{ margin: 0 }}>{bossName}</h1>
          <p style={{ fontStyle: 'italic', margin: '0.3rem 0 0.6rem' }}>“{bossTaunt}”</p>
          <Bar value={st.hp} max={100} color="var(--red)" label={`${t('boss.hp')} ${st.hp}/100`} />
          <div className="row" style={{ marginTop: 8 }}>{b.phases.map((p: Any, i: number) => <Pill key={i} kind={i < st.phase ? 'green' : i === st.phase ? 'amber' : ''}>{i + 1}. {retrievalBoss && p.i18n_key ? t(`boss.retrieval.phases.${p.i18n_key}.title`) : p.title}</Pill>)}<small className="muted">{t('boss.mistakes')}: {st.mistakes}</small></div>
        </div>
      </div>
      <ErrorBox error={err || (retrievalBoss ? ragExperiments.error : null)} />
      {ev && (
        <div className={ev.passed ? 'ok-box' : 'error-box'}>
          <b>{ev.passed ? `⚔️ ${t('boss.hit')} ${ev.damage} ${t('boss.damage')}` : `🛡️ ${t('boss.shrugs')}`}</b>
          {ev.checks?.length > 0 && <div style={{ marginTop: 4 }}>{ev.checks.map((c: Any, i: number) => <div key={i}>{c.passed ? '✅' : '❌'} {checkLabel(c, i)}{c.value !== undefined ? <span className="mono muted"> ({Array.isArray(c.value) ? c.value.join(', ') : String(c.value)})</span> : ''}</div>)}</div>}
          {phaseExplanation && phaseExplanation !== `boss.retrieval.phases.${ev.phaseKey}.explanation` && <Rich text={phaseExplanation} />}
        </div>
      )}
      {won ? (
        <Card title={`${bossName} · ${t('boss.defeated')}`} icon="🏆" className="glow">
          <p><b>{t('boss.lessonLabel')}:</b> {bossLesson}</p>
          <div className="row"><Btn onClick={() => go('/bosses')}>{t('boss.nextBoss')}</Btn><Btn kind="ghost" onClick={() => go('/')}>{t('nav.campusMap')}</Btn></div>
        </Card>
      ) : (
        <>
          {intro && st.phase === 0 && <Card title={retrievalBoss ? t('boss.retrieval.introTitle') : intro.title} icon="👁️"><IntroView intro={intro} /></Card>}
          <Card title={`${t('boss.phase')} ${st.phase + 1}: ${phaseTitle}`} icon="⚔️">
            {phase.kind === 'mcq' && (
              <div className="col">
                <Rich text={phasePrompt} />
                {phaseOptions.map((option: string, i: number) => <button key={i} className={`option ${choice === i ? 'sel' : ''}`} onClick={() => setChoice(i)}><span className="key">{i + 1}</span>{option}</button>)}
                <div><Btn kind="danger" disabled={choice === null || busy} onClick={() => send({ answer: choice })}>⚔️ {t('boss.strike')}</Btn></div>
              </div>
            )}
            {phase.kind === 'select' && (
              <div className="col">
                <p><b>{phase.prompt}</b></p>
                {phase.options.map((o: string, i: number) => <label key={i} className={`option ${sel.includes(i) ? 'sel' : ''}`}><input type="checkbox" checked={sel.includes(i)} onChange={() => setSel(sel.includes(i) ? sel.filter((x) => x !== i) : [...sel, i])} /> <span className="mono">{o}</span></label>)}
                <div><Btn kind="danger" disabled={sel.length === 0 || busy} onClick={() => send({ selected: sel.map((i) => phase.options[i]) })}>⚔️ Strike</Btn></div>
              </div>
            )}
            {!['mcq', 'select', 'rag_real'].includes(phase.kind) && (
              <div className="col">
                <div className="info-box"><b>{t('boss.brief')}:</b> {phaseBrief}</div>
                {phaseCriteria.length > 0 && <div className="row">{phaseCriteria.map((criterion: string) => <Pill key={criterion} kind="amber">{criterion}</Pill>)}</div>}
                <small className="muted">{t('boss.autoSubmit')}</small>
              </div>
            )}
            {phase.kind === 'rag_real' && (
              <div className="col">
                <div className="info-box"><b>{t('boss.brief')}:</b> {phaseBrief}</div>
                <a className="btn btn-ghost btn-sm" href="#/personal-rag">📚 {t('boss.openAdvancedRag')}</a>
                {phaseCriteria.length > 0 && <div className="row">{phaseCriteria.map((criterion: string) => <Pill key={criterion} kind="amber">{criterion}</Pill>)}</div>}
                <Select label={t('boss.selectExperiment')} value={selectedRagExperiment} onChange={setSelectedRagExperiment} options={[
                  { value: '', label: t('boss.selectExperiment') }, ...(ragExperiments.data || []).map((item: Any) => ({
                    value: item.id,
                    label: `#${item.id.slice(0, 8)} · ${item.config?.method} · ${item.metrics?.retrieved_chunks || 0} ${t('advancedRag.chunks')} · ${item.query.slice(0, 70)}`,
                  })),
                ]} />
                <small className="muted">{t('boss.retrieval.realPersonalNote')}</small>
                <Btn kind="danger" disabled={busy || !selectedRagExperiment} onClick={() => send({ experiment_id: selectedRagExperiment })}>⚔️ {t('boss.submitExperiment')}</Btn>
              </div>
            )}
          </Card>
          {phase.kind === 'ml' && <Workbench key={st.phase} dataset={phase.dataset} lockDataset context={`boss:${id}`} initial={intro?.kind === 'ml' ? { dataset: phase.dataset, features: intro.run.features_used, model: id === 'imbalance' ? 'logistic_regression' : id === 'leak' ? 'random_forest' : 'decision_tree', params: {}, preprocessing: {} } : undefined} onRun={onRun} />}
          {phase.kind === 'nn' && <NNLab key={st.phase} embedded initial={{ ...(intro?.run ? { dataset: phase.dataset, hidden: intro.run.sizes.slice(1, -1), activation: 'relu', optimizer: 'sgd', lr: 8, batch_size: 32, epochs: 60, dropout: 0, l2: 0, seed: 0 } : {}), dataset: phase.dataset }} context={`boss:${id}`} onRun={onRun} />}
          {phase.kind === 'rag' && <RagLab key={st.phase} initial={intro?.run?.config} fixed={phase.fixed} context={`boss:${id}`} onRun={onRun} />}
          {phase.kind === 'answers' && <RagLab key={st.phase} answers context={`boss:${id}`} onRun={onRun} />}
          {phase.kind === 'agent' && <AgentLab key={st.phase} context={`boss:${id}`} onRun={onRun} />}
        </>
      )}
      {st.log.length > 0 && <Card title={t('boss.battleLog')} icon="📜">{st.log.slice().reverse().map((l: Any, i: number) => <div key={i}><small>{l.text}</small></div>)}</Card>}
    </div>
  )
}
