import { Fragment, useState } from 'react'
import { post, type Any } from '../api'
import { Ltr, useI18n } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Select, Slider, Toggle, fmt, pct, useGame } from '../ui'

export const RAG_DEFAULT: Any = { chunk_size: 60, overlap: 10, embedding: 'lsa', method: 'dense', top_k: 3, rerank: false, context_budget: 150 }
export const ANS_DEFAULT: Any = { mode: 'closed_book', chunk_size: 60, overlap: 10, embedding: 'lsa', method: 'hybrid', top_k: 3, rerank: false, context_budget: 150, abstain_threshold: 0, citations: true, temperature: 1 }

export function RagControls({ cfg, set, answers }: { cfg: Any; set: (k: string, v: Any) => void; answers?: boolean }) {
  const { t } = useI18n()
  const grounded = !answers || cfg.mode === 'grounded'
  return (
    <div className="col">
      {answers && <Select label={t('ragSimulation.answerMode')} value={cfg.mode} onChange={(v) => set('mode', v)} options={[{ value: 'closed_book', label: t('ragSimulation.closedBook') }, { value: 'grounded', label: t('ragSimulation.grounded') }]} />}
      {answers && cfg.mode === 'closed_book' && <Slider label={t('ragSimulation.temperature')} value={cfg.temperature} min={0.1} max={2} step={0.1} onChange={(v) => set('temperature', v)} />}
      {grounded && <>
        <div className="grid g2">
          <Slider label={t('ragSimulation.chunkSize')} value={cfg.chunk_size} min={10} max={300} step={10} onChange={(v) => set('chunk_size', v)} hint={t('ragSimulation.chunkSizeHint')} />
          <Slider label={t('ragSimulation.overlap')} value={cfg.overlap} min={0} max={Math.max(0, Math.min(50, cfg.chunk_size - 5))} step={5} onChange={(v) => set('overlap', v)} />
        </div>
        <div className="grid g2">
          <Select label={t('ragSimulation.embedding')} value={cfg.embedding} onChange={(v) => set('embedding', v)} options={[{ value: 'lsa', label: t('ragSimulation.lsa') }, { value: 'tfidf', label: t('ragSimulation.tfidf') }, { value: 'hash32', label: t('ragSimulation.hash32') }]} />
          <Select label={t('ragSimulation.retrieval')} value={cfg.method} onChange={(v) => set('method', v)} options={[{ value: 'dense', label: t('ragSimulation.dense') }, { value: 'bm25', label: t('ragSimulation.bm25') }, { value: 'hybrid', label: t('ragSimulation.hybrid') }]} />
        </div>
        <div className="grid g2">
          <Slider label={t('ragSimulation.topK')} value={cfg.top_k} min={1} max={10} onChange={(v) => set('top_k', v)} />
          <Slider label={t('ragSimulation.contextBudget')} value={cfg.context_budget} min={30} max={600} step={10} onChange={(v) => set('context_budget', v)} hint={t('ragSimulation.contextBudgetHint')} />
        </div>
        <Toggle label={t('ragSimulation.rerank')} checked={!!cfg.rerank} onChange={(v) => set('rerank', v)} />
        {answers && <>
          <Slider label={t('ragSimulation.abstainThreshold')} value={cfg.abstain_threshold} min={0} max={1} step={0.05} onChange={(v) => set('abstain_threshold', v)} hint={t('ragSimulation.abstainThresholdHint')} />
          <Toggle label={t('ragSimulation.citeSource')} checked={!!cfg.citations} onChange={(v) => set('citations', v)} />
        </>}
      </>}
    </div>
  )
}

export function RetrievalResult({ res }: { res: Any }) {
  const { t, language } = useI18n()
  const [open, setOpen] = useState<number | null>(null)
  const m = res.metrics
  const rk = Object.keys(m).find((k) => k.startsWith('recall@')) || 'recall@3'
  return (
    <div className="stack">
      <div className="metrics">
        <div className="metric" title={t('ragSimulation.metricRecallHint')}><div className="metric-l"><Ltr>{rk}</Ltr></div><div className="metric-v"><Ltr>{pct(m[rk])}</Ltr></div></div>
        <div className="metric" title={t('ragSimulation.metricMrrHint')}><div className="metric-l"><Ltr>MRR</Ltr></div><div className="metric-v"><Ltr>{fmt(m.mrr)}</Ltr></div></div>
        <div className="metric" title={t('ragSimulation.metricContextHint')}><div className="metric-l">{t('ragSimulation.answerInContext')}</div><div className="metric-v"><Ltr>{pct(m.context_hit_rate)}</Ltr></div></div>
        <div className="metric" title={t('ragSimulation.metricPrecisionHint')}><div className="metric-l">{t('ragSimulation.precision')}</div><div className="metric-v"><Ltr>{pct(m.doc_precision)}</Ltr></div></div>
        <div className="metric"><div className="metric-l">{t('ragSimulation.contextWords')}</div><div className="metric-v"><Ltr>{fmt(m.avg_context_words, 0)}</Ltr></div></div>
      </div>
      {res.diagnosis?.length > 0 && <div className="col">{res.diagnosis.map((a: string, i: number) => <div key={i} className="info-box">{localizeDiagnosis(t, a)}</div>)}</div>}
      <div className="tbl-wrap">
        <table className="tbl">
          <thead><tr><th>{t('ragSimulation.question')}</th><th>{t('ragSimulation.answer')}</th><th className="num">{t('ragSimulation.hitRank')}</th><th>{t('ragSimulation.inContext')}</th></tr></thead>
          <tbody>{res.questions.map((q: Any, i: number) => (
            <Fragment key={i}>
              <tr onClick={() => setOpen(open === i ? null : i)} style={{ cursor: 'pointer' }}>
                <td><bdi dir="auto">{localizeQuestion(t, q.q)}</bdi></td><td className={language === 'ar' ? '' : 'mono'}><bdi dir="auto">{localizeAnswer(t, q.answer)}</bdi></td><td className="num"><Ltr>{q.hit_rank ?? '✗'}</Ltr></td><td>{q.in_context ? <Pill kind="green">{t('common.yes')}</Pill> : <Pill kind="red">{t('common.no')}</Pill>}</td>
              </tr>
              {open === i && <tr><td colSpan={4}>{q.retrieved.map((r: Any, j: number) => (
                <div key={j} className={r.relevant ? 'ok-box' : 'info-box'} style={{ marginBottom: 4 }}>
                  <small className="mono">#{j + 1} {r.id} · {t('ragSimulation.score')} {fmt(r.score, 4)}{r.rerank != null ? ` · ${t('ragSimulation.rerankLabel')} ${fmt(r.rerank, 3)}` : ''}</small>
                  <div dir="auto" style={{ fontSize: '0.85rem' }}>{r.text}</div>
                </div>
              ))}</td></tr>}
            </Fragment>
          ))}</tbody>
        </table>
      </div>
      <small className="muted">{t('ragSimulation.clickQuestion')} {t('ragSimulation.colorLegend')}</small>
      <small className="muted">{t('ragSimulation.sourceLanguageNotice')}</small>
    </div>
  )
}

export function AnswersResult({ res }: { res: Any }) {
  const { t, language } = useI18n()
  const m = res.metrics
  return (
    <div className="stack">
      <div className="metrics">
        <div className="metric"><div className="metric-l">{t('ragSimulation.hallucinationRate')}</div><div className="metric-v" style={{ color: m.hallucination_rate > 0.2 ? 'var(--red)' : 'var(--green)' }}><Ltr>{pct(m.hallucination_rate)}</Ltr></div></div>
        <div className="metric"><div className="metric-l">{t('ragSimulation.answerAccuracy')}</div><div className="metric-v"><Ltr>{pct(m.answer_accuracy)}</Ltr></div></div>
        <div className="metric" title={t('ragSimulation.coverageHint')}><div className="metric-l">{t('ragSimulation.coverage')}</div><div className="metric-v"><Ltr>{pct(m.coverage)}</Ltr></div></div>
        <div className="metric" title={t('ragSimulation.abstainHint')}><div className="metric-l">{t('ragSimulation.correctAbstentions')}</div><div className="metric-v"><Ltr>{pct(m.correct_abstentions)}</Ltr></div></div>
        <div className="metric"><div className="metric-l">{t('ragSimulation.citationAccuracy')}</div><div className="metric-v"><Ltr>{m.citation_accuracy == null ? '—' : pct(m.citation_accuracy)}</Ltr></div></div>
      </div>
      <div className="tbl-wrap">
        <table className="tbl">
          <thead><tr><th>{t('ragSimulation.question')}</th><th>{t('ragSimulation.gold')}</th><th>{t('ragSimulation.systemAnswer')}</th><th>{t('ragSimulation.verdict')}</th></tr></thead>
          <tbody>{res.rows.map((r: Any, i: number) => (
            <tr key={i}>
              <td><bdi dir="auto">{localizeQuestion(t, r.q)}{!r.answerable && <Pill kind="violet">{t('ragSimulation.notInDocs')}</Pill>}</bdi></td>
              <td className={language === 'ar' ? '' : 'mono'}><small><bdi dir="auto">{r.gold == null ? `— (${t('ragSimulation.shouldAbstain')})` : localizeAnswer(t, r.gold)}</bdi></small></td>
              <td className={language === 'ar' ? '' : 'mono'}><small>{r.abstained ? <i>{t('ragSimulation.abstainAnswer')}</i> : <bdi dir="auto">{localizeAnswer(t, r.answer)}</bdi>}</small></td>
              <td>{r.correct ? <Pill kind="green">{t('ragSimulation.correct')}</Pill> : r.abstained ? <Pill kind={r.answerable ? 'amber' : 'green'}>{r.answerable ? t('ragSimulation.missed') : t('ragSimulation.abstained')}</Pill> : <Pill kind="red">{t('ragSimulation.hallucinated')}</Pill>}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <small className="muted">{t('ragSimulation.sourceLanguageNotice')}</small>
    </div>
  )
}

export default function RagLab({ answers, context, onRun, initial, fixed }: { answers?: boolean; context?: string; onRun?: (r: Any) => void; initial?: Any; fixed?: Any }) {
  const { t } = useI18n()
  const { pid, reward } = useGame()
  const [cfg, setCfg] = useState<Any>({ ...(answers ? ANS_DEFAULT : RAG_DEFAULT), ...(initial || {}), ...(fixed || {}) })
  const [busy, setBusy] = useState(false)
  const [out, setOut] = useState<Any>(null)
  const set = (k: string, v: Any) => { if (fixed && k in fixed) return; setCfg((c: Any) => ({ ...c, [k]: v })) }
  const run = async () => {
    setBusy(true)
    try {
      const r = await post(`/api/p/${pid}/run/${answers ? 'answers' : 'rag'}`, { config: cfg, context })
      setOut(r); reward(r); onRun?.(r)
    } catch (e: Any) { setOut({ ok: false, error: e }) } finally { setBusy(false) }
  }
  return (
    <div className="grid g-side">
      <Card title={answers ? t('ragSimulation.qaSystem') : t('ragSimulation.retrievalPipeline')} icon="⚙️">
        <div className="info-box" style={{ marginBottom: 8 }}>{t('ragSimulation.educationalNotice')}</div>
        {fixed && <div className="warn-box" style={{ marginBottom: 8 }}>{t('ragSimulation.fixedConstraint')}: <Ltr>{Object.entries(fixed).map(([k, v]) => `${k.replace('_', ' ')} = ${v}`).join(', ')}</Ltr> ({t('ragSimulation.cannotChange')}).</div>}
        <RagControls cfg={cfg} set={set} answers={answers} />
        <div className="row" style={{ marginTop: 10 }}><Btn onClick={run} disabled={busy}>{busy ? t('ragSimulation.evaluating') : answers ? `▶ ${t('ragSimulation.answerEval')}` : `▶ ${t('ragSimulation.evaluateRetrieval')}`}</Btn></div>
      </Card>
      <div className="col">
        {busy && <Loading text={t('ragSimulation.embeddingRetrievingScoring')} />}
        {out && !out.ok && <ErrorBox error={out.error} />}
        {out?.ok && <Card title={<>{t('ragSimulation.run')} <Ltr>#{out.run_id}</Ltr></>} icon="📊">{answers ? <AnswersResult res={out.result} /> : <RetrievalResult res={out.result} />}</Card>}
        {!out && !busy && <Card><div className="empty">{answers ? t('ragSimulation.emptyAnswers') : t('ragSimulation.emptyRetrieval')}</div><small className="muted">{t('ragSimulation.sourceLanguageNotice')}</small></Card>}
      </div>
    </div>
  )
}

function localizeQuestion(t: (key: string) => string, text: string): string {
  const lookup: Record<string, string> = {
    'When does the study building shut its doors Monday to Friday?': 'question0',
    'Is the library open all night when exams happen?': 'question1',
    "What's the most GPUs one student can book at once?": 'question2',
    'My training run went past 12 hours on the cluster, what happens?': 'question3',
    'When is temporary scratch space cleared?': 'question4',
    'Within how many days can I bring back headphones I bought?': 'question5',
    'Can I get my money back for an engraved mug?': 'question6',
    'How late can I show up to an exam and still be let in?': 'question7',
    'How much additional time do students with accessibility needs receive in exams?': 'question8',
    'How long per day can visitors stay on the guest wifi?': 'question9',
    'What special dish is served at lunch on Thursdays?': 'question10',
    'By what date must the Ada Lovelace Scholarship application be submitted?': 'question11',
    'What does ERR-4471 mean?': 'question12',
    'Explain error ERR-6021.': 'question13',
    'When do I need to keep noise down in the dorms?': 'question14',
    'How many pages can I print for free each term?': 'question15',
    'When can I visit Dr. Synapse to ask questions?': 'question16',
    'How frequently does the campus bus come?': 'question17',
    'How long does the quicker ethics review for anonymised public data take?': 'question18',
    'How many people do I need to start a new student club?': 'question19',
    'What is the Wi-Fi password for ForgeNet-Secure?': 'unanswerable0',
    'Who won the Robotics Society challenge last year?': 'unanswerable1',
    'How much does a car permit cost per year for staff?': 'unanswerable2',
    "What is the café's dinner menu on Saturdays?": 'unanswerable3',
    "What is the name of the library's head chef?": 'unanswerable4',
  }
  const key = lookup[text]
  return key ? t(`ragSimulation.${key}`) : text
}

function localizeAnswer(t: (key: string) => string, text: string): string {
  const lookup: Record<string, string> = {
    '22:00': 'answer0',
    '24 hours': 'answer1',
    '2 GPUs': 'answer2',
    'terminated automatically': 'answer3',
    'every Sunday at 03:00': 'answer4',
    '14 days': 'answer5',
    'not refundable': 'answer6',
    '20 minutes': 'answer7',
    '25 percent': 'answer8',
    '4 hours': 'answer9',
    'Backprop Burrito': 'answer10',
    'March 15': 'answer11',
    '500 MB': 'answer12',
    'personal data': 'answer13',
    '23:00 to 07:00': 'answer14',
    '300 free print credits': 'answer15',
    'Tuesdays from 14:00 to 16:00': 'answer16',
    'every 12 minutes': 'answer17',
    '5 working days': 'answer18',
    '8 founding members': 'answer19',
  }
  const key = lookup[text]
  return key ? t(`ragSimulation.${key}`) : text
}

function localizeDiagnosis(t: (key: string, params?: Record<string, string | number>) => string, text: string): string {
  const contextMatch = text.match(/Retrieval found the answer in ([0-9.]+%) of cases, but only ([0-9.]+%) survived into the context window/)
  if (contextMatch) return t('ragSimulation.diagnosisContext', { retrieval: contextMatch[1], context: contextMatch[2] })
  if (text.includes('hashing embedding squeezes')) return t('ragSimulation.diagnosisHash')
  if (text.includes('Large chunks blur')) return t('ragSimulation.diagnosisLargeChunks')
  if (text.includes('Very small chunks')) return t('ragSimulation.diagnosisSmallChunks')
  if (text.includes('Top-K = 1')) return t('ragSimulation.diagnosisTopK')
  if (text.includes('Pure dense retrieval')) return t('ragSimulation.diagnosisDense')
  if (text.includes('Healthy pipeline')) return t('ragSimulation.diagnosisHealthy')
  return text
}
