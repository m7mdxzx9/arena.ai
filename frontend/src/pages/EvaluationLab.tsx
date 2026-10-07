import { useState } from 'react'
import { del, post, put, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Select, useApi, useGame } from '../ui'

const Metric = ({ label, value }: { label: string; value: string | number }) => <div className="metric"><b>{value}</b><span>{label}</span></div>

const EXAMPLE_CASES = JSON.stringify([
  { id: 'definition', input: 'Define overfitting.', expected_behavior: 'Mentions poor generalization', reference_answer: ['generalization', 'unseen'], category: 'concept', difficulty: 'introductory', tags: ['ml'], evaluator: { type: 'contains', case_sensitive: false } },
  { id: 'numeric', input: 'Return the answer to 0.1 + 0.2.', reference_answer: 0.3, category: 'numeric', difficulty: 'introductory', tags: ['math'], evaluator: { type: 'numeric_tolerance', absolute: 1e-9 } },
], null, 2)

export default function EvaluationLabPage() {
  const { t } = useI18n()
  const { pid } = useGame()
  const datasets = useApi<Any[]>(`/api/p/${pid}/evaluation-datasets`)
  const [datasetId, setDatasetId] = useState('')
  const activeDatasetId = datasetId || datasets.data?.[0]?.id || ''
  const detail = useApi<Any>(activeDatasetId ? `/api/p/${pid}/evaluation-datasets/${activeDatasetId}` : null, [activeDatasetId])
  const [name, setName] = useState('Prompt Quality Baseline')
  const [casesText, setCasesText] = useState(EXAMPLE_CASES)
  const [outputsText, setOutputsText] = useState('{\n  "definition": "Overfitting memorizes training data and fails to generalize to unseen examples.",\n  "numeric": 0.30000000000000004\n}')
  const [result, setResult] = useState<Any>(null)
  const [editing, setEditing] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const parse = (text: string, expected: 'array' | 'object') => {
    const value = JSON.parse(text)
    if ((expected === 'array' && !Array.isArray(value)) || (expected === 'object' && (!value || typeof value !== 'object' || Array.isArray(value)))) throw new Error(`Expected a JSON ${expected}.`)
    return value
  }
  const create = async () => {
    setBusy(true); setError(null)
    try { const created = await post(`/api/p/${pid}/evaluation-datasets`, { name, cases: parse(casesText, 'array') }); datasets.reload(); setDatasetId(created.id) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const loadForEdit = () => {
    if (!detail.data) return
    setName(detail.data.name); setCasesText(JSON.stringify(detail.data.cases, null, 2)); setEditing(true)
  }
  const update = async () => {
    if (!activeDatasetId) return
    setBusy(true); setError(null)
    try { await put(`/api/p/${pid}/evaluation-datasets/${activeDatasetId}`, { name, cases: parse(casesText, 'array') }); detail.reload(); datasets.reload(); setEditing(false) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const remove = async () => {
    if (!activeDatasetId || !window.confirm(t('evaluationLab.deleteConfirm'))) return
    setBusy(true); setError(null)
    try { await del(`/api/p/${pid}/evaluation-datasets/${activeDatasetId}`); setDatasetId(''); setResult(null); setEditing(false); datasets.reload() } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const run = async () => {
    setBusy(true); setError(null); setResult(null)
    try { setResult(await post(`/api/p/${pid}/evaluation-datasets/${activeDatasetId}/run`, { outputs: parse(outputsText, 'object') })) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  return <div className="stack">
    <div className="topbar"><div><div className="kicker">{t('evaluationLab.kicker')}</div><h1>{t('evaluationLab.title')}</h1></div></div>
    <p className="muted">{t('evaluationLab.subtitle')}</p>
    <div className="grid g-side">
      <div className="col"><Card title={t('evaluationLab.createDataset')} icon="🧪">
        <label className="field"><span className="field-l">{t('evaluationLab.datasetName')}</span><input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label className="field"><span className="field-l">{t('evaluationLab.cases')}</span><textarea className="technical-ltr" dir="ltr" rows={22} value={casesText} onChange={(event) => setCasesText(event.target.value)} /></label>
        <Btn onClick={create} disabled={busy || !name.trim()}>{t('evaluationLab.createDataset')}</Btn>
      </Card></div>
      <div className="col"><Card title={t('evaluationLab.datasets')} icon="📏">
        {!datasets.data?.length && <p className="muted">{t('evaluationLab.noDatasets')}</p>}
        {datasets.data?.length ? <Select value={activeDatasetId} onChange={(value) => { setEditing(false); setDatasetId(value) }} options={datasets.data.map((dataset) => ({ value: dataset.id, label: `${dataset.name} · ${dataset.cases}` }))} /> : null}
        {detail.data && <div className="row"><Btn small kind="ghost" onClick={loadForEdit}>{t('evaluationLab.loadEdit')}</Btn>{editing && <><Btn small onClick={update} disabled={busy}>{t('common.save')}</Btn><Btn small kind="ghost" onClick={() => setEditing(false)}>{t('common.cancel')}</Btn></>}<Btn small kind="danger" onClick={remove} disabled={busy}>{t('common.delete')}</Btn></div>}
        {detail.data && <div className="table-wrap"><table><thead><tr><th>ID</th><th>{t('common.type')}</th><th>{t('common.status')}</th></tr></thead><tbody>{detail.data.cases.map((item: Any) => <tr key={item.id}><td><Ltr>{item.id}</Ltr></td><td><Pill kind="cyan"><Ltr>{item.evaluator.type}</Ltr></Pill></td><td>{item.category} · {item.difficulty}</td></tr>)}</tbody></table></div>}
        {detail.data && <><label className="field"><span className="field-l">{t('evaluationLab.outputs')}</span><textarea className="technical-ltr" dir="ltr" rows={12} value={outputsText} onChange={(event) => setOutputsText(event.target.value)} /></label><Btn onClick={run} disabled={busy}>{busy ? t('common.loading') : t('evaluationLab.run')}</Btn></>}
      </Card>
      <ErrorBox error={error || datasets.error || detail.error} />{busy && <Loading />}
      {result && <Card title={t('evaluationLab.title')} icon="✅" right={<Pill kind="green">{t('evaluationLab.deterministic')}</Pill>}>
        <div className="metrics"><Metric label={t('evaluationLab.score')} value={`${Math.round(result.summary.score * 100)}%`} /><Metric label={t('evaluationLab.passed')} value={result.summary.passed} /><Metric label={t('evaluationLab.failed')} value={result.summary.failed} /></div>
        <div className="stack">{result.results.map((item: Any) => <div className={item.passed ? 'info-box' : 'warn-box'} key={item.case_id}><div className="row between"><b><Ltr>{item.case_id}</Ltr></b><Pill kind={item.passed ? 'green' : 'red'}>{item.passed ? t('evaluationLab.passed') : t('evaluationLab.failed')}</Pill></div><div><Ltr>{item.detail}</Ltr></div></div>)}</div>
      </Card>}
      </div>
    </div>
  </div>
}
