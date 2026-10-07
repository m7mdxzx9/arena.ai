import { useMemo, useState } from 'react'
import { del, post, qs, upload, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { BarChart } from '../charts'
import { Btn, Card, ErrorBox, Loading, Pill, Select, Toggle, fmt, useApi, useGame } from '../ui'
import { ColumnChart } from './DataLab'
import { RunResult } from '../labs/Workbench'

function UploadPanel({ onUploaded }: { onUploaded: (dataset: Any) => void }) {
  const { t } = useI18n()
  const { pid } = useGame()
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const submit = async () => {
    if (!file) return
    setBusy(true); setError(null)
    try { onUploaded(await upload(`/api/p/${pid}/datasets`, file)) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  return (
    <Card title={t('datasets.uploadTitle')} icon="⬆️">
      <p className="muted"><small>{t('datasets.formats')}</small></p>
      <div className="upload-zone">
        <input aria-label={t('datasets.choose')} type="file" accept=".csv,.tsv,.json,.xlsx" onChange={(event) => setFile(event.target.files?.[0] || null)} />
        {file && <span><Ltr>{file.name}</Ltr> · {fmt(file.size / 1024, 1)} KB</span>}
        <Btn onClick={submit} disabled={!file || busy}>{busy ? t('datasets.uploading') : t('common.upload')}</Btn>
      </div>
      <ErrorBox error={error} />
    </Card>
  )
}

function Workspace({ dataset, onDeleted }: { dataset: Any; onDeleted: () => void }) {
  const { t } = useI18n()
  const { pid, meta, ov, reward } = useGame()
  const columnNames = dataset.columns.map((column: Any) => column.name) as string[]
  const [target, setTarget] = useState(() => columnNames[columnNames.length - 1] || '')
  const [features, setFeatures] = useState<string[]>(() => columnNames.filter((column) => column !== target))
  const [model, setModel] = useState('')
  const [scale, setScale] = useState('standard')
  const [dedupe, setDedupe] = useState(true)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [run, setRun] = useState<Any>(null)
  const [error, setError] = useState<string | null>(null)
  const detail = useApi<Any>(`/api/p/${pid}/datasets/${dataset.id}${qs({ target })}`, [dataset.id, target])
  const profile = detail.data?.profile
  const targetInfo = profile?.column_info.find((column: Any) => column.name === target)
  const task = target && targetInfo && (targetInfo.kind === 'categorical' || targetInfo.kind === 'binary' || targetInfo.unique <= 20) ? 'classification' : 'regression'
  const modelOptions = useMemo(() => Object.entries(meta.models)
    .filter(([, spec]: [string, Any]) => spec.task === task && !spec.label.startsWith('Baseline'))
    .map(([value, spec]: [string, Any]) => ({ value, label: spec.label })), [meta.models, task])

  const activeModel = modelOptions.some((option) => option.value === model) ? model : modelOptions[0]?.value || ''
  const changeTarget = (value: string) => {
    setTarget(value)
    setFeatures(columnNames.filter((column) => column !== value))
    setModel('')
    setRun(null)
  }

  const train = async () => {
    setBusy(true); setError(null); setRun(null)
    try {
      const config = {
        dataset: `user:${dataset.id}`, target, features, model: activeModel, params: {}, seed: 42, test_size: 0.2,
        preprocessing: { impute_numeric: 'median', impute_categorical: 'most_frequent', scaling: scale, encoding: 'onehot', dedupe },
      }
      const result = await post(`/api/p/${pid}/run/ml`, { config, context: 'personal_dataset', name: name || dataset.name })
      setRun(result)
      if (result.ok) reward(result)
    } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const remove = async () => {
    if (!window.confirm(t('datasets.deletionConfirm'))) return
    try { await del(`/api/p/${pid}/datasets/${dataset.id}`); onDeleted() } catch (e: Any) { setError(e.message) }
  }
  const toggleFeature = (column: string) => setFeatures((current) => current.includes(column) ? current.filter((item) => item !== column) : [...current, column])
  if (!profile) return detail.error ? <ErrorBox error={detail.error} /> : <Loading />
  return (
    <div className="stack">
      <Card title={<><Ltr>{dataset.name}</Ltr> <Pill>{dataset.source}</Pill></>} icon="🗂️" right={<Btn small kind="danger" onClick={remove}>{t('common.delete')}</Btn>}>
        <div className="metrics">
          <div className="metric"><div className="metric-l">{t('common.rows')}</div><div className="metric-v">{profile.rows}</div></div>
          <div className="metric"><div className="metric-l">{t('common.columns')}</div><div className="metric-v">{profile.columns}</div></div>
          <div className="metric"><div className="metric-l">{t('datasets.missing')}</div><div className="metric-v">{profile.total_missing}</div></div>
          <div className="metric"><div className="metric-l">{t('datasets.duplicates')}</div><div className="metric-v">{profile.duplicates}</div></div>
        </div>
        {dataset.warnings?.map((warning: string, index: number) => <div className="warn-box" key={index}>{warning}</div>)}
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr><th>{t('common.columns')}</th><th>dtype</th><th>kind</th><th>{t('datasets.missing')}</th><th>unique</th><th>distribution</th></tr></thead>
            <tbody>{profile.column_info.map((column: Any) => <tr key={column.name}>
              <td className="mono technical-ltr">{column.name}</td><td className="technical-ltr">{column.dtype}</td><td>{column.kind}</td><td>{column.missing}</td><td>{column.unique}</td><td style={{ minWidth: 160 }}><ColumnChart chart={column.chart} /></td>
            </tr>)}</tbody>
          </table>
        </div>
      </Card>
      <div className="grid g-side">
        <div className="col">
          <Card title={t('datasets.train')} icon="🧪">
            <Select label={t('datasets.chooseTarget')} value={target} onChange={changeTarget} options={dataset.columns.map((column: Any) => ({ value: column.name, label: column.name }))} />
            <div className="field-l">{t('datasets.chooseFeatures')} ({features.length})</div>
            <div className="chips">{dataset.columns.filter((column: Any) => column.name !== target).map((column: Any) => (
              <button key={column.name} className={`chip technical-ltr ${features.includes(column.name) ? 'on' : ''}`} onClick={() => toggleFeature(column.name)}>{column.name}</button>
            ))}</div>
            <div className="info-box">{t('datasets.task')}: <b>{t(task === 'classification' ? 'datasets.classification' : 'datasets.regression')}</b></div>
            <Select label={t('datasets.algorithm')} value={activeModel} onChange={setModel} options={modelOptions} />
            <Select label={t('datasets.scaling')} value={scale} onChange={setScale} options={['none', 'standard', 'minmax']} />
            <Toggle label={t('datasets.dedupe')} checked={dedupe} onChange={setDedupe} />
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder={t('datasets.runName')} />
            <Btn onClick={train} disabled={busy || !activeModel || !target || features.length === 0}>{busy ? t('torch.training') : `▶ ${t('datasets.trainEvaluate')}`}</Btn>
          </Card>
          {profile.target?.type === 'classes' && <Card title={t('datasets.targetBalance')} icon="⚖️"><BarChart items={profile.target.labels.map((label: string, index: number) => ({ label, value: profile.target.counts[index] }))} /></Card>}
        </div>
        <div className="col">
          {busy && <Loading text={t('torch.training')} />}
          <ErrorBox error={error} />
          {run && !run.ok && <Card className="danger"><ErrorBox error={run.error} /><p>{run.lesson}</p><p className="muted">{run.fix}</p></Card>}
          {run?.ok && <Card title={`Run #${run.run_id}`} icon="📊" right={<Pill kind="green">{t('datasets.saved')}</Pill>}><RunResult res={run.result} config={run.config} showCode={!!ov.player.settings.show_code} /></Card>}
          {!run && !busy && <Card><div className="empty">{t('datasets.trainPrompt')}</div></Card>}
        </div>
      </div>
    </div>
  )
}

export default function DatasetsPage() {
  const { t } = useI18n()
  const { pid } = useGame()
  const datasets = useApi<Any[]>(`/api/p/${pid}/datasets`)
  const [selected, setSelected] = useState<string | null>(null)
  const current = datasets.data?.find((dataset) => dataset.id === selected) || datasets.data?.[0]
  const uploaded = (dataset: Any) => { datasets.reload(); setSelected(dataset.id) }
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('datasets.kicker')}</div><h1>{t('datasets.title')}</h1></div></div>
      <p className="muted">{t('datasets.subtitle')}</p>
      <UploadPanel onUploaded={uploaded} />
      <ErrorBox error={datasets.error} />
      {datasets.loading && <Loading />}
      {datasets.data && datasets.data.length === 0 && <Card><div className="empty">{t('datasets.empty')}</div></Card>}
      {datasets.data && datasets.data.length > 0 && <>
        <div className="dataset-list">{datasets.data.map((dataset) => <button key={dataset.id} className={`dataset-tile ${current?.id === dataset.id ? 'active' : ''}`} onClick={() => setSelected(dataset.id)}><span>🗃️</span><b className="technical-ltr">{dataset.name}</b><small>{dataset.rows} {t('common.rows')} · {dataset.columns.length} {t('common.columns')}</small></button>)}</div>
        {current && <Workspace key={current.id} dataset={current} onDeleted={() => { setSelected(null); datasets.reload() }} />}
      </>}
    </div>
  )
}
