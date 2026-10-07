import { useState } from 'react'
import { del, patch, post, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Select, useApi, useGame } from '../ui'

const EMPTY = { title: '', problem: '', dataset: '', method: '', metrics: '{}', interpretation: '', limitations: '', next_steps: '' }

function PortfolioForm({ initial, busy, submitLabel, onSubmit }: { initial: Any; busy: boolean; submitLabel: string; onSubmit: (payload: Any) => void }) {
  const { t } = useI18n()
  const [form, setForm] = useState<Record<string, string>>(() => ({
    title: initial.title || '', problem: initial.problem || '', dataset: initial.dataset || '', method: initial.method || '',
    metrics: typeof initial.metrics === 'string' ? initial.metrics : JSON.stringify(initial.metrics || {}, null, 2),
    interpretation: initial.interpretation || '', limitations: initial.limitations || '', next_steps: initial.next_steps || '',
  }))
  const field = (key: string, label: string, rows = 3, technical = false) => <label className="field"><span className="field-l">{label}</span><textarea dir={technical ? 'ltr' : undefined} className={technical ? 'technical-ltr' : ''} rows={rows} value={form[key]} onChange={(event) => setForm((value) => ({ ...value, [key]: event.target.value }))} /></label>
  const submit = () => { let metrics: Any = form.metrics; try { metrics = JSON.parse(form.metrics) } catch { /* narrative metrics are supported */ } onSubmit({ ...form, metrics }) }
  return <>
    <label className="field"><span className="field-l">{t('common.title')}</span><input value={form.title} onChange={(event) => setForm((value) => ({ ...value, title: event.target.value }))} /></label>
    {field('problem', t('portfolio.problem'))}{field('dataset', t('portfolio.dataset'), 2, true)}{field('method', t('portfolio.method'))}{field('metrics', t('portfolio.metrics'), 5, true)}{field('interpretation', t('portfolio.interpretation'))}{field('limitations', t('portfolio.limitations'))}{field('next_steps', t('portfolio.nextSteps'))}
    <Btn onClick={submit} disabled={busy}>{submitLabel}</Btn>
  </>
}

export default function PortfolioPage() {
  const { t } = useI18n()
  const { pid } = useGame()
  const runs = useApi<Any[]>(`/api/p/${pid}/runs`)
  const projects = useApi<Any[]>(`/api/p/${pid}/portfolio`)
  const [runId, setRunId] = useState('')
  const [projectId, setProjectId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const activeRunId = runId || String(runs.data?.[0]?.id || '')
  const selected = projects.data?.find((project) => project.id === projectId) || projects.data?.[0]
  const create = async (payload: Any) => {
    setBusy(true); setError(null)
    try { const project = await post(`/api/p/${pid}/portfolio`, { run_id: Number(activeRunId), ...payload }); projects.reload(); setProjectId(project.id) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const update = async (payload: Any) => {
    if (!selected) return
    setBusy(true); setError(null)
    try { await patch(`/api/p/${pid}/portfolio/${selected.id}`, payload); projects.reload() } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const remove = async () => {
    if (!selected || !window.confirm(t('portfolio.deleteConfirm'))) return
    setBusy(true); setError(null)
    try { await del(`/api/p/${pid}/portfolio/${selected.id}`); setProjectId(''); projects.reload() } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const download = (format: string) => selected && window.open(`/api/p/${pid}/portfolio/${selected.id}/export?format=${format}`, '_blank', 'noopener')
  return <div className="stack">
    <div className="topbar"><div><div className="kicker">{t('portfolio.kicker')}</div><h1>{t('portfolio.title')}</h1></div></div>
    <p className="muted">{t('portfolio.subtitle')}</p>
    <div className="grid g-side">
      <div className="col"><Card title={t('portfolio.create')} icon="🧾">
        {!runs.data ? <Loading /> : runs.data.length ? <><Select label={t('portfolio.selectRun')} value={activeRunId} onChange={setRunId} options={runs.data.map((run) => ({ value: String(run.id), label: `#${run.id} · ${run.name || run.kind}` }))} /><PortfolioForm initial={EMPTY} busy={busy || !activeRunId} submitLabel={t('portfolio.create')} onSubmit={create} /></> : <p className="muted">{t('portfolio.noRuns')}</p>}
      </Card></div>
      <div className="col"><Card title={t('portfolio.projects')} icon="💼">
        {!projects.data ? <Loading /> : !projects.data.length ? <p className="muted">{t('portfolio.noProjects')}</p> : <><Select value={selected?.id || ''} onChange={setProjectId} options={projects.data.map((project) => ({ value: project.id, label: project.title }))} />{selected && <div key={selected.id}>
          <div className="row"><Pill><Ltr>run #{selected.source_run_id}</Ltr></Pill><Pill kind="cyan"><Ltr>{selected.source_run_kind}</Ltr></Pill></div>
          <PortfolioForm initial={selected} busy={busy} submitLabel={t('portfolio.update')} onSubmit={update} />
          <div className="row"><Btn kind="ghost" onClick={() => download('markdown')}>{t('portfolio.exportMarkdown')}</Btn><Btn kind="ghost" onClick={() => download('html')}>{t('portfolio.exportHtml')}</Btn><Btn kind="ghost" onClick={() => download('json')}>{t('portfolio.exportJson')}</Btn><Btn kind="danger" onClick={remove} disabled={busy}>{t('common.delete')}</Btn></div>
        </div>}</>}
      </Card><ErrorBox error={error || projects.error || runs.error} /></div>
    </div>
  </div>
}
