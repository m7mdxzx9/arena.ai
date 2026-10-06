import { useState } from 'react'
import { get, post, type Any } from '../api'
import { RunResult } from '../labs/Workbench'
import { Btn, Card, ErrorBox, Loading, Modal, Pill, Tabs, fmt, go, useApi, useGame } from '../ui'

const KIND_LABEL: Record<string, string> = { ml: 'Workbench', nn: 'Neural net', rag: 'RAG retrieval', answers: 'RAG answers', agent: 'Agent security' }

function summaryLine(r: Any): string {
  const s = r.summary || {}
  if (r.kind === 'ml') {
    const t = s.test || {}
    if (s.task === 'classification') return `F1 ${fmt(t.f1)} · acc ${fmt(t.accuracy)}${s.cv ? ` · CV ${fmt(s.cv.mean)}±${fmt(s.cv.std)}` : ''}`
    if (s.task === 'regression') return `RMSE ${fmt(t.rmse, 1)} · R² ${fmt(t.r2)}`
    return `silhouette ${fmt(s.train?.silhouette)}`
  }
  if (r.kind === 'nn') return `val acc ${fmt(s.final?.val_acc)} · train acc ${fmt(s.final?.train_acc)}${s.diverged_at ? ` · diverged @${s.diverged_at}` : ''}`
  return Object.entries(s.metrics || {}).map(([k, v]) => `${k} ${fmt(v)}`).join(' · ')
}

export default function History() {
  const { pid, reward, toast } = useGame()
  const [kind, setKind] = useState('ml')
  const { data, error, reload } = useApi<Any[]>(`/api/p/${pid}/runs?kind=${kind}`, [kind])
  const [sel, setSel] = useState<number[]>([])
  const [cmp, setCmp] = useState<Any>(null)
  const [cmpErr, setCmpErr] = useState<string | null>(null)
  const [detail, setDetail] = useState<Any>(null)
  const [note, setNote] = useState('')

  const compare = async () => {
    setCmpErr(null)
    try { const r = await post(`/api/p/${pid}/compare`, { run_ids: sel }); setCmp(r); reward(r) } catch (e: Any) { setCmpErr(e.message) }
  }
  const open = async (id: number) => { const d = await get(`/api/p/${pid}/runs/${id}`); setDetail(d); setNote(d.notes || '') }
  const saveNote = async () => {
    const r = await post(`/api/p/${pid}/runs/${detail.id}`, { notes: note })
    reward(r); toast({ kind: 'info', text: 'Notes saved' }); reload()
  }
  const toggle = (id: number) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length >= 5 ? s : [...s, id]))

  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">Evaluation Chamber</div><h1 style={{ margin: 0 }}>Experiment History</h1></div></div>
      <p className="muted">Every experiment is stored with its dataset, features, target, preprocessing, algorithm, hyperparameters, seed, metrics, timestamp and your notes — the minimum needed to reproduce it.</p>
      <Tabs tabs={Object.entries(KIND_LABEL).map(([id, label]) => ({ id, label }))} value={kind} onChange={(k) => { setKind(k); setSel([]); setCmp(null) }} />
      <ErrorBox error={error} />
      {!data ? <Loading /> : data.length === 0 ? (
        <Card><div className="empty">No {KIND_LABEL[kind]} runs yet. <a href={kind === 'ml' ? '#/workbench' : kind === 'nn' ? '#/nn' : kind === 'agent' ? '#/agent' : '#/rag'}>Run your first experiment →</a></div></Card>
      ) : (
        <Card title={`${data.length} runs`} right={kind === 'ml' && <Btn small onClick={compare} disabled={sel.length < 2}>Compare {sel.length} selected</Btn>}>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr>{kind === 'ml' && <th />}<th>#</th><th>name</th><th>when</th><th>setup</th><th>result</th><th>notes</th><th /></tr></thead>
              <tbody>
                {data.map((r) => (
                  <tr key={r.id}>
                    {kind === 'ml' && <td><input type="checkbox" checked={sel.includes(r.id)} onChange={() => toggle(r.id)} /></td>}
                    <td className="mono">{r.id}</td>
                    <td>{r.name || <span className="muted">—</span>}{r.context && r.context !== 'workbench' && <Pill kind="violet">{r.context}</Pill>}</td>
                    <td><small>{new Date(r.ts * 1000).toLocaleString()}</small></td>
                    <td><small>{r.kind === 'ml' ? `${r.summary.model_label} on ${r.config.dataset} · ${r.summary.n_features} feats · seed ${r.config.seed}` : r.kind === 'nn' ? `${r.config.dataset} · [${(r.summary.hidden || []).join(', ')}] · ${r.config.optimizer || 'sgd'} lr ${r.config.lr}` : r.kind === 'agent' ? (r.config.defences || []).join(', ') || 'no defences' : `${r.config.embedding || ''} chunk ${r.config.chunk_size || ''} ${r.config.retrieval || ''}`}</small></td>
                    <td className="mono"><small>{summaryLine(r)}</small></td>
                    <td><small>{r.notes ? '📝 ' + r.notes.slice(0, 40) : ''}</small></td>
                    <td><Btn small kind="ghost" onClick={() => open(r.id)}>Open</Btn></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
      <ErrorBox error={cmpErr} />
      {cmp && (
        <Card title="Model comparison — beyond the leaderboard" icon="⚖️" className="glow">
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>run</th><th>model</th><th className="num">{cmp.task === 'classification' ? 'test F1' : cmp.task === 'regression' ? 'test RMSE' : 'silhouette'}</th><th className="num">train–test gap</th><th className="num">CV</th><th className="num">predict ms/1k</th><th className="num">size KB</th><th className="num">interpretability</th><th className="num">features</th></tr></thead>
              <tbody>{cmp.rows.map((r: Any) => (
                <tr key={r.id}><td>{r.name}</td><td>{r.model}</td><td className="num">{fmt(r.primary)}</td><td className="num" style={{ color: r.gap > 0.1 ? 'var(--red)' : undefined }}>{fmt(r.gap)}</td><td className="num">{r.cv ? `${fmt(r.cv.mean)}±${fmt(r.cv.std)}` : '—'}</td><td className="num">{fmt(r.predict_ms_per_1k, 2)}</td><td className="num">{fmt(r.model_size_kb, 1)}</td><td className="num">{'★'.repeat(r.interpretability || 0)}</td><td className="num">{r.n_features}</td></tr>
              ))}</tbody>
            </table>
          </div>
          <div className="col" style={{ marginTop: '0.8rem' }}>{cmp.insights.map((i: string, k: number) => <div key={k} className={i.startsWith('⚠') ? 'warn-box' : 'info-box'}>{i}</div>)}</div>
        </Card>
      )}
      {detail && (
        <Modal wide onClose={() => setDetail(null)}>
          <h2>Run #{detail.id} {detail.name && `· ${detail.name}`}</h2>
          <div className="grid g2">
            <div><div className="field-l">Configuration (reproduce with these exact values)</div><pre className="console" style={{ maxHeight: 260 }}>{JSON.stringify(detail.config, null, 2)}</pre></div>
            <div className="col">
              <div className="field-l">Notes — what did you learn? what would you try next?</div>
              <textarea rows={6} value={note} onChange={(e) => setNote(e.target.value)} />
              <div className="row"><Btn small onClick={saveNote}>Save notes</Btn>{detail.kind === 'ml' && <Btn small kind="ghost" onClick={() => go(`/workbench/${detail.config.dataset}`)}>Open dataset in Workbench</Btn>}</div>
            </div>
          </div>
          {detail.kind === 'ml' && <div style={{ marginTop: '1rem' }}><RunResult res={detail.result} config={detail.config} showCode /></div>}
        </Modal>
      )}
    </div>
  )
}
