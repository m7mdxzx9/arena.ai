import { Fragment, useState } from 'react'
import { post, type Any } from '../api'
import { Btn, Card, ErrorBox, Loading, Pill, Select, Slider, Toggle, fmt, pct, useGame } from '../ui'

export const RAG_DEFAULT: Any = { chunk_size: 60, overlap: 10, embedding: 'lsa', method: 'dense', top_k: 3, rerank: false, context_budget: 150 }
export const ANS_DEFAULT: Any = { mode: 'closed_book', chunk_size: 60, overlap: 10, embedding: 'lsa', method: 'hybrid', top_k: 3, rerank: false, context_budget: 150, abstain_threshold: 0, citations: true, temperature: 1 }

export function RagControls({ cfg, set, answers }: { cfg: Any; set: (k: string, v: Any) => void; answers?: boolean }) {
  const grounded = !answers || cfg.mode === 'grounded'
  return (
    <div className="col">
      {answers && <Select label="Answering mode" value={cfg.mode} onChange={(v) => set('mode', v)} options={[{ value: 'closed_book', label: '🗣️ Closed book — the language model answers from memory' }, { value: 'grounded', label: '📚 Grounded — answer only from retrieved documents' }]} />}
      {answers && cfg.mode === 'closed_book' && <Slider label="temperature" value={cfg.temperature} min={0.1} max={2} step={0.1} onChange={(v) => set('temperature', v)} />}
      {grounded && <>
        <div className="grid g2">
          <Slider label="chunk size (words)" value={cfg.chunk_size} min={10} max={300} step={10} onChange={(v) => set('chunk_size', v)} hint="Documents are split into chunks of this many words" />
          <Slider label="overlap (words)" value={cfg.overlap} min={0} max={Math.max(0, Math.min(50, cfg.chunk_size - 5))} step={5} onChange={(v) => set('overlap', v)} />
        </div>
        <div className="grid g2">
          <Select label="Embedding" value={cfg.embedding} onChange={(v) => set('embedding', v)} options={[{ value: 'lsa', label: 'LSA (semantic, 64-d)' }, { value: 'tfidf', label: 'TF-IDF (exact words)' }, { value: 'hash32', label: 'Hashing 32-d (cheap, collisions)' }]} />
          <Select label="Retrieval" value={cfg.method} onChange={(v) => set('method', v)} options={[{ value: 'dense', label: 'Dense (embedding cosine)' }, { value: 'bm25', label: 'BM25 (keywords)' }, { value: 'hybrid', label: 'Hybrid (reciprocal rank fusion)' }]} />
        </div>
        <div className="grid g2">
          <Slider label="Top-K chunks" value={cfg.top_k} min={1} max={10} onChange={(v) => set('top_k', v)} />
          <Slider label="context budget (words)" value={cfg.context_budget} min={30} max={600} step={10} onChange={(v) => set('context_budget', v)} hint="How many words fit in the prompt" />
        </div>
        <Toggle label="Re-rank the top 10 candidates with a cross-check scorer" checked={!!cfg.rerank} onChange={(v) => set('rerank', v)} />
        {answers && <>
          <Slider label="abstain threshold" value={cfg.abstain_threshold} min={0} max={1} step={0.05} onChange={(v) => set('abstain_threshold', v)} hint="Say 'I don't know' when the best supporting sentence scores below this" />
          <Toggle label="Cite the source chunk" checked={!!cfg.citations} onChange={(v) => set('citations', v)} />
        </>}
      </>}
    </div>
  )
}

export function RetrievalResult({ res }: { res: Any }) {
  const [open, setOpen] = useState<number | null>(null)
  const m = res.metrics
  const rk = Object.keys(m).find((k) => k.startsWith('recall@')) || 'recall@3'
  return (
    <div className="stack">
      <div className="metrics">
        <div className="metric" title="Share of questions whose answer document is among the retrieved Top-K chunks"><div className="metric-l">{rk}</div><div className="metric-v">{pct(m[rk])}</div></div>
        <div className="metric" title="Mean reciprocal rank of the first relevant chunk"><div className="metric-l">MRR</div><div className="metric-v">{fmt(m.mrr)}</div></div>
        <div className="metric" title="Share of questions where the answer text actually made it into the (budgeted) context"><div className="metric-l">answer in context</div><div className="metric-v">{pct(m.context_hit_rate)}</div></div>
        <div className="metric" title="Share of retrieved chunks that come from the right document"><div className="metric-l">precision</div><div className="metric-v">{pct(m.doc_precision)}</div></div>
        <div className="metric"><div className="metric-l">context words</div><div className="metric-v">{fmt(m.avg_context_words, 0)}</div></div>
      </div>
      {res.diagnosis?.length > 0 && <div className="col">{res.diagnosis.map((a: string, i: number) => <div key={i} className="info-box">{a}</div>)}</div>}
      <div className="tbl-wrap">
        <table className="tbl">
          <thead><tr><th>question</th><th>answer</th><th className="num">hit rank</th><th>in context</th></tr></thead>
          <tbody>{res.questions.map((q: Any, i: number) => (
            <Fragment key={i}>
              <tr onClick={() => setOpen(open === i ? null : i)} style={{ cursor: 'pointer' }}>
                <td>{q.q}</td><td className="mono">{q.answer}</td><td className="num">{q.hit_rank ?? '✗'}</td><td>{q.in_context ? <Pill kind="green">yes</Pill> : <Pill kind="red">no</Pill>}</td>
              </tr>
              {open === i && <tr><td colSpan={4}>{q.retrieved.map((r: Any, j: number) => (
                <div key={j} className={r.relevant ? 'ok-box' : 'info-box'} style={{ marginBottom: 4 }}>
                  <small className="mono">#{j + 1} {r.id} · score {fmt(r.score, 4)}{r.rerank != null ? ` · rerank ${fmt(r.rerank, 3)}` : ''}</small>
                  <div style={{ fontSize: '0.85rem' }}>{r.text}</div>
                </div>
              ))}</td></tr>}
            </Fragment>
          ))}</tbody>
        </table>
      </div>
      <small className="muted">Click a question to see exactly which chunks were retrieved. Green = from the document that contains the answer.</small>
    </div>
  )
}

export function AnswersResult({ res }: { res: Any }) {
  const m = res.metrics
  return (
    <div className="stack">
      <div className="metrics">
        <div className="metric"><div className="metric-l">hallucination rate</div><div className="metric-v" style={{ color: m.hallucination_rate > 0.2 ? 'var(--red)' : 'var(--green)' }}>{pct(m.hallucination_rate)}</div></div>
        <div className="metric"><div className="metric-l">answer accuracy</div><div className="metric-v">{pct(m.answer_accuracy)}</div></div>
        <div className="metric" title="Share of answerable questions the system attempted"><div className="metric-l">coverage</div><div className="metric-v">{pct(m.coverage)}</div></div>
        <div className="metric" title="Unanswerable questions where it correctly said 'I don't know'"><div className="metric-l">correct abstentions</div><div className="metric-v">{pct(m.correct_abstentions)}</div></div>
        <div className="metric"><div className="metric-l">citation accuracy</div><div className="metric-v">{m.citation_accuracy == null ? '—' : pct(m.citation_accuracy)}</div></div>
      </div>
      <div className="tbl-wrap">
        <table className="tbl">
          <thead><tr><th>question</th><th>gold</th><th>system answer</th><th>verdict</th></tr></thead>
          <tbody>{res.rows.map((r: Any, i: number) => (
            <tr key={i}>
              <td>{r.q}{!r.answerable && <Pill kind="violet">not in docs</Pill>}</td>
              <td className="mono"><small>{r.gold ?? '— (should abstain)'}</small></td>
              <td><small>{r.abstained ? <i>I don't know — not in the documents.</i> : r.answer}</small></td>
              <td>{r.correct ? <Pill kind="green">correct</Pill> : r.abstained ? <Pill kind={r.answerable ? 'amber' : 'green'}>{r.answerable ? 'missed' : 'abstained ✓'}</Pill> : <Pill kind="red">hallucinated</Pill>}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  )
}

export default function RagLab({ answers, context, onRun, initial, fixed }: { answers?: boolean; context?: string; onRun?: (r: Any) => void; initial?: Any; fixed?: Any }) {
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
    } catch (e: Any) { setOut({ ok: false, error: e.message }) } finally { setBusy(false) }
  }
  return (
    <div className="grid g-side">
      <Card title={answers ? 'Question-answering system' : 'Retrieval pipeline'} icon="⚙️">
        {fixed && <div className="warn-box" style={{ marginBottom: 8 }}>Fixed constraint: {Object.entries(fixed).map(([k, v]) => `${k.replace('_', ' ')} = ${v}`).join(', ')} (you can't change this).</div>}
        <RagControls cfg={cfg} set={set} answers={answers} />
        <div className="row" style={{ marginTop: 10 }}><Btn onClick={run} disabled={busy}>{busy ? 'Evaluating…' : answers ? '▶ Answer the evaluation questions' : '▶ Evaluate retrieval'}</Btn></div>
      </Card>
      <div className="col">
        {busy && <Loading text="Embedding, retrieving and scoring…" />}
        {out && !out.ok && <ErrorBox error={out.error} />}
        {out?.ok && <Card title={`Run #${out.run_id}`} icon="📊">{answers ? <AnswersResult res={out.result} /> : <RetrievalResult res={out.result} />}</Card>}
        {!out && !busy && <Card><div className="empty">{answers ? 'Run the system on 20 answerable and 5 unanswerable questions about the campus documents. Every answer is checked against the gold answer.' : 'Evaluate how well your pipeline finds the right chunk for 20 paraphrased questions about campus documents.'}</div></Card>}
      </div>
    </div>
  )
}
