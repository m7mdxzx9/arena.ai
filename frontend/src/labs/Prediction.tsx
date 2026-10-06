import { useState } from 'react'
import { post, type Any } from '../api'
import { BarChart, BoundaryPlot, ConfusionMatrix, LineChart, PALETTE } from '../charts'
import { Btn, ErrorBox, Loading, fmt, pct, useGame } from '../ui'

export function PredictionResult({ data }: { data: Any }) {
  if (data.kind === 'curves') {
    return (
      <div className="grid g2">
        <div>
          <div className="field-l">Validation loss per epoch (real training runs)</div>
          <LineChart series={data.runs.map((r: Any, i: number) => ({ name: r.label, values: r.history.val_loss, color: PALETTE[i] }))} x={data.runs[0].history.epoch} xLabel="epoch" yLabel="val loss" logY />
        </div>
        <div>
          <div className="field-l">Accuracy (solid = train, dashed = validation)</div>
          <LineChart series={data.runs.flatMap((r: Any, i: number) => [
            { name: `${r.label} train`, values: r.history.train_acc, color: PALETTE[i] },
            { name: `${r.label} val`, values: r.history.val_acc, color: PALETTE[i], dashed: true },
          ])} x={data.runs[0].history.epoch} xLabel="epoch" yDomain={[0, 1]} />
        </div>
        {data.runs.map((r: Any) => (
          <div key={r.label} className="metric"><div className="metric-l">{r.label}</div>
            <div className="metric-s">{r.diverged_at ? `diverged at epoch ${r.diverged_at}` : r.final ? `train acc ${pct(r.final.train_acc)} · val acc ${pct(r.final.val_acc)} · val loss ${fmt(r.final.val_loss)}` : 'no result'}</div>
          </div>
        ))}
      </div>
    )
  }
  if (data.kind === 'bars') {
    return (
      <BarChart items={data.runs.flatMap((r: Any) => [
        { label: `${r.label} · train`, value: r.train, color: '#64748b' },
        { label: `${r.label} · test`, value: r.test, color: PALETTE[0] },
      ])} max={1} format={(v) => v.toFixed(3)} />
    )
  }
  if (data.kind === 'table') {
    return (
      <div className="grid g2">
        <table className="tbl"><tbody>{data.rows.map((r: Any) => <tr key={r.metric}><td>{r.metric}</td><td className="num">{typeof r.value === 'number' ? fmt(r.value) : r.value}</td></tr>)}</tbody></table>
        {data.confusion && <ConfusionMatrix labels={data.confusion.labels} matrix={data.confusion.matrix} />}
      </div>
    )
  }
  if (data.kind === 'boundaries' || data.kind === 'nn_boundaries') {
    return (
      <div className="grid g2">
        {data.runs.map((r: Any) => (
          <div key={r.label}>
            <b>{r.label}</b>
            {r.boundary && <BoundaryPlot grid={r.boundary} points={r.points || r.boundary.points} height={240} />}
            <small className="muted">{r.final ? `train acc ${pct(r.final.train_acc)} · validation acc ${pct(r.final.val_acc)}` : `train acc ${pct(r.train)} · test acc ${pct(r.test)}`}</small>
          </div>
        ))}
      </div>
    )
  }
  if (data.kind === 'elbow') {
    return (
      <div className="grid g2">
        <LineChart series={[{ name: 'inertia', values: data.elbow.map((e: Any) => e.inertia) }]} x={data.elbow.map((e: Any) => e.k)} xLabel="k (number of clusters)" yLabel="inertia" />
        <table className="tbl"><tbody>{data.runs.map((r: Any) => <tr key={r.label}><td>{r.label}</td><td className="num">inertia {fmt(r.inertia, 1)}</td><td className="num">silhouette {fmt(r.silhouette)}</td></tr>)}</tbody></table>
      </div>
    )
  }
  return <pre className="console">{JSON.stringify(data, null, 1)}</pre>
}

/** Predict-before-running: the player commits to an answer, then the real experiment runs on the server. */
export function PredictionCard({ exp, onDone }: { exp: Any; onDone?: (res: Any) => void }) {
  const { pid, reward } = useGame()
  const [choice, setChoice] = useState<number | null>(null)
  const [res, setRes] = useState<Any>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const run = async () => {
    if (choice === null) return
    setBusy(true)
    setErr(null)
    try {
      const r = await post(`/api/p/${pid}/predict/${exp.id}`, { choice })
      setRes(r)
      reward(r)
      onDone?.(r)
    } catch (e: Any) {
      setErr(e.message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="predict-box">
      <div className="kicker" style={{ color: 'var(--violet)' }}>🔮 Predict before you run · {exp.title}</div>
      <p>{exp.setup}</p>
      <p><b>{exp.question}</b></p>
      {exp.options.map((o: string, i: number) => {
        const cls = res ? (i === res.correct_option ? 'right' : i === choice ? 'wrong' : '') : choice === i ? 'sel' : ''
        return <button key={i} className={`option ${cls}`} disabled={!!res} onClick={() => setChoice(i)}><span className="key">{String.fromCharCode(65 + i)}</span>{o}</button>
      })}
      {!res && <Btn onClick={run} disabled={choice === null || busy}>{busy ? 'Running the real experiment…' : 'Lock in prediction & run experiment'}</Btn>}
      {busy && <Loading text="Training real models on the server…" />}
      <ErrorBox error={err} />
      {res && (
        <div className="stack" style={{ marginTop: '0.8rem' }}>
          <div className={res.was_right ? 'ok-box' : 'warn-box'}>
            <b>{res.was_right ? '✅ Your prediction matched reality.' : '😮 Reality disagreed with your prediction — that is where learning happens.'}</b>
            <p style={{ margin: '0.4rem 0 0' }}>{res.explanation}</p>
          </div>
          <div><div className="field-l" style={{ marginBottom: 6 }}>What actually happened (computed just now):</div><PredictionResult data={res.data} /></div>
        </div>
      )}
    </div>
  )
}
