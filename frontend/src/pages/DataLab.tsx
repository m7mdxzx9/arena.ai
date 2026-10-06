import { Fragment, useEffect, useState } from 'react'
import { qs, type Any } from '../api'
import { BarChart, Heatmap, Histogram, Scatter, divergent } from '../charts'
import { Btn, Card, ErrorBox, Loading, Pill, Select, fmt, go, pct, useApi, useGame } from '../ui'

export function ColumnChart({ chart }: { chart: Any }) {
  if (!chart || chart.type === 'empty') return <small className="muted">no values</small>
  if (chart.type === 'numeric') return <Histogram counts={chart.counts} edges={chart.edges} height={110} />
  return <BarChart items={chart.labels.map((l: string, i: number) => ({ label: l, value: chart.counts[i] }))} height={16} />
}

export function DataProfile({ ds, compact }: { ds: string; compact?: boolean }) {
  const { pid, reward } = useGame()
  const { data, error } = useApi(`/api/p/${pid}/data/${ds}`, [ds])
  const [sel, setSel] = useState<string | null>(null)
  const [sx, setSx] = useState('')
  const [sy, setSy] = useState('')
  useEffect(() => { if (data?.achievements?.length) reward(data) }, [data, reward])
  const numeric: string[] = data ? data.profile.column_info.filter((c: Any) => c.stats).map((c: Any) => c.name) : []
  useEffect(() => { if (numeric.length >= 2) { setSx(numeric[0]); setSy(numeric[1]) } }, [data]) // eslint-disable-line react-hooks/exhaustive-deps
  const scatter = useApi(sx && sy ? `/api/data/${ds}/scatter${qs({ x: sx, y: sy, color: data?.dataset.target || undefined })}` : null, [sx, sy, ds])
  if (error) return <ErrorBox error={error} />
  if (!data) return <Loading text="Profiling dataset…" />
  const p = data.profile
  const col = p.column_info.find((c: Any) => c.name === sel) || p.column_info[0]
  return (
    <div className="stack">
      <div className="metrics">
        <div className="metric"><div className="metric-l">rows</div><div className="metric-v">{p.rows}</div></div>
        <div className="metric"><div className="metric-l">columns</div><div className="metric-v">{p.columns}</div></div>
        <div className="metric"><div className="metric-l">missing cells</div><div className="metric-v">{p.total_missing}</div></div>
        <div className="metric"><div className="metric-l">duplicate rows</div><div className="metric-v" style={{ color: p.duplicates ? 'var(--amber)' : undefined }}>{p.duplicates}</div></div>
        <div className="metric"><div className="metric-l">memory</div><div className="metric-v">{p.memory_kb} KB</div></div>
      </div>
      {p.warnings.length > 0 && <div className="warn-box"><b>Data quality warnings</b><ul style={{ margin: '0.3rem 0 0', paddingLeft: '1.2rem' }}>{p.warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul></div>}
      <div className={compact ? 'stack' : 'grid g2'}>
        <Card title="Columns" icon="🧾">
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>column</th><th>kind</th><th className="num">missing</th><th className="num">unique</th><th>issue</th></tr></thead>
              <tbody>
                {p.column_info.map((c: Any) => (
                  <tr key={c.name} onClick={() => setSel(c.name)} className={col.name === c.name ? 'sel' : ''} style={{ cursor: 'pointer' }}>
                    <td className="mono">{c.name}{c.is_target && <Pill kind="violet">target</Pill>}</td>
                    <td><small>{c.kind.replace(/_/g, ' ')}</small></td>
                    <td className="num" style={{ color: c.missing ? 'var(--amber)' : undefined }}>{c.missing ? `${c.missing} (${c.missing_pct}%)` : '0'}</td>
                    <td className="num">{c.unique}</td>
                    <td><small style={{ color: 'var(--amber)' }}>{c.issue || ''}</small></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <small className="muted">Click a row to inspect that column.</small>
        </Card>
        <Card title={<>Column: <span className="mono">{col.name}</span></>} icon="🔬">
          <ColumnChart chart={col.chart} />
          {col.stats && <div className="kv" style={{ marginTop: 8 }}>{Object.entries(col.stats).map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{fmt(v, 2)}</dd></Fragment>)}<dt>outliers (1.5×IQR)</dt><dd>{col.outliers}</dd></div>}
          <div className="field-l" style={{ marginTop: 8 }}>sample values</div>
          <div className="mono" style={{ fontSize: '0.8rem' }}>{col.sample.map((s: Any) => JSON.stringify(s)).join(', ')}</div>
          {col.stats?.skew !== undefined && Math.abs(col.stats.skew) > 1 && <div className="info-box" style={{ marginTop: 8 }}>Skewed (skew {fmt(col.stats.skew, 2)}): the mean is pulled by the long tail — the median ({fmt(col.stats['50%'], 2)}) is a more typical value.</div>}
        </Card>
      </div>
      <div className={compact ? 'stack' : 'grid g2'}>
        {data.profile.target && (
          <Card title="Target distribution" icon="🎯">
            {p.target.type === 'classes' ? <>
              <BarChart items={p.target.labels.map((l: string, i: number) => ({ label: l, value: p.target.counts[i] }))} />
              <p className="muted"><small>Majority class share: <b>{pct(p.target.majority_share)}</b> — a model that always predicts the majority gets this accuracy for free.{p.target.majority_share > 0.8 && ' ⚠️ Strong imbalance: accuracy will be misleading; look at recall, precision, F1.'}</small></p>
            </> : <Histogram counts={p.target.chart.counts} edges={p.target.chart.edges} height={130} />}
            {p.target.associations?.length > 0 && <>
              <div className="field-l">Strongest numeric associations with the target</div>
              <BarChart items={p.target.associations.map((a: Any) => ({ label: a.feature, value: a.corr }))} max={1} format={(v) => v.toFixed(2)} />
              <small className="muted">Correlation ≠ causation, and a suspiciously perfect association (≈1.0) usually means leakage.</small>
            </>}
          </Card>
        )}
        {p.correlations?.columns?.length > 1 && (
          <Card title="Correlations (numeric columns)" icon="🧮">
            <Heatmap matrix={p.correlations.matrix} rows={p.correlations.columns} cols={p.correlations.columns} color={divergent} domain={[-1, 1]} fmtv={(v) => v.toFixed(2)} cell={p.correlations.columns.length > 7 ? 30 : 42} />
            <small className="muted">Pearson correlation: +1 move together, −1 opposite, 0 no linear relation (there could still be a non-linear one!).</small>
          </Card>
        )}
      </div>
      {numeric.length >= 2 && (
        <Card title="Scatter explorer" icon="✳️">
          <div className="row"><Select label="x" value={sx} onChange={setSx} options={numeric} /><Select label="y" value={sy} onChange={setSy} options={numeric} /></div>
          {scatter.data && <><Scatter x={scatter.data.x} y={scatter.data.y} c={scatter.data.c || undefined} xLabel={sx} yLabel={sy} height={240} /><small className="muted">correlation r = {fmt(scatter.data.corr, 3)}; colour = target</small></>}
        </Card>
      )}
      <Card title="First rows" icon="📄">
        <div className="tbl-wrap">
          <table className="tbl"><thead><tr>{Object.keys(p.head[0] || {}).map((k) => <th key={k}>{k}</th>)}</tr></thead>
            <tbody>{p.head.map((r: Any, i: number) => <tr key={i}>{Object.values(r).map((v: Any, j) => <td key={j} style={{ color: v === null ? 'var(--amber)' : undefined }}>{v === null ? 'NaN' : typeof v === 'string' && v.length > 50 ? v.slice(0, 50) + '…' : String(v)}</td>)}</tr>)}</tbody></table>
        </div>
      </Card>
    </div>
  )
}

export default function DataLab({ initial }: { initial?: string }) {
  const { meta } = useGame()
  const ds = initial || 'student_success'
  const info = meta.datasets.find((d: Any) => d.id === ds)
  return (
    <div className="stack">
      <div className="topbar">
        <div><div className="kicker">Data District · Data Lens</div><h1 style={{ margin: 0 }}>Data Lab</h1></div>
        <div className="spacer" />
        <Select value={ds} onChange={(v) => go(`/data/${v}`)} options={meta.datasets.filter((d: Any) => !d.toy).map((d: Any) => ({ value: d.id, label: `${d.icon} ${d.title}` }))} />
        {info?.target && <Btn onClick={() => go(`/workbench/${ds}`)}>Model it in the Workbench →</Btn>}
      </div>
      {info && <div className="info-box"><b>{info.icon} {info.title}.</b> {info.story} <small className="muted">({info.notes})</small></div>}
      <DataProfile key={ds} ds={ds} />
    </div>
  )
}
