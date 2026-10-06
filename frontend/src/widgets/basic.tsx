// Interactive concept visuals, part 1. Client-side widgets compute real maths in the browser;
// data widgets fetch real computations from the backend. Nothing is faked.
import { useEffect, useMemo, useState } from 'react'
import { post, qs, type Any } from '../api'
import { BarChart, Heatmap, Histogram, LineChart, PALETTE, Scatter } from '../charts'
import { Btn, CodeBlock, ErrorBox, Loading, Select, Slider, fmt, highlight, useApi } from '../ui'

// ------------------------------------------------------------------ AI ⊃ ML ⊃ DL
export function AiVenn() {
  const [sel, setSel] = useState<string>('ml')
  const info: Record<string, [string, string]> = {
    ai: ['Artificial Intelligence', 'Any technique that makes computers do "intelligent" tasks — including hand-written rules (chess engines, expert systems).'],
    ml: ['Machine Learning', 'AI systems that learn patterns from data instead of following hand-written rules. Spam filters, recommendations.'],
    dl: ['Deep Learning', 'Machine learning with many-layered neural networks. Image recognition, speech, large language models.'],
    gen: ['Generative AI', 'Deep learning models that generate new text, images or audio (e.g. LLMs). A subset of deep learning.'],
  }
  const C = (id: string, cx: number, cy: number, r: number, color: string, label: string) => (
    <g onClick={(e) => { e.stopPropagation(); setSel(id) }} style={{ cursor: 'pointer' }}>
      <circle cx={cx} cy={cy} r={r} fill={color} fillOpacity={sel === id ? 0.35 : 0.13} stroke={color} strokeWidth={sel === id ? 3 : 1.5} />
      <text x={cx} y={cy - r + 20} textAnchor="middle" fill="#e2e8f0" fontSize={14} fontWeight={700}>{label}</text>
    </g>
  )
  return (
    <div className="grid g2">
      <svg viewBox="0 0 400 300" style={{ width: '100%' }}>
        {C('ai', 200, 155, 140, '#22d3ee', 'AI')}
        {C('ml', 200, 178, 105, '#a78bfa', 'Machine Learning')}
        {C('dl', 200, 200, 70, '#f472b6', 'Deep Learning')}
        {C('gen', 200, 222, 36, '#fbbf24', 'GenAI')}
      </svg>
      <div><div className="kicker">Click a circle</div><h3>{info[sel][0]}</h3><p>{info[sel][1]}</p><small className="muted">Each field sits inside the one around it: every deep-learning system is machine learning, but not every AI system learns.</small></div>
    </div>
  )
}

// ------------------------------------------------------------------ Python tracer (real execution in the sandbox)
export function LoopTracer({ initial, boxes }: { initial?: string; boxes?: boolean }) {
  const [code, setCode] = useState(initial || (boxes ? 'x = 5\ny = x + 2\nx = x * 3\nname = "Ada"\ngreeting = "Hi " + name' : 'total = 0\nfor i in range(1, 5):\n    total = total + i\n    print(i, total)'))
  const [trace, setTrace] = useState<Any>(null)
  const [step, setStep] = useState(0)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const run = async () => {
    setBusy(true); setErr(null)
    try {
      const t = await post('/api/code/trace', { code })
      setTrace(t); setStep(0)
      if (t.error) setErr(t.error)
    } catch (e: Any) { setErr(e.message) } finally { setBusy(false) }
  }
  const steps = trace?.steps || []
  const cur = steps[step]
  const prev = steps[step - 1]
  const lines = code.split('\n')
  return (
    <div className="grid g2">
      <div className="col">
        <textarea value={code} onChange={(e) => { setCode(e.target.value); setTrace(null) }} rows={7} className="mono" spellCheck={false} style={{ fontSize: '0.85rem' }} />
        <div className="row"><Btn small onClick={run} disabled={busy}>{busy ? 'Running…' : '▶ Run & trace (real Python)'}</Btn>
          {steps.length > 0 && <>
            <Btn small kind="ghost" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>◀</Btn>
            <span className="mono">step {step + 1}/{steps.length}</span>
            <Btn small kind="ghost" onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step >= steps.length - 1}>▶</Btn>
          </>}
        </div>
        <ErrorBox error={err} />
      </div>
      <div className="col">
        {cur ? (
          <>
            <div className="codeblock"><pre>{lines.map((l, i) => <div key={i} className={i + 1 === cur.line ? 'trace-line' : ''}>{String(i + 1).padStart(2)}  {highlight(l)}</div>)}</pre></div>
            <small className="muted">Highlighted line is about to run. Variables right now:</small>
            <div className="row">
              {Object.entries(cur.vars).length === 0 && <span className="muted">(none yet)</span>}
              {Object.entries(cur.vars).map(([k, v]) => (
                <div key={k + String(v) + step} className={`var-box ${prev && prev.vars[k] !== v ? 'flash' : ''}`}><div className="n">{k}</div><div className="v">{String(v)}</div></div>
              ))}
            </div>
            {trace.stdout && <div className="console">{trace.stdout}</div>}
          </>
        ) : <p className="muted">Edit the code, then run it. You'll step through it line by line and watch each variable box change — this is exactly what Python does.</p>}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ y = ax + b and friends
export function FunctionPlot() {
  const [kind, setKind] = useState('linear')
  const [a, setA] = useState(2)
  const [b, setB] = useState(1)
  const xs = Array.from({ length: 81 }, (_, i) => -4 + i * 0.1)
  const f = (x: number) => (kind === 'linear' ? a * x + b : kind === 'quadratic' ? a * x * x + b : a * Math.exp(0.5 * x) + b)
  const label = kind === 'linear' ? `f(x) = ${a}·x + ${b}` : kind === 'quadratic' ? `f(x) = ${a}·x² + ${b}` : `f(x) = ${a}·e^(x/2) + ${b}`
  return (
    <div className="grid g2">
      <LineChart series={[{ name: label, values: xs.map(f) }]} x={xs} xLabel="x" yLabel="f(x)" />
      <div className="col">
        <Select label="Function" value={kind} onChange={setKind} options={[{ value: 'linear', label: 'Linear' }, { value: 'quadratic', label: 'Quadratic' }, { value: 'exp', label: 'Exponential' }]} />
        <Slider label="a" value={a} min={-3} max={3} step={0.5} onChange={setA} />
        <Slider label="b" value={b} min={-5} max={5} step={0.5} onChange={setB} />
        <p className="muted"><small>A function maps each input x to exactly one output. In ML, a model <i>is</i> a function — training searches for good values of its parameters (like a and b).</small></p>
        <div className="mono">f(1) = {fmt(f(1))} · f(2) = {fmt(f(2))}</div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ vectors & dot product
export function VectorPlot() {
  const [a, setA] = useState<[number, number]>([3, 1])
  const [b, setB] = useState<[number, number]>([1, 2])
  const dot = a[0] * b[0] + a[1] * b[1]
  const na = Math.hypot(...a), nb = Math.hypot(...b)
  const cos = na && nb ? dot / (na * nb) : 0
  const S = 30, O = 150
  const P = (v: number[]) => [O + v[0] * S, O - v[1] * S]
  return (
    <div className="grid g2">
      <svg viewBox="0 0 300 300" style={{ width: '100%', maxWidth: 320, background: '#0a1022', borderRadius: 10 }}>
        {Array.from({ length: 11 }, (_, i) => <g key={i}><line x1={i * 30} x2={i * 30} y1={0} y2={300} stroke="#1b2643" /><line y1={i * 30} y2={i * 30} x1={0} x2={300} stroke="#1b2643" /></g>)}
        <line x1={0} x2={300} y1={O} y2={O} stroke="#475569" /><line y1={0} y2={300} x1={O} x2={O} stroke="#475569" />
        {[[a, PALETTE[0], 'a'], [b, PALETTE[1], 'b']].map(([v, c, n]) => {
          const [x, y] = P(v as number[])
          return <g key={n as string}><line x1={O} y1={O} x2={x} y2={y} stroke={c as string} strokeWidth={3} markerEnd="url(#arr)" /><text x={x + 6} y={y - 6} fill={c as string} fontWeight={700}>{n as string}</text></g>
        })}
        <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#e2e8f0" /></marker></defs>
      </svg>
      <div className="col">
        {(['a', 'b'] as const).map((n) => {
          const v = n === 'a' ? a : b, set = n === 'a' ? setA : setB
          return <div key={n} className="row"><b style={{ color: n === 'a' ? PALETTE[0] : PALETTE[1] }}>{n}</b>
            <Slider label="x" value={v[0]} min={-4} max={4} step={0.5} onChange={(x) => set([x, v[1]])} />
            <Slider label="y" value={v[1]} min={-4} max={4} step={0.5} onChange={(y) => set([v[0], y])} /></div>
        })}
        <div className="kv">
          <dt>a · b</dt><dd>{a[0]}×{b[0]} + {a[1]}×{b[1]} = {fmt(dot)}</dd>
          <dt>|a|, |b|</dt><dd>{fmt(na)}, {fmt(nb)}</dd>
          <dt>cosine similarity</dt><dd>{fmt(cos)}</dd>
        </div>
        <small className="muted">Dot product is large when vectors point the same way, zero when perpendicular, negative when opposite. Cosine similarity divides out the lengths — exactly how embedding search compares meanings.</small>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ activation functions
const ACT: Record<string, (z: number) => number> = { relu: (z) => Math.max(0, z), sigmoid: (z) => 1 / (1 + Math.exp(-z)), tanh: Math.tanh, linear: (z) => z }
export function ActivationPlot() {
  const [z, setZ] = useState(1)
  const xs = Array.from({ length: 81 }, (_, i) => -4 + i * 0.1)
  return (
    <div className="grid g2">
      <LineChart series={Object.entries(ACT).map(([k, f], i) => ({ name: k, values: xs.map(f), color: PALETTE[i] }))} x={xs} xLabel="z (weighted sum)" yLabel="output" markX={z} yDomain={[-1.5, 4]} />
      <div className="col">
        <Slider label="input z" value={z} min={-4} max={4} step={0.1} onChange={setZ} />
        <table className="tbl"><tbody>{Object.entries(ACT).map(([k, f]) => <tr key={k}><td>{k}({fmt(z, 2)})</td><td className="num">{fmt(f(z))}</td></tr>)}</tbody></table>
        <small className="muted">Without a non-linear activation, stacking layers just produces another straight line. ReLU is the default in hidden layers; sigmoid squashes to a probability.</small>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ single neuron
export function NeuronPlayground() {
  const [x, setX] = useState([1, 0.5])
  const [w, setW] = useState([0.8, -0.4])
  const [b, setB] = useState(0.1)
  const [act, setAct] = useState('sigmoid')
  const z = x[0] * w[0] + x[1] * w[1] + b
  const out = ACT[act](z)
  return (
    <div className="grid g2">
      <svg viewBox="0 0 360 200" style={{ width: '100%' }}>
        {[0, 1].map((i) => (
          <g key={i}>
            <circle cx={50} cy={60 + i * 80} r={22} fill="#0f172a" stroke={PALETTE[0]} strokeWidth={2} />
            <text x={50} y={65 + i * 80} textAnchor="middle" fill="#e2e8f0" fontSize={13}>x{i + 1}={x[i]}</text>
            <line x1={72} y1={60 + i * 80} x2={178} y2={100} stroke={w[i] >= 0 ? PALETTE[0] : PALETTE[1]} strokeWidth={1 + Math.abs(w[i]) * 4} />
            <text x={120} y={70 + i * 50} fill="#94a3b8" fontSize={12}>w{i + 1}={w[i]}</text>
          </g>
        ))}
        <circle cx={205} cy={100} r={30} fill="#1e1b4b" stroke={PALETTE[4]} strokeWidth={3} />
        <text x={205} y={96} textAnchor="middle" fill="#e2e8f0" fontSize={12}>Σ + b</text>
        <text x={205} y={112} textAnchor="middle" fill="#fbbf24" fontSize={11}>z={fmt(z, 2)}</text>
        <line x1={235} y1={100} x2={290} y2={100} stroke="#e2e8f0" strokeWidth={2} />
        <text x={262} y={90} textAnchor="middle" fill="#94a3b8" fontSize={11}>{act}</text>
        <circle cx={318} cy={100} r={24} fill="#052e16" stroke={PALETTE[2]} strokeWidth={2} />
        <text x={318} y={105} textAnchor="middle" fill="#e2e8f0" fontSize={13}>{fmt(out, 2)}</text>
      </svg>
      <div className="col">
        <div className="grid g2">
          <Slider label="x1" value={x[0]} min={-2} max={2} step={0.1} onChange={(v) => setX([v, x[1]])} />
          <Slider label="x2" value={x[1]} min={-2} max={2} step={0.1} onChange={(v) => setX([x[0], v])} />
          <Slider label="w1" value={w[0]} min={-2} max={2} step={0.1} onChange={(v) => setW([v, w[1]])} />
          <Slider label="w2" value={w[1]} min={-2} max={2} step={0.1} onChange={(v) => setW([w[0], v])} />
          <Slider label="bias b" value={b} min={-2} max={2} step={0.1} onChange={setB} />
          <Select label="activation" value={act} onChange={setAct} options={Object.keys(ACT)} />
        </div>
        <div className="mono" style={{ fontSize: '0.85rem' }}>z = {x[0]}×{w[0]} + {x[1]}×{w[1]} + {b} = {fmt(z, 3)}<br />output = {act}(z) = {fmt(out, 3)}</div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ gradient descent on a real loss curve
export function GdDescent() {
  const [lr, setLr] = useState(0.1)
  const [start, setStart] = useState(-3.5)
  const [steps, setSteps] = useState(15)
  const [fn, setFn] = useState('bowl')
  const L = (w: number) => (fn === 'bowl' ? (w - 1) ** 2 + 0.5 : 0.15 * w ** 4 - 0.8 * w ** 2 + 0.3 * w + 2)
  const dL = (w: number) => (fn === 'bowl' ? 2 * (w - 1) : 0.6 * w ** 3 - 1.6 * w + 0.3)
  const path = useMemo(() => {
    const p = [start]
    for (let i = 0; i < steps; i++) {
      const w = p[p.length - 1] - lr * dL(p[p.length - 1])
      if (!isFinite(w) || Math.abs(w) > 1e6) break
      p.push(w)
    }
    return p
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lr, start, steps, fn])
  const xs = Array.from({ length: 121 }, (_, i) => -4 + i * (8 / 120))
  const last = path[path.length - 1]
  const diverged = Math.abs(last) > 4 || path.length < steps + 1
  return (
    <div className="grid g2">
      <div>
        <Scatter x={xs} y={xs.map(L)} size={1.2} xLabel="weight w" yLabel="loss L(w)" height={240}
          extra={(sx, sy) => <>
            {path.slice(0, -1).map((w, i) => Math.abs(w) <= 4 && Math.abs(path[i + 1]) <= 4 && <line key={i} x1={sx(w)} y1={sy(L(w))} x2={sx(path[i + 1])} y2={sy(L(path[i + 1]))} stroke="#fbbf24" strokeWidth={1.5} />)}
            {path.map((w, i) => Math.abs(w) <= 4 && <circle key={`p${i}`} cx={sx(w)} cy={sy(L(w))} r={i === path.length - 1 ? 5 : 3} fill={i === 0 ? '#f472b6' : '#fbbf24'} />)}
          </>} />
      </div>
      <div className="col">
        <Select label="Loss landscape" value={fn} onChange={setFn} options={[{ value: 'bowl', label: 'Convex bowl (one minimum)' }, { value: 'bumpy', label: 'Bumpy (local + global minimum)' }]} />
        <Slider label="learning rate" value={lr} min={0.01} max={1.1} step={0.01} onChange={setLr} />
        <Slider label="start w" value={start} min={-3.8} max={3.8} step={0.1} onChange={setStart} />
        <Slider label="steps" value={steps} min={1} max={60} onChange={setSteps} />
        <div className="kv"><dt>final w</dt><dd>{fmt(last)}</dd><dt>final loss</dt><dd>{fmt(L(last))}</dd><dt>gradient there</dt><dd>{fmt(dL(last))}</dd></div>
        {diverged ? <div className="warn-box">The steps overshoot and fly off — learning rate too large for this curve.</div> : <small className="muted">Each step: w ← w − lr × dL/dw. Too small = slow crawl; too big = overshoot. On the bumpy curve, the start point decides which valley you end in.</small>}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ network diagram
export function NnDiagram({ sizes: initial, weights }: { sizes?: number[]; weights?: number[][][] }) {
  const [hidden, setHidden] = useState<number[]>(initial ? initial.slice(1, -1) : [4, 3])
  const sizes = initial && weights ? initial : [2, ...hidden, 2]
  const W = 520, H = 240
  const cols = sizes.length
  const pos = (l: number, i: number) => [40 + (l * (W - 80)) / (cols - 1), H / 2 + (i - (Math.min(sizes[l], 10) - 1) / 2) * Math.min(34, (H - 30) / Math.min(sizes[l], 10))]
  const nParams = sizes.slice(1).reduce((acc, s, i) => acc + s * sizes[i] + s, 0)
  return (
    <div className="col">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
        {sizes.slice(1).map((s, l) => Array.from({ length: Math.min(s, 10) }, (_, j) => Array.from({ length: Math.min(sizes[l], 10) }, (_, i) => {
          const [x1, y1] = pos(l, i), [x2, y2] = pos(l + 1, j)
          const wv = weights?.[l]?.[i]?.[j]
          return <line key={`${l}-${i}-${j}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={wv === undefined ? '#334155' : wv >= 0 ? PALETTE[0] : PALETTE[1]} strokeOpacity={wv === undefined ? 0.6 : Math.min(0.9, 0.15 + Math.abs(wv) / 2)} strokeWidth={wv === undefined ? 1 : 0.5 + Math.min(3, Math.abs(wv) * 1.5)} />
        })))}
        {sizes.map((s, l) => Array.from({ length: Math.min(s, 10) }, (_, i) => {
          const [x, y] = pos(l, i)
          return <circle key={`${l}-${i}`} cx={x} cy={y} r={9} fill="#0f172a" stroke={l === 0 ? PALETTE[0] : l === cols - 1 ? PALETTE[2] : PALETTE[4]} strokeWidth={2} />
        }))}
        {sizes.map((s, l) => <text key={l} x={pos(l, 0)[0]} y={H - 4} textAnchor="middle" fill="#94a3b8" fontSize={11}>{l === 0 ? 'input' : l === cols - 1 ? 'output' : `hidden ${l}`} ({s}{s > 10 ? ', 10 shown' : ''})</text>)}
      </svg>
      {!weights && (
        <div className="row">
          {hidden.map((h, i) => <Slider key={i} label={`hidden layer ${i + 1}`} value={h} min={1} max={10} onChange={(v) => setHidden(hidden.map((x, j) => (j === i ? v : x)))} />)}
          <Btn small kind="ghost" onClick={() => hidden.length < 4 && setHidden([...hidden, 3])}>+ layer</Btn>
          <Btn small kind="ghost" onClick={() => hidden.length > 1 && setHidden(hidden.slice(0, -1))}>− layer</Btn>
        </div>
      )}
      <small className="muted">Parameters (weights + biases): <b className="mono">{nParams}</b>. Each line is one weight; each non-input neuron also has a bias.{weights ? ' Colour = sign (cyan +, pink −), thickness = magnitude of the trained weights.' : ''}</small>
    </div>
  )
}

// ------------------------------------------------------------------ distributions (real samples drawn on the server)
export function DistributionPlot() {
  const [kind, setKind] = useState('normal')
  const { data } = useApi(`/api/widget/distribution${qs({ kind })}`)
  return (
    <div className="grid g2">
      <div>{data ? <Histogram counts={data.counts} edges={data.edges} marks={[{ x: data.mean, label: 'mean', color: '#fbbf24' }, { x: data.median, label: 'median', color: '#a3e635' }]} /> : <Loading />}</div>
      <div className="col">
        <Select label="Distribution (2,000 random samples)" value={kind} onChange={setKind} options={[{ value: 'normal', label: 'Normal (bell curve)' }, { value: 'right_skewed', label: 'Right-skewed (like incomes)' }, { value: 'uniform', label: 'Uniform' }, { value: 'bimodal', label: 'Bimodal (two groups)' }]} />
        {data && <div className="kv"><dt>mean</dt><dd>{fmt(data.mean)}</dd><dt>median</dt><dd>{fmt(data.median)}</dd><dt>std</dt><dd>{fmt(data.std)}</dd></div>}
        <small className="muted">In skewed data a few large values drag the mean away from the median — the median is the more "typical" value there.</small>
      </div>
    </div>
  )
}

export function HistogramWidget() {
  const [ds, setDs] = useState('house_prices')
  const [col, setCol] = useState('')
  const [bins, setBins] = useState(20)
  const { data, error } = useApi(`/api/widget/histogram${qs({ dataset: ds, column: col, bins })}`)
  return (
    <div className="grid g2">
      <div>{error ? <ErrorBox error={error} /> : data ? <Histogram counts={data.hist.counts} edges={data.hist.edges} marks={[{ x: data.mean, label: 'mean', color: '#fbbf24' }, { x: data.median, label: 'median', color: '#a3e635' }]} /> : <Loading />}</div>
      <div className="col">
        <Select label="Dataset" value={ds} onChange={(v) => { setDs(v); setCol('') }} options={['house_prices', 'student_success', 'medical', 'customer_churn', 'fraud']} />
        {data && <Select label="Column" value={data.column} onChange={setCol} options={data.columns} />}
        <Slider label="bins" value={bins} min={5} max={50} onChange={setBins} />
        <small className="muted">A histogram counts how many rows fall into each value range. Look for skew, multiple peaks, impossible values and outliers.</small>
      </div>
    </div>
  )
}

export function TablePeek() {
  const [ds, setDs] = useState('student_success')
  const { data } = useApi(`/api/widget/peek${qs({ dataset: ds })}`)
  return (
    <div className="col">
      <Select label="Dataset" value={ds} onChange={setDs} options={['student_success', 'house_prices', 'customer_churn', 'animals', 'data_chaos']} />
      {data ? (
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr>{data.columns.map((c: string) => <th key={c} style={c === data.target ? { color: 'var(--amber)' } : {}}>{c}{c === data.target ? ' 🎯' : ''}</th>)}</tr></thead>
            <tbody>{data.rows.map((r: Any[], i: number) => <tr key={i}>{r.map((v, j) => <td key={j} className={v === null ? 'muted' : ''}>{v === null ? 'NaN' : String(v)}</td>)}</tr>)}</tbody>
          </table>
          <small className="muted">{data.n_rows} rows × {data.n_cols} columns. Each <b>row</b> is one example; each <b>column</b> is a feature; 🎯 marks the <b>target</b> (label) a model would learn to predict.</small>
        </div>
      ) : <Loading />}
    </div>
  )
}

export function MissingMap() {
  const { data } = useApi('/api/widget/missing')
  if (!data) return <Loading />
  return (
    <div className="col">
      <Heatmap matrix={data.missing} rows={data.columns} showValues={false} cell={9} color={(t) => (t > 0.5 ? '#fb7185' : '#1e293b')} />
      <small className="muted">First 80 students (columns of this grid = rows of the data). Pink = missing value. Totals in the full dataset: {Object.entries(data.totals).filter(([, v]) => (v as number) > 0).map(([k, v]) => `${k}: ${v}`).join(', ')}.</small>
    </div>
  )
}

export function OneHot() {
  const { data } = useApi('/api/widget/onehot')
  if (!data) return <Loading />
  return (
    <div className="grid g2">
      <div><div className="field-l">Before: text categories</div>
        <table className="tbl"><thead><tr>{Object.keys(data.before[0]).map((k) => <th key={k}>{k}</th>)}</tr></thead>
          <tbody>{data.before.map((r: Any, i: number) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>)}</tbody></table></div>
      <div><div className="field-l">After one-hot encoding (pandas get_dummies — same idea as scikit-learn's OneHotEncoder)</div>
        <div className="tbl-wrap"><table className="tbl"><thead><tr>{data.after_columns.map((k: string) => <th key={k} style={{ fontSize: '0.65rem' }}>{k}</th>)}</tr></thead>
          <tbody>{data.after.map((r: number[], i: number) => <tr key={i}>{r.map((v, j) => <td key={j} className="num" style={{ color: v ? 'var(--cyan)' : 'var(--muted)' }}>{v}</td>)}</tr>)}</tbody></table></div></div>
    </div>
  )
}

export function ScalingDemo() {
  const { data } = useApi('/api/widget/scaling')
  const [scaled, setScaled] = useState(false)
  if (!data) return <Loading />
  const d = scaled ? data.scaled : data.raw
  return (
    <div className="grid g2">
      <Scatter x={d.x} y={d.y} xLabel={`${data.names[0]}${scaled ? ' (z-score)' : ''}`} yLabel={`${data.names[1]}${scaled ? ' (z-score)' : ''}`} height={240} />
      <div className="col">
        <Btn small onClick={() => setScaled(!scaled)}>{scaled ? 'Show raw values' : 'Apply StandardScaler'}</Btn>
        <table className="tbl"><thead><tr><th>feature</th><th className="num">mean</th><th className="num">std</th></tr></thead>
          <tbody>{data.names.map((n: string) => <tr key={n}><td>{n}</td><td className="num">{scaled ? '0' : fmt(data.raw_stats[n].mean)}</td><td className="num">{scaled ? '1' : fmt(data.raw_stats[n].std)}</td></tr>)}</tbody></table>
        <small className="muted">Same points, new units: z = (x − mean) / std. Distance-based models (k-NN, k-means) and gradient descent care a lot; trees don't.</small>
      </div>
    </div>
  )
}

export function ImbalanceBar() {
  const { data } = useApi('/api/widget/imbalance')
  if (!data) return <Loading />
  return (
    <div className="col">
      <BarChart items={[{ label: 'legit (0)', value: data.counts['0'], color: PALETTE[0] }, { label: 'fraud (1)', value: data.counts['1'], color: PALETTE[1] }]} />
      <div className="warn-box">A model that always answers "legit" is right <b>{(data.majority_share * 100).toFixed(1)}%</b> of the time — and catches zero fraud. On imbalanced data, accuracy alone is misleading.</div>
    </div>
  )
}

export function CodeWidget({ code }: { code: string }) {
  return <CodeBlock code={code} />
}

export function useNoop() { useEffect(() => {}, []) }
