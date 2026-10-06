// Interactive concept visuals, part 2: ML, evaluation, vision, language and RAG.
import { useMemo, useState } from 'react'
import { qs, type Any } from '../api'
import { BarChart, BoundaryPlot, ConfusionMatrix, Heatmap, LineChart, PALETTE, Scatter } from '../charts'
import { Btn, ErrorBox, Loading, Select, Slider, Toggle, fmt, pct, useApi } from '../ui'

// ------------------------------------------------------------------ regression line: drag it yourself, then compare with least squares
export function ScatterLine() {
  const { data } = useApi('/api/widget/linreg')
  const [slope, setSlope] = useState(1.5)
  const [icpt, setIcpt] = useState(80)
  const [showFit, setShowFit] = useState(false)
  if (!data) return <Loading />
  const mse = (s: number, b: number) => data.x.reduce((acc: number, x: number, i: number) => acc + (data.y[i] - (s * x + b)) ** 2, 0) / data.x.length
  const mine = mse(slope, icpt)
  return (
    <div className="grid g2">
      <Scatter x={data.x} y={data.y} xLabel={data.x_label} yLabel={data.y_label} line={showFit ? { slope: data.fit.slope, intercept: data.fit.intercept, color: '#a3e635' } : { slope, intercept: icpt }} residuals height={260} />
      <div className="col">
        <Slider label="slope (k€ per m²)" value={slope} min={0} max={5} step={0.05} onChange={(v) => { setSlope(v); setShowFit(false) }} />
        <Slider label="intercept (k€)" value={icpt} min={-100} max={250} step={1} onChange={(v) => { setIcpt(v); setShowFit(false) }} />
        <div className="kv"><dt>your line MSE</dt><dd>{fmt(mine, 1)}</dd><dt>least-squares MSE</dt><dd>{showFit ? fmt(data.fit.mse, 1) : 'hidden'}</dd></div>
        <Toggle label="Show the least-squares line (computed with NumPy)" checked={showFit} onChange={setShowFit} />
        {showFit && <small className="muted">Best fit: price ≈ {fmt(data.fit.slope)} × size + {fmt(data.fit.intercept, 1)}. Your MSE is {fmt(mine / data.fit.mse, 2)}× the minimum. Red lines are residuals — training minimises the average of their squares.</small>}
        {!showFit && <small className="muted">Move the line to make the red residuals as short as possible. Can you get close to the minimum?</small>}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ decision boundaries from real scikit-learn models
export function DecisionBoundaryWidget() {
  const [ds, setDs] = useState('moons')
  const [model, setModel] = useState('logistic_regression')
  const [depth, setDepth] = useState(4)
  const [k, setK] = useState(5)
  const params: Any = { dataset: ds, model }
  if (model === 'decision_tree' || model === 'random_forest') params.max_depth = depth
  if (model === 'knn') params.n_neighbors = k
  const { data, loading, error } = useApi(`/api/widget/boundary${qs(params)}`)
  return (
    <div className="grid g2">
      <div>{error ? <ErrorBox error={error} /> : data ? <BoundaryPlot grid={data.boundary} points={data.boundary.points} height={280} /> : <Loading />}</div>
      <div className="col">
        <Select label="Dataset" value={ds} onChange={setDs} options={['moons', 'circles', 'linear2d', 'xor']} />
        <Select label="Model" value={model} onChange={setModel} options={[{ value: 'logistic_regression', label: 'Logistic regression (linear)' }, { value: 'decision_tree', label: 'Decision tree' }, { value: 'random_forest', label: 'Random forest' }, { value: 'knn', label: 'k-nearest neighbours' }, { value: 'naive_bayes', label: 'Naive Bayes' }]} />
        {(model === 'decision_tree' || model === 'random_forest') && <Slider label="max_depth" value={depth} min={1} max={15} onChange={setDepth} />}
        {model === 'knn' && <Slider label="k (neighbours)" value={k} min={1} max={60} onChange={setK} />}
        {data && <div className="kv"><dt>train accuracy</dt><dd>{pct(data.train.accuracy)}</dd><dt>test accuracy</dt><dd>{pct(data.test.accuracy)}</dd></div>}
        {loading && <small className="muted">training…</small>}
        <small className="muted">Background colour is the trained model's real predicted probability at each point of a 60×60 grid. Linear models draw straight boundaries; trees draw boxes; k-NN follows the data (jagged when k is small).</small>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ train/test split
export function SplitViz() {
  const [ts, setTs] = useState(0.2)
  const [strat, setStrat] = useState(true)
  const [seed, setSeed] = useState(42)
  const { data } = useApi(`/api/widget/split${qs({ test_size: ts, stratify: strat, seed })}`)
  return (
    <div className="grid g2">
      <div>{data ? (
        <Scatter x={data.x} y={data.y} c={data.split.map((s: number, i: number) => `${s ? 'test' : 'train'} · class ${data.label[i]}`)} height={250} />
      ) : <Loading />}</div>
      <div className="col">
        <Slider label="test size" value={ts} min={0.1} max={0.5} step={0.05} onChange={setTs} fmt={(v) => `${Math.round(v * 100)}%`} />
        <Toggle label="Stratify by class" checked={strat} onChange={setStrat} />
        <Btn small kind="ghost" onClick={() => setSeed(seed + 1)}>🎲 New random split (seed {seed})</Btn>
        {data && <div className="kv"><dt>train / test rows</dt><dd>{data.n_train} / {data.n_test}</dd><dt>class-1 share in train</dt><dd>{pct(data.train_class_share)}</dd><dt>class-1 share in test</dt><dd>{pct(data.test_class_share)}</dd></div>}
        <small className="muted">The test set is locked away during training — it estimates performance on data the model has never seen. Stratifying keeps class proportions equal in both parts.</small>
      </div>
    </div>
  )
}

export function CvFolds() {
  const [folds, setFolds] = useState(5)
  const { data } = useApi(`/api/widget/cv${qs({ folds })}`)
  return (
    <div className="grid g2">
      <div className="col">
        {data ? data.folds.map((f: Any) => {
          const val = new Set(f.val_idx)
          return (
            <div key={f.fold} className="row" style={{ gap: 6 }}>
              <span className="mono" style={{ width: 52 }}>fold {f.fold}</span>
              <svg viewBox={`0 0 ${data.n} 10`} preserveAspectRatio="none" style={{ flex: 1, height: 16, borderRadius: 4 }}>
                {Array.from({ length: data.n }, (_, i) => <rect key={i} x={i} width={1.05} height={10} fill={val.has(i) ? '#f472b6' : '#164e63'} />)}
              </svg>
              <span className="mono" style={{ width: 52, textAlign: 'right' }}>{f.score.toFixed(3)}</span>
            </div>
          )
        }) : <Loading />}
        <div className="legend"><span><i style={{ background: '#164e63' }} />train</span><span><i style={{ background: '#f472b6' }} />validation</span></div>
      </div>
      <div className="col">
        <Slider label="number of folds (k)" value={folds} min={2} max={10} onChange={setFolds} />
        {data && <div className="kv"><dt>mean accuracy</dt><dd>{fmt(data.mean)}</dd><dt>std across folds</dt><dd>{fmt(data.std)}</dd></div>}
        <small className="muted">Each row trains a real model on the cyan part and scores it on the pink part. Every example is validated exactly once. Report mean ± std, not your luckiest fold.</small>
      </div>
    </div>
  )
}

export function OverfitCurve() {
  const [ds, setDs] = useState('student_success')
  const { data } = useApi(`/api/widget/overfit${qs({ dataset: ds })}`)
  return (
    <div className="grid g2">
      <div>{data ? <LineChart series={[{ name: 'train accuracy', values: data.rows.map((r: Any) => r.train) }, { name: 'test accuracy', values: data.rows.map((r: Any) => r.test), color: PALETTE[1] }]} x={data.rows.map((r: Any, i: number) => (typeof r.depth === 'number' ? r.depth : i + 1))} xLabel="tree max_depth (model complexity; last point = unlimited)" markX={data.best_depth} /> : <Loading />}</div>
      <div className="col">
        <Select label="Dataset" value={ds} onChange={setDs} options={['student_success', 'medical', 'customer_churn']} />
        {data && <p>Best test accuracy at depth <b>{data.best_depth}</b>. Beyond it, training accuracy keeps climbing while test accuracy stalls or falls — the model starts memorising noise.</p>}
        <small className="muted">16 real decision trees, one per depth. Left of the sweet spot: underfitting (both low). Right: overfitting (big gap).</small>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ threshold → confusion matrix (real model probabilities)
export function CmThreshold() {
  const { data } = useApi('/api/widget/threshold')
  const [t, setT] = useState(0.5)
  const [cell, setCell] = useState<[number, number] | null>(null)
  const stats = useMemo(() => {
    if (!data) return null
    let tp = 0, fp = 0, fn = 0, tn = 0
    data.y_true.forEach((y: number, i: number) => {
      const p = data.p[i] >= t ? 1 : 0
      if (y && p) tp++; else if (!y && p) fp++; else if (y && !p) fn++; else tn++
    })
    const prec = tp + fp ? tp / (tp + fp) : 0, rec = tp + fn ? tp / (tp + fn) : 0
    return { tp, fp, fn, tn, acc: (tp + tn) / data.y_true.length, prec, rec, f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0 }
  }, [data, t])
  if (!data || !stats) return <Loading />
  const explain: Record<string, string> = {
    '0,0': 'True negatives: legit transactions correctly left alone.',
    '0,1': 'False positives: legit customers wrongly flagged (annoyed customers, support calls).',
    '1,0': 'False negatives: fraud the model MISSED — usually the most expensive error here.',
    '1,1': 'True positives: fraud correctly caught.',
  }
  return (
    <div className="grid g2">
      <div className="col">
        <ConfusionMatrix labels={['legit', 'fraud']} matrix={[[stats.tn, stats.fp], [stats.fn, stats.tp]]} onCell={(r, c) => setCell([r, c])} selected={cell} />
        {cell && <div className="info-box">{explain[cell.join(',')]}</div>}
        {!cell && <small className="muted">Click a cell to see what it means.</small>}
      </div>
      <div className="col">
        <Slider label="decision threshold" value={t} min={0.01} max={0.99} step={0.01} onChange={setT} />
        <div className="metrics">
          <div className="metric"><div className="metric-l">accuracy</div><div className="metric-v">{pct(stats.acc)}</div></div>
          <div className="metric"><div className="metric-l">precision</div><div className="metric-v">{pct(stats.prec)}</div></div>
          <div className="metric"><div className="metric-l">recall</div><div className="metric-v">{pct(stats.rec)}</div></div>
          <div className="metric"><div className="metric-l">F1</div><div className="metric-v">{fmt(stats.f1)}</div></div>
        </div>
        <small className="muted">{data.model}, {data.y_true.length} test transactions ({pct(data.positive_share)} fraud). Lower the threshold → catch more fraud (recall ↑) but more false alarms (precision ↓). Accuracy barely moves — it's dominated by the majority class.</small>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ k-means, step by step
export function KmeansSteps() {
  const [k, setK] = useState(3)
  const [seed, setSeed] = useState(0)
  const [step, setStep] = useState(0)
  const { data } = useApi(`/api/widget/kmeans${qs({ k, seed })}`, [])
  if (!data) return <Loading />
  const s = data.steps[Math.min(step, data.steps.length - 1)]
  const xs = data.points.map((p: number[]) => p[0]), ys = data.points.map((p: number[]) => p[1])
  return (
    <div className="grid g2">
      <Scatter x={xs} y={ys} c={s.assign} height={260} extra={(sx, sy) => s.centroids.map((c: number[], i: number) => (
        <g key={i}><circle cx={sx(c[0])} cy={sy(c[1])} r={9} fill="none" stroke="#fff" strokeWidth={3} /><circle cx={sx(c[0])} cy={sy(c[1])} r={5} fill={PALETTE[i % PALETTE.length]} /></g>
      ))} />
      <div className="col">
        <Slider label="k" value={k} min={1} max={8} onChange={(v) => { setK(v); setStep(0) }} />
        <div className="row">
          <Btn small kind="ghost" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>◀</Btn>
          <span className="mono">iteration {s.iter} / {data.steps.length - 1}</span>
          <Btn small onClick={() => setStep(Math.min(data.steps.length - 1, step + 1))} disabled={step >= data.steps.length - 1}>Next step ▶</Btn>
          <Btn small kind="ghost" onClick={() => { setSeed(seed + 1); setStep(0) }}>🎲 new start</Btn>
        </div>
        <div className="kv"><dt>inertia</dt><dd>{fmt(s.inertia, 1)}</dd>{step >= data.steps.length - 1 && <><dt>scikit-learn check</dt><dd>{fmt(data.sklearn_inertia_check, 1)}</dd></>}</div>
        <small className="muted">Each step: (1) assign every point to its nearest centroid (white rings), (2) move each centroid to the mean of its points. Repeat until nothing changes. Different starts can end in different solutions.</small>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ convolution on a real 8×8 digit
export function ConvFilter() {
  const [idx, setIdx] = useState(0)
  const [kernel, setKernel] = useState('vertical_edge')
  const { data } = useApi(`/api/widget/conv${qs({ index: idx, kernel })}`)
  if (!data) return <Loading />
  const grid = (m: number[][], title: string, color: (t: number, v: number) => string) => (
    <div><div className="field-l">{title}</div><Heatmap matrix={m} showValues={false} cell={18} color={color} /></div>
  )
  return (
    <div className="col">
      <div className="row">
        <Select label="Kernel" value={kernel} onChange={setKernel} options={data.kernels} />
        <Btn small kind="ghost" onClick={() => setIdx(idx + 1)}>Next digit (this is a {data.label})</Btn>
      </div>
      <div className="grid g4">
        {grid(data.image, 'input digit (8×8 pixels)', (t) => `rgba(226,232,240,${t})`)}
        <div><div className="field-l">kernel (3×3 weights)</div><Heatmap matrix={data.kernel} cell={30} color={(t, v) => (v >= 0 ? `rgba(34,211,238,${0.15 + 0.7 * t})` : `rgba(244,114,182,${0.15 + 0.7 * (1 - t)})`)} fmtv={(v) => String(Math.round(v * 100) / 100)} /></div>
        {grid(data.feature_map, 'feature map (convolution)', (t) => `rgba(167,139,250,${t})`)}
        {grid(data.pooled, '2×2 max-pool of ReLU', (t) => `rgba(163,230,53,${t})`)}
      </div>
      <small className="muted">The kernel slides over the image; each output pixel = sum of (kernel × pixels underneath). An edge kernel lights up where brightness changes. CNNs <i>learn</i> their kernel values during training.</small>
    </div>
  )
}

// ------------------------------------------------------------------ language
export function TokenizerWidget() {
  const [text, setText] = useState('The unbelievable tokenizer reads retrieval-augmented generation.')
  const [merges, setMerges] = useState(60)
  const [q, setQ] = useState(text)
  const { data } = useApi(`/api/widget/tokenizer${qs({ text: q, merges })}`)
  return (
    <div className="col">
      <div className="row"><input style={{ flex: 1 }} value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && setQ(text)} /><Btn small onClick={() => setQ(text)}>Tokenize</Btn></div>
      <Slider label="BPE merges learned from the campus corpus" value={merges} min={0} max={300} step={10} onChange={setMerges} />
      {data ? <>
        <div className="tokens">{data.bpe_tokens.map((t: string, i: number) => <span key={i} className="tok" style={{ background: `${PALETTE[i % PALETTE.length]}33`, border: `1px solid ${PALETTE[i % PALETTE.length]}88` }}>{t.replace('</w>', '␣')}</span>)}</div>
        <div className="kv"><dt>characters</dt><dd>{data.char_tokens}</dd><dt>words</dt><dd>{data.word_tokens}</dd><dt>BPE tokens</dt><dd>{data.bpe_tokens.length}</dd><dt>token ids</dt><dd>{data.ids.slice(0, 20).join(' ')}{data.ids.length > 20 ? ' …' : ''}</dd></div>
        <small className="muted">Byte-Pair Encoding starts from characters and repeatedly merges the most frequent adjacent pair (first merges: {data.first_merges.slice(0, 6).map((m: Any) => m.pair.join('+')).join(', ')}). More merges → fewer, longer tokens. Rare words split into pieces; ␣ marks a word end.</small>
      </> : <Loading />}
    </div>
  )
}

export function AttentionHeatmap() {
  const [s, setS] = useState('the library closes at midnight during exam week so students can study')
  const [q, setQ] = useState(s)
  const [row, setRow] = useState<number | null>(null)
  const { data } = useApi(`/api/widget/attention${qs({ sentence: q })}`)
  return (
    <div className="col">
      <div className="row"><input style={{ flex: 1 }} value={s} onChange={(e) => setS(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && setQ(s)} /><Btn small onClick={() => setQ(s)}>Compute attention</Btn></div>
      {data ? <div className="grid g2">
        <Heatmap matrix={data.weights} rows={data.tokens} cols={data.tokens} cell={30} domain={[0, 1]} fmtv={(v) => (v >= 0.1 ? v.toFixed(1) : '')} onCell={(r) => setRow(r)} />
        <div className="col">
          {row !== null ? <>
            <b>“{data.tokens[row]}” attends to:</b>
            <BarChart items={data.tokens.map((t: string, j: number) => ({ label: t, value: data.weights[row][j] })).sort((a: Any, b: Any) => b.value - a.value).slice(0, 6)} max={1} format={(v) => v.toFixed(2)} />
          </> : <small className="muted">Click a row to see where that word looks.</small>}
          <small className="muted">Each row is a softmax over (query · key) similarities, so it sums to 1. {data.note}</small>
        </div>
      </div> : <Loading />}
    </div>
  )
}

export function NgramLm() {
  const [word, setWord] = useState('the')
  const [temp, setTemp] = useState(1)
  const [seed, setSeed] = useState(0)
  const { data } = useApi(`/api/widget/lm${qs({ word, temperature: temp, seed })}`)
  return (
    <div className="grid g2">
      <div className="col">
        <div className="row"><input value={word} onChange={(e) => setWord(e.target.value.split(' ').pop() || '')} style={{ width: 140 }} /><Btn small kind="ghost" onClick={() => setSeed(seed + 1)}>🎲 sample again</Btn></div>
        <Slider label="temperature" value={temp} min={0.1} max={2.5} step={0.1} onChange={setTemp} />
        {data && <>
          <div className="field-l">Next-word probabilities after “{data.word}”</div>
          {data.next.length ? <BarChart items={data.next.map((n: Any) => ({ label: n.token, value: n.p }))} format={(v) => v.toFixed(3)} /> : <p className="muted">Never seen this word — the model has nothing to say.</p>}
        </>}
      </div>
      <div className="col">
        <div className="field-l">Generated continuations</div>
        {data?.samples.map((s: string, i: number) => <div key={i} className="console" style={{ maxHeight: 'none' }}>{s}</div>)}
        <small className="muted">{data?.note} Low temperature → repetitive and safe; high → creative and incoherent. It generates fluent-looking text with no notion of truth — the root of hallucination.</small>
      </div>
    </div>
  )
}

export function EmbeddingMap() {
  const [emb, setEmb] = useState('lsa')
  const [query, setQuery] = useState('when does the library close')
  const [q, setQ] = useState(query)
  const { data } = useApi(`/api/rag/map${qs({ embedding: emb, q })}`)
  const docs = data ? Array.from(new Set(data.points.map((p: Any) => p.title))) as string[] : []
  return (
    <div className="col">
      <div className="row">
        <Select value={emb} onChange={setEmb} options={[{ value: 'lsa', label: 'LSA embeddings (64-d)' }, { value: 'tfidf', label: 'TF-IDF vectors' }, { value: 'hash32', label: 'Hashing (32-d, collisions)' }]} />
        <input style={{ flex: 1 }} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && setQ(query)} />
        <Btn small onClick={() => setQ(query)}>Embed query</Btn>
      </div>
      {data ? <div className="grid g2">
        <Scatter x={data.points.map((p: Any) => p.x)} y={data.points.map((p: Any) => p.y)} c={data.points.map((p: Any) => String(docs.indexOf(p.title)))} classes={docs} size={4.5} height={300}
          extra={(sx, sy) => data.query && <g><circle cx={sx(data.query.x)} cy={sy(data.query.y)} r={8} fill="#fff" stroke="#000" /><text x={sx(data.query.x) + 10} y={sy(data.query.y) + 4} fill="#fff" fontSize={12}>query</text></g>} />
        <div className="col">
          {data.query && <><b>Nearest chunks (real cosine similarity in full dimensions):</b>
            {data.query.nearest.map((n: Any) => <div key={n.id} className="row between"><span className="mono">{n.id}</span><span className="pill cyan">{n.cosine}</span></div>)}</>}
          <small className="muted">{data.note} Explained variance of these 2 axes: {data.explained_variance.map((v: number) => pct(v, 0)).join(' + ')}.</small>
        </div>
      </div> : <Loading />}
    </div>
  )
}

export function Chunker() {
  const [size, setSize] = useState(60)
  const [overlap, setOverlap] = useState(10)
  const { data } = useApi(`/api/widget/chunks${qs({ chunk_size: size, overlap })}`)
  return (
    <div className="col">
      <div className="grid g2">
        <Slider label="chunk size (words)" value={size} min={20} max={300} step={10} onChange={setSize} />
        <Slider label="overlap (words)" value={overlap} min={0} max={Math.min(40, size - 10)} step={5} onChange={setOverlap} />
      </div>
      {data ? <div className="col" style={{ maxHeight: 340, overflowY: 'auto' }}>
        {data.chunks.map((c: Any, i: number) => (
          <div key={c.id} style={{ borderLeft: `4px solid ${PALETTE[i % PALETTE.length]}`, paddingLeft: 10 }}>
            <small className="mono muted">{c.id} · {c.text.split(/\s+/).length} words</small>
            <div style={{ fontSize: '0.85rem' }}>{c.text}</div>
          </div>
        ))}
      </div> : <Loading />}
      <small className="muted">Small chunks are precise but may cut an answer in half; big chunks keep context but dilute the embedding and eat the context budget. Overlap repeats words so facts at boundaries survive.</small>
    </div>
  )
}
