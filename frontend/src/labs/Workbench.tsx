import { useEffect, useMemo, useState } from 'react'
import { post, type Any } from '../api'
import { BarChart, BoundaryPlot, ConfusionMatrix, LineChart, PALETTE, Scatter } from '../charts'
import { Btn, Card, CodeBlock, Diagnosis, ErrorBox, Loading, Select, Slider, Toggle, fmt, pct, useGame } from '../ui'

export type WorkbenchProps = {
  dataset?: string
  lockDataset?: boolean
  context?: string
  recommended?: Any | null
  banned?: string[]
  initial?: Any
  onRun?: (res: Any) => void
  compact?: boolean
}

const METRIC_HELP: Record<string, string> = {
  accuracy: 'Share of all predictions that were right.',
  balanced_accuracy: 'Average recall over classes — not fooled by imbalance.',
  precision: 'Of the rows predicted positive, how many really were.',
  recall: 'Of the truly positive rows, how many the model caught.',
  f1: 'Harmonic mean of precision and recall.',
  mae: 'Mean absolute error: average size of a miss, in target units.',
  rmse: 'Root mean squared error: like MAE but punishes big misses more.',
  r2: 'Share of the target variance explained (1 = perfect, 0 = predicting the mean).',
  silhouette: 'How well separated clusters are (−1…1, higher = better).',
  inertia: 'Sum of squared distances to the nearest centroid (always falls as k grows).',
}

export function MetricGrid({ res }: { res: Any }) {
  const test = res.metrics.test || {}
  const train = res.metrics.train || {}
  const keys = res.task === 'clustering' ? Object.keys(train) : Object.keys(test)
  return (
    <div className="metrics">
      {keys.map((k) => (
        <div key={k} className="metric" title={METRIC_HELP[k]}>
          <div className="metric-l">{k.replace('_', ' ')}{res.primary_metric === k ? ' ★' : ''}</div>
          <div className="metric-v">{fmt(res.task === 'clustering' ? train[k] : test[k])}</div>
          {res.task !== 'clustering' && <div className="metric-s">train {fmt(train[k])}{res.baseline?.[k] !== undefined ? ` · baseline ${fmt(res.baseline[k])}` : ''}</div>}
        </div>
      ))}
      {res.roc_auc != null && <div className="metric" title="Area under the ROC curve: probability a random positive is ranked above a random negative."><div className="metric-l">ROC AUC</div><div className="metric-v">{fmt(res.roc_auc)}</div></div>}
      {res.cv && <div className="metric" title="k-fold cross-validation on the training set"><div className="metric-l">CV {res.cv.scoring} ({res.cv.folds} folds)</div><div className="metric-v">{fmt(res.cv.mean)}</div><div className="metric-s">± {fmt(res.cv.std)}</div></div>}
    </div>
  )
}

export function RunResult({ res, config, showCode }: { res: Any; config: Any; showCode: boolean }) {
  const [cell, setCell] = useState<[number, number] | null>(null)
  return (
    <div className="stack">
      <MetricGrid res={res} />
      <Diagnosis items={res.diagnosis} />
      <div className="grid g2">
        {res.confusion_matrix && (
          <div>
            <div className="field-l">Confusion matrix (test set) — click a cell</div>
            <ConfusionMatrix labels={res.confusion_matrix.labels} matrix={res.confusion_matrix.matrix} onCell={(r, c) => setCell([r, c])} selected={cell} />
            {cell && <div className="info-box" style={{ marginTop: 6 }}>{res.confusion_matrix.matrix[cell[0]][cell[1]]} test rows were actually <b>{res.confusion_matrix.labels[cell[0]]}</b> and predicted <b>{res.confusion_matrix.labels[cell[1]]}</b>{cell[0] === cell[1] ? ' — correct.' : ' — an error.'}</div>}
          </div>
        )}
        {res.decision_boundary && (
          <div><div className="field-l">Decision boundary ({res.decision_boundary.features.join(' vs ')})</div><BoundaryPlot grid={res.decision_boundary} points={res.decision_boundary.points} classes={res.decision_boundary.classes} height={260} /></div>
        )}
        {res.importances?.items?.length > 0 && (
          <div><div className="field-l">Feature influence — {res.importances.method}</div>
            <BarChart items={res.importances.items.slice(0, 12).map((i: Any) => ({ label: i.feature, value: i.value }))} format={(v) => fmt(v, 3)} /></div>
        )}
        {res.task === 'clustering' && res.projection && (
          <div><div className="field-l">Clusters ({res.projection.note})</div><Scatter x={res.projection.x} y={res.projection.y} c={res.projection.cluster} height={260} /></div>
        )}
        {res.elbow && (
          <div><div className="field-l">Elbow method — real inertia for k = 1…8</div><LineChart series={[{ name: 'inertia', values: res.elbow.map((e: Any) => e.inertia) }]} x={res.elbow.map((e: Any) => e.k)} xLabel="k" markX={config.params?.n_clusters ?? 3} /></div>
        )}
        {res.cluster_profiles && (
          <div className="tbl-wrap"><div className="field-l">Cluster profiles (feature means)</div>
            <table className="tbl"><thead><tr><th>cluster</th><th className="num">size</th>{res.cluster_profiles.columns.map((c: string) => <th key={c} className="num">{c}</th>)}</tr></thead>
              <tbody>{res.cluster_profiles.rows.map((r: number[], i: number) => <tr key={i}><td><span style={{ color: PALETTE[i % PALETTE.length] }}>●</span> {i}</td><td className="num">{res.cluster_sizes?.[i]}</td>{r.slice(1).map((v, j) => <td key={j} className="num">{fmt(v, 2)}</td>)}</tr>)}</tbody></table></div>
        )}
        {res.cv && (
          <div><div className="field-l">Cross-validation fold scores</div><BarChart items={res.cv.scores.map((s: number, i: number) => ({ label: `fold ${i + 1}`, value: s }))} format={(v) => v.toFixed(3)} /></div>
        )}
        {res.per_class && res.per_class.length > 2 && (
          <div className="tbl-wrap"><div className="field-l">Per-class results</div><table className="tbl"><thead><tr><th>class</th><th className="num">support</th><th className="num">precision</th><th className="num">recall</th></tr></thead>
            <tbody>{res.per_class.map((p: Any) => <tr key={p.label}><td>{p.label}</td><td className="num">{p.support}</td><td className="num">{fmt(p.precision)}</td><td className="num">{fmt(p.recall)}</td></tr>)}</tbody></table></div>
        )}
      </div>
      {res.sample_predictions?.length > 0 && (
        <div className="tbl-wrap"><div className="field-l">Sample test predictions</div>
          <table className="tbl"><thead><tr>{Object.keys(res.sample_predictions[0].features).slice(0, 6).map((k) => <th key={k}>{k}</th>)}<th>actual</th><th>predicted</th>{res.sample_predictions[0].probability != null && <th className="num">P(positive)</th>}</tr></thead>
            <tbody>{res.sample_predictions.map((s: Any, i: number) => (
              <tr key={i}>{Object.values(s.features).slice(0, 6).map((v: Any, j) => <td key={j}>{typeof v === 'string' && v.length > 40 ? v.slice(0, 40) + '…' : fmt(v, 2)}</td>)}
                <td>{fmt(s.actual, 2)}</td><td style={{ color: String(s.actual) === String(s.predicted) || (typeof s.actual === 'number' && Math.abs(s.actual - s.predicted) / (Math.abs(s.actual) || 1) < 0.1) ? 'var(--green)' : 'var(--red)' }}>{fmt(s.predicted, 2)}</td>
                {s.probability != null && <td className="num">{fmt(s.probability)}</td>}</tr>
            ))}</tbody></table></div>
      )}
      <div className="row" style={{ gap: '1.2rem', fontSize: '0.82rem' }}>
        <span className="muted">⏱ train {fmt(res.train_ms, 1)} ms</span>
        {res.predict_ms_per_1k != null && <span className="muted">⚡ {fmt(res.predict_ms_per_1k, 2)} ms / 1k predictions</span>}
        {res.model_size_kb != null && <span className="muted">💾 {fmt(res.model_size_kb, 1)} KB</span>}
        <span className="muted">🔎 interpretability {res.interpretability}/5</span>
        <span className="muted">rows: {res.n_train} train / {res.n_test} test</span>
      </div>
      {res.cleaning_log?.length > 0 && <div className="info-box"><b>Cleaning log</b><ul style={{ margin: '0.3rem 0 0', paddingLeft: '1.2rem' }}>{res.cleaning_log.map((l: string, i: number) => <li key={i}>{l}</li>)}</ul></div>}
      {showCode && res.code && <CodeBlock code={res.code} title="equivalent scikit-learn code — this is what just ran" />}
    </div>
  )
}

export default function Workbench({ dataset, lockDataset, context, recommended, banned = [], initial, onRun, compact }: WorkbenchProps) {
  const { pid, meta, ov, reward } = useGame()
  const catalog: Any[] = meta.datasets
  const models: Record<string, Any> = meta.models
  const [ds, setDs] = useState<string>(initial?.dataset || dataset || 'student_success')
  const info = catalog.find((d) => d.id === ds) || catalog[0]
  const task = info.task === 'text_classification' ? 'classification' : info.task
  const allFeatures: string[] = info.columns.filter((c: string) => c !== info.target)
  const [features, setFeatures] = useState<string[]>(initial?.features || [])
  const [model, setModel] = useState<string>(initial?.model || '')
  const [params, setParams] = useState<Any>(initial?.params || {})
  const [prep, setPrep] = useState<Any>({ impute_numeric: 'median', impute_categorical: 'most_frequent', scaling: 'none', encoding: 'onehot', text_vectorizer: 'tfidf', ...(initial?.preprocessing || {}) })
  const [testSize, setTestSize] = useState(initial?.test_size ?? 0.2)
  const [seed, setSeed] = useState(initial?.seed ?? 42)
  const [cv, setCv] = useState(initial?.cv_folds ?? 0)
  const [threshold, setThreshold] = useState(initial?.threshold ?? 0.5)
  const [extraRows, setExtraRows] = useState(initial?.extra_rows ?? 0)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [out, setOut] = useState<Any>(null)
  const [showCode, setShowCode] = useState<boolean>(!!ov.player.settings.show_code)
  const [advanced, setAdvanced] = useState(false)

  // reset sensible defaults when the dataset changes
  useEffect(() => {
    if (initial && initial.dataset === ds) return
    const feats = allFeatures.filter((f) => !banned.includes(f) && !/_id$/.test(f))
    setFeatures(feats)
    const first = Object.entries(models).find(([, m]) => m.task === task && !m.label.startsWith('Baseline'))
    setModel(first ? first[0] : '')
    setParams({})
    setOut(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ds])

  const modelOpts = useMemo(() => Object.entries(models).filter(([, m]) => m.task === task).map(([id, m]) => ({ value: id, label: m.label })), [models, task])
  const spec = models[model]
  const binary = task === 'classification' && ['student_success', 'spam', 'customer_churn', 'sentiment', 'fraud', 'overfit_lab', 'loan_leak', 'data_chaos', 'moons', 'circles', 'spiral', 'xor', 'linear2d'].includes(ds)
  const hasText = ['spam', 'sentiment'].includes(ds)

  const applyRecommended = () => {
    if (!recommended) return
    if (recommended.features) setFeatures(recommended.features)
    if (recommended.model) setModel(recommended.model)
    setParams(recommended.params || {})
    if (recommended.preprocessing) setPrep((p: Any) => ({ ...p, ...recommended.preprocessing }))
    if (recommended.cv_folds) setCv(recommended.cv_folds)
    if (recommended.threshold) setThreshold(recommended.threshold)
  }

  const run = async () => {
    setBusy(true)
    try {
      const config: Any = { dataset: ds, target: task === 'clustering' ? null : info.target, features, model, params, preprocessing: prep, test_size: testSize, seed, cv_folds: cv, threshold, extra_rows: extraRows }
      const r = await post(`/api/p/${pid}/run/ml`, { config, context, name: name || null })
      setOut(r)
      if (r.ok) reward(r)
      onRun?.(r)
    } catch (e: Any) {
      setOut({ ok: false, error: e.message })
    } finally {
      setBusy(false)
    }
  }

  const toggleFeature = (f: string) => setFeatures((fs) => (fs.includes(f) ? fs.filter((x) => x !== f) : [...fs, f]))
  const setP = (k: string, v: Any) => setPrep((p: Any) => ({ ...p, [k]: v }))

  return (
    <div className={`grid ${compact ? '' : 'g-side'}`}>
      <div className="col">
        <Card title="1 · Data" icon="🗃️">
          <Select label="Dataset" value={ds} onChange={(v) => !lockDataset && setDs(v)} options={catalog.filter((d) => !d.toy && (!lockDataset || d.id === ds)).map((d) => ({ value: d.id, label: `${d.icon} ${d.title}` }))} />
          <p className="muted" style={{ margin: '0.4rem 0' }}><small>{info.description} Target: <b className="mono">{info.target || '— (unsupervised)'}</b> · {info.rows} rows</small></p>
          <div className="field-l">Features ({features.length} selected)</div>
          <div className="chips">
            {allFeatures.map((f) => <button key={f} className={`chip ${features.includes(f) ? 'on' : ''} ${banned.includes(f) ? 'banned' : ''}`} onClick={() => toggleFeature(f)} title={banned.includes(f) ? 'Not allowed in this mission' : ''}>{f}</button>)}
          </div>
          <div className="row" style={{ marginTop: 6 }}><button className="link" onClick={() => setFeatures(allFeatures.filter((f) => !banned.includes(f)))}>all</button><button className="link" onClick={() => setFeatures([])}>none</button></div>
          {ds === 'overfit_lab' && <Slider label="collect extra rows (more data)" value={extraRows} min={0} max={400} step={20} onChange={setExtraRows} hint="Gather more labelled sensor readings" />}
        </Card>
        <Card title="2 · Model" icon="🤖">
          <Select label="Algorithm" value={model} onChange={(v) => { setModel(v); setParams({}) }} options={modelOpts} />
          {spec && <p className="muted" style={{ margin: '0.4rem 0' }}><small>{spec.notes}</small></p>}
          {spec?.params.map((p: Any) => {
            const v = params[p.name] ?? p.default
            if (p.kind === 'choice') return <Select key={p.name} label={p.name} value={String(v ?? 'none')} onChange={(x) => setParams({ ...params, [p.name]: x })} options={p.choices} hint={p.help} />
            if (p.kind === 'int_or_none') return (
              <div key={p.name} className="col" style={{ gap: 2 }}>
                <Toggle label={`limit ${p.name}`} checked={v !== null && v !== undefined} onChange={(on) => setParams({ ...params, [p.name]: on ? 4 : null })} hint={p.help} />
                {v !== null && v !== undefined && <Slider label={p.name} value={Number(v)} min={p.min} max={p.max} onChange={(x) => setParams({ ...params, [p.name]: x })} hint={p.help} />}
              </div>
            )
            if (p.kind === 'float' && p.log) {
              const lv = Math.log10(Number(v))
              return <Slider key={p.name} label={p.name} value={lv} min={Math.log10(p.min)} max={Math.log10(p.max)} step={0.1} onChange={(x) => setParams({ ...params, [p.name]: Number(Math.pow(10, x).toPrecision(2)) })} fmt={(x) => String(Number(Math.pow(10, x).toPrecision(2)))} hint={p.help} />
            }
            return <Slider key={p.name} label={p.name} value={Number(v)} min={p.min} max={p.max} step={p.kind === 'float' ? 0.01 : 1} onChange={(x) => setParams({ ...params, [p.name]: x })} hint={p.help} />
          })}
        </Card>
        <Card title="3 · Preprocessing & evaluation" icon="🧹" right={<button className="link" onClick={() => setAdvanced(!advanced)}>{advanced ? 'hide' : 'show'} options</button>}>
          {advanced ? (
            <div className="col">
              <div className="grid g2">
                <Select label="Impute numbers" value={prep.impute_numeric} onChange={(v) => setP('impute_numeric', v)} options={['median', 'mean', 'most_frequent', 'none']} hint="How missing numeric values are filled (fit on training data only)" />
                <Select label="Scaling" value={prep.scaling} onChange={(v) => setP('scaling', v)} options={['none', 'standard', 'minmax']} />
                <Select label="Impute categories" value={prep.impute_categorical} onChange={(v) => setP('impute_categorical', v)} options={['most_frequent', 'none']} />
                <Select label="Encode categories" value={prep.encoding} onChange={(v) => setP('encoding', v)} options={['onehot', 'ordinal']} />
                {hasText && <Select label="Text → numbers" value={prep.text_vectorizer} onChange={(v) => setP('text_vectorizer', v)} options={[{ value: 'tfidf', label: 'TF-IDF (1–2 grams)' }, { value: 'count', label: 'Word counts' }]} />}
              </div>
              <div className="field-l">Cleaning steps</div>
              <div className="col" style={{ gap: 4 }}>
                <Toggle label="Remove duplicate rows (before splitting)" checked={!!prep.dedupe} onChange={(v) => setP('dedupe', v)} />
                <Toggle label="Fix types ('34 yrs' → 34, 'N/A' → missing)" checked={!!prep.fix_types} onChange={(v) => setP('fix_types', v)} />
                <Toggle label="Normalise category spellings ('BASIC ' → 'basic')" checked={!!prep.normalize_categories} onChange={(v) => setP('normalize_categories', v)} />
                <Toggle label="Impossible values → missing (negative age…)" checked={!!prep.invalid_to_missing} onChange={(v) => setP('invalid_to_missing', v)} />
                <Toggle label="Clip outliers to the 1st–99th percentile" checked={!!prep.clip_outliers} onChange={(v) => setP('clip_outliers', v)} />
              </div>
              <hr />
              {task !== 'clustering' && <>
                <Slider label="test size" value={testSize} min={0.1} max={0.5} step={0.05} onChange={setTestSize} fmt={(v) => `${Math.round(v * 100)}%`} />
                <Select label="Cross-validation" value={String(cv)} onChange={(v) => setCv(Number(v))} options={[{ value: '0', label: 'off' }, { value: '3', label: '3 folds' }, { value: '5', label: '5 folds' }, { value: '10', label: '10 folds' }]} />
                {binary && <Slider label="decision threshold" value={threshold} min={0.05} max={0.95} step={0.05} onChange={setThreshold} hint="Predict positive when P(positive) ≥ threshold" />}
              </>}
              <Slider label="random seed" value={seed} min={0} max={100} onChange={setSeed} hint="Same seed + same config = same result (reproducibility)" />
            </div>
          ) : <small className="muted">Imputation: {prep.impute_numeric} · scaling: {prep.scaling} · test {Math.round(testSize * 100)}% · seed {seed}{cv ? ` · ${cv}-fold CV` : ''}{threshold !== 0.5 ? ` · threshold ${threshold}` : ''}</small>}
        </Card>
        <Card>
          <div className="col">
            <input placeholder="Run name (optional, e.g. 'RF depth 5 scaled')" value={name} onChange={(e) => setName(e.target.value)} />
            <div className="row">
              <Btn onClick={run} disabled={busy || !model || (task !== 'clustering' && features.length === 0)}>{busy ? 'Training…' : '▶ Train & evaluate'}</Btn>
              {recommended && <Btn kind="ghost" onClick={applyRecommended} title="Beginner mode suggestion">✨ Use recommended setup</Btn>}
              <Toggle label="Code" checked={showCode} onChange={setShowCode} />
            </div>
          </div>
        </Card>
      </div>
      <div className="col">
        {busy && <Loading text="Training a real scikit-learn model…" />}
        {!out && !busy && <Card><div className="empty">Choose features and a model, then train. Every number you'll see is computed by a real model on real (synthetic, openly generated) data — nothing is pre-baked.</div></Card>}
        {out && !out.ok && (
          <Card title="The pipeline refused to run" icon="🛑" className="danger">
            <ErrorBox error={out.error} />
            {out.lesson && <p style={{ marginTop: '0.6rem' }}>📘 {out.lesson}</p>}
            {out.fix && <p className="muted">🔧 {out.fix}</p>}
          </Card>
        )}
        {out?.ok && (
          <Card title={<>Run #{out.run_id} · {models[out.config.model].label}</>} icon="📊" right={<a href="#/history" className="pill cyan">saved to experiment history</a>}>
            <RunResult res={out.result} config={out.config} showCode={showCode} />
          </Card>
        )}
      </div>
    </div>
  )
}
