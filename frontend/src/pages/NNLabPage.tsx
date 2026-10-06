import { useEffect, useState } from 'react'
import { post, type Any } from '../api'
import { BoundaryPlot, LineChart, PALETTE } from '../charts'
import { Btn, Card, CodeBlock, Diagnosis, ErrorBox, Loading, Select, Slider, Tabs, Toggle, fmt, pct, useApi, useGame } from '../ui'
import { NnDiagram } from '../widgets/basic'

const PRESETS: Record<string, { label: string; cfg: Any; why: string }> = {
  good: { label: '✅ Sensible start', cfg: { dataset: 'moons', hidden: [16, 16], activation: 'relu', optimizer: 'adam', lr: 0.01, batch_size: 32, epochs: 150, dropout: 0, l2: 0 }, why: 'Two small hidden layers, Adam, moderate learning rate.' },
  tiny: { label: '🐜 Too small', cfg: { dataset: 'spiral', hidden: [2], activation: 'relu', optimizer: 'adam', lr: 0.01, batch_size: 32, epochs: 200, dropout: 0, l2: 0 }, why: 'Two neurons cannot bend the boundary enough for a spiral → underfitting.' },
  linear: { label: '📏 No non-linearity', cfg: { dataset: 'circles', hidden: [32, 32], activation: 'linear', optimizer: 'adam', lr: 0.01, batch_size: 32, epochs: 150, dropout: 0, l2: 0 }, why: 'Stacked linear layers are still one linear function — it cannot separate circles.' },
  hot: { label: '🔥 Learning rate too high', cfg: { dataset: 'moons', hidden: [16, 16], activation: 'relu', optimizer: 'sgd', lr: 20, batch_size: 32, epochs: 100, dropout: 0, l2: 0 }, why: 'Steps overshoot the valley. Watch the loss curve.' },
  cold: { label: '🧊 Learning rate too low', cfg: { dataset: 'moons', hidden: [16, 16], activation: 'relu', optimizer: 'sgd', lr: 0.0005, batch_size: 32, epochs: 100, dropout: 0, l2: 0 }, why: 'Tiny steps: after 100 epochs it has barely moved.' },
  memorise: { label: '🧠 Memoriser', cfg: { dataset: 'overfit_lab', hidden: [128, 128], activation: 'relu', optimizer: 'adam', lr: 0.01, batch_size: 16, epochs: 300, dropout: 0, l2: 0 }, why: 'Huge network, little noisy data, long training → it memorises.' },
}

const EXPECT = [
  { value: 'healthy', label: 'It learns well (train ≈ validation, both high)' },
  { value: 'underfit', label: 'Underfits (both scores low)' },
  { value: 'overfit', label: 'Overfits (train ≫ validation)' },
  { value: 'unstable', label: 'Training is unstable / diverges' },
  { value: 'slow', label: 'Learns too slowly (not finished in time)' },
]

export default function NNLabPage() {
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">Neural Network Tower</div><h1 style={{ margin: 0 }}>Neural Network Lab</h1></div></div>
      <NNLab />
    </div>
  )
}

export function NNLab({ initial, context = 'nn_lab', onRun, embedded }: { initial?: Any; context?: string; onRun?: (r: Any) => void; embedded?: boolean }) {
  const { pid, reward } = useGame()
  const [cfg, setCfg] = useState<Any>(initial || PRESETS.good.cfg)
  const [expect, setExpect] = useState('')
  const [busy, setBusy] = useState(false)
  const [out, setOut] = useState<Any>(null)
  const [snap, setSnap] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [tab, setTab] = useState('curves')
  const [showSrc, setShowSrc] = useState(false)
  const src = useApi<string>(showSrc ? '/api/nn/source' : null, [showSrc])
  const set = (k: string, v: Any) => setCfg((c: Any) => ({ ...c, [k]: v }))
  const res = out?.result

  useEffect(() => {
    if (!playing || !res?.snapshots?.length) return
    if (snap >= res.snapshots.length) { setPlaying(false); return }
    const t = setTimeout(() => setSnap((s) => s + 1), 550)
    return () => clearTimeout(t)
  }, [playing, snap, res])

  const run = async () => {
    setBusy(true)
    try {
      const r = await post(`/api/p/${pid}/run/nn`, { config: cfg, context })
      setOut(r)
      if (r.ok) { reward(r); setSnap(r.result.snapshots?.length || 0); setTab('curves') }
      onRun?.(r)
    } catch (e: Any) { setOut({ ok: false, error: e.message }) } finally { setBusy(false) }
  }

  const codes: string[] = res ? res.diagnosis.map((d: Any) => d.code) : []
  const expectRight = expect && res ? (expect === 'unstable' ? codes.includes('unstable') || codes.includes('diverged') : expect === 'underfit' ? codes.includes('underfit') || codes.includes('linear') : codes.includes(expect)) : null
  const hist = res?.history
  const grid = res?.snapshots?.length && snap < res.snapshots.length ? res.snapshots[snap].grid : res?.boundary

  return (
    <div className="stack">
      {!embedded && <div className="row" style={{ flexWrap: 'wrap' }}>
        {Object.entries(PRESETS).map(([k, p]) => <Btn key={k} small kind="ghost" title={p.why} onClick={() => { setCfg(p.cfg); setOut(null) }}>{p.label}</Btn>)}
      </div>}
      <div className="grid g-side">
        <div className="col">
          <Card title="Architecture" icon="🧬">
            <Select label="Dataset" value={cfg.dataset} onChange={(v) => !embedded && set('dataset', v)} options={[
              { value: 'moons', label: '🌙 Two moons (2D)' }, { value: 'circles', label: '⭕ Circles (2D)' }, { value: 'spiral', label: '🌀 Spiral (2D, hard)' }, { value: 'xor', label: '✖️ XOR (2D)' },
              { value: 'linear2d', label: '📏 Linearly separable (2D)' }, { value: 'blobs', label: '🫧 Blobs, 3 classes (2D)' }, { value: 'student_success', label: '🎓 Student success (4 features)' },
              { value: 'digits', label: '🔢 Handwritten digits (64 pixels, 10 classes)' }, { value: 'overfit_lab', label: '🧪 Overfit lab (tiny + noisy)' }]} />
            <div className="field-l">Hidden layers (neurons per layer)</div>
            {cfg.hidden.map((h: number, i: number) => (
              <div key={i} className="row">
                <div style={{ flex: 1 }}><Slider label={`layer ${i + 1}`} value={h} min={1} max={128} onChange={(v) => set('hidden', cfg.hidden.map((x: number, j: number) => (j === i ? v : x)))} /></div>
                <button className="link" onClick={() => set('hidden', cfg.hidden.filter((_: number, j: number) => j !== i))}>✕</button>
              </div>
            ))}
            {cfg.hidden.length < 5 && <button className="link" onClick={() => set('hidden', [...cfg.hidden, 8])}>+ add layer</button>}
            {cfg.hidden.length === 0 && <small className="muted">No hidden layers = logistic regression.</small>}
            <Select label="Activation" value={cfg.activation} onChange={(v) => set('activation', v)} options={['relu', 'tanh', 'sigmoid', 'linear']} hint="The non-linearity applied after each hidden layer" />
          </Card>
          <Card title="Training" icon="🏋️">
            <Select label="Optimizer" value={cfg.optimizer} onChange={(v) => set('optimizer', v)} options={[{ value: 'sgd', label: 'SGD' }, { value: 'momentum', label: 'SGD + momentum' }, { value: 'adam', label: 'Adam' }]} />
            <Slider label="learning rate" value={Math.log10(cfg.lr)} min={-4} max={1.5} step={0.1} onChange={(v) => set('lr', Number(Math.pow(10, v).toPrecision(2)))} fmt={(v) => String(Number(Math.pow(10, v).toPrecision(2)))} hint="Step size of each weight update" />
            <Select label="Batch size" value={String(cfg.batch_size)} onChange={(v) => set('batch_size', Number(v))} options={['1', '8', '16', '32', '64', '128', '512']} />
            <Slider label="epochs" value={cfg.epochs} min={1} max={500} step={1} onChange={(v) => set('epochs', v)} />
            <Slider label="dropout" value={cfg.dropout} min={0} max={0.8} step={0.05} onChange={(v) => set('dropout', v)} hint="Randomly switch off neurons during training" />
            <Slider label="L2 weight decay" value={cfg.l2} min={0} max={0.05} step={0.001} onChange={(v) => set('l2', v)} />
          </Card>
          <Card title="Predict, then train" icon="🔮">
            <Select label="What do you expect will happen?" value={expect} onChange={setExpect} options={[{ value: '', label: '— make a prediction (optional) —' }, ...EXPECT]} />
            <div className="row" style={{ marginTop: 8 }}>
              <Btn onClick={run} disabled={busy}>{busy ? 'Training…' : '▶ Train network'}</Btn>
              <small className="muted">Real NumPy back-propagation, every run.</small>
            </div>
          </Card>
        </div>
        <div className="col">
          <Card title="Network" icon="🕸️" right={res && <small className="muted">{res.n_params.toLocaleString()} parameters</small>}>
            <NnDiagram key={JSON.stringify(res?.sizes || cfg.hidden)} sizes={res ? res.sizes : [cfg.dataset === 'digits' ? 64 : cfg.dataset === 'student_success' ? 4 : 2, ...cfg.hidden, 2]} weights={res?.weights} />
          </Card>
          {busy && <Loading text="Back-propagating…" />}
          {out && !out.ok && <ErrorBox error={out.error} />}
          {res && (
            <>
              {expectRight !== null && <div className={expectRight ? 'ok-box' : 'warn-box'}>{expectRight ? '🎯 Your prediction matched the real outcome!' : `🤔 You predicted “${EXPECT.find((e) => e.value === expect)?.label}”. The real run says otherwise — read the diagnosis below to see why.`}</div>}
              <div className="metrics">
                <div className="metric"><div className="metric-l">train accuracy</div><div className="metric-v">{pct(res.final.train_acc)}</div></div>
                <div className="metric"><div className="metric-l">validation accuracy</div><div className="metric-v">{pct(res.final.val_acc)}</div></div>
                <div className="metric"><div className="metric-l">val loss</div><div className="metric-v">{fmt(res.final.val_loss)}</div></div>
                <div className="metric"><div className="metric-l">best val epoch</div><div className="metric-v">{res.best_val_epoch}</div><div className="metric-s">of {res.epochs_run}{res.capped ? ' (capped)' : ''}</div></div>
                <div className="metric"><div className="metric-l">time</div><div className="metric-v">{fmt(res.train_ms / 1000, 2)}s</div></div>
              </div>
              <Diagnosis items={res.diagnosis} />
              <Tabs value={tab} onChange={setTab} tabs={[{ id: 'curves', label: 'Learning curves' }, ...(res.boundary ? [{ id: 'boundary', label: 'Decision boundary' }] : []), { id: 'grads', label: 'Gradients & weights' }, { id: 'code', label: 'Code' }]} />
              {tab === 'curves' && (
                <div className="grid g2">
                  <div><div className="field-l">Loss (log scale)</div><LineChart logY series={[{ name: 'train loss', values: hist.train_loss }, { name: 'val loss', values: hist.val_loss, color: PALETTE[1] }]} x={hist.epoch} xLabel="epoch" markX={res.diverged_at} /></div>
                  <div><div className="field-l">Accuracy</div><LineChart yDomain={[0, 1]} series={[{ name: 'train acc', values: hist.train_acc }, { name: 'val acc', values: hist.val_acc, color: PALETTE[1] }]} x={hist.epoch} xLabel="epoch" markX={res.best_val_epoch} /></div>
                </div>
              )}
              {tab === 'boundary' && res.boundary && (
                <div className="grid g2">
                  <BoundaryPlot grid={grid} points={res.points} classes={res.classes} height={300} />
                  <div className="col">
                    {res.snapshots?.length > 0 && <>
                      <Slider label="training progress" value={snap} min={0} max={res.snapshots.length} onChange={(v) => { setSnap(v); setPlaying(false) }} fmt={(v) => (v >= res.snapshots.length ? `final (epoch ${res.epochs_run})` : `epoch ${res.snapshots[v].epoch}`)} />
                      <Btn small kind="ghost" onClick={() => { setSnap(0); setPlaying(true) }}>▶ Replay how the boundary formed</Btn>
                    </>}
                    <small className="muted">Colours show the network's real predicted probability across the input space. Points are the training data (validation points omitted for clarity).</small>
                  </div>
                </div>
              )}
              {tab === 'grads' && (
                <div className="grid g2">
                  <div><div className="field-l">Gradient norm per epoch (log)</div><LineChart logY series={[{ name: '‖gradient‖', values: hist.grad_norm, color: PALETTE[3] }]} x={hist.epoch} xLabel="epoch" /><small className="muted">Exploding (huge) or vanishing (≈0) gradients explain unstable or stalled training.</small></div>
                  <div className="tbl-wrap"><div className="field-l">Weights per layer</div><table className="tbl"><thead><tr><th>layer</th><th>shape</th><th className="num">mean |w|</th></tr></thead><tbody>{res.weight_stats.map((w: Any, i: number) => <tr key={i}><td>{i + 1}</td><td className="mono">{w.shape.join('×')}</td><td className="num">{fmt(w.mean_abs)}</td></tr>)}</tbody></table></div>
                </div>
              )}
              {tab === 'code' && (
                <div className="col">
                  <CodeBlock code={res.code} title="Equivalent PyTorch" />
                  <Toggle label="Show the NumPy implementation that actually ran" checked={showSrc} onChange={setShowSrc} />
                  {showSrc && (src.data ? <CodeBlock code={src.data} title="neural_forge/nn.py — forward pass, back-propagation, optimizers" /> : <Loading />)}
                </div>
              )}
            </>
          )}
          {!res && !busy && <Card><div className="empty">Pick a preset or build your own network, predict what will happen, then train. Bad configurations are allowed — breaking things is how you learn what each knob does.</div></Card>}
        </div>
      </div>
    </div>
  )
}
