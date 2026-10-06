import { useState } from 'react'
import { post, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { ConfusionMatrix, LineChart, PALETTE } from '../charts'
import { Btn, Card, ErrorBox, Loading, Pill, Select, Slider, Tabs, fmt, pct, useGame } from '../ui'

function PixelImage({ pixels, label }: { pixels: number[][]; label?: string }) {
  return (
    <div className="digit-sample" title={label}>
      <div className="pixel-grid" style={{ gridTemplateColumns: `repeat(${pixels[0]?.length || 8}, 1fr)` }} dir="ltr">
        {pixels.flat().map((value, index) => <div key={index} style={{ background: `rgba(226,232,240,${Math.max(0, Math.min(1, value))})` }} />)}
      </div>
      {label && <small>{label}</small>}
    </div>
  )
}

function Curves({ result }: { result: Any }) {
  const { t } = useI18n()
  const history = result.history
  return (
    <Card title={t('torch.curves')} icon="📈">
      <div className="grid g2">
        <div><div className="field-l">{t('torch.trainLoss')} / {t('torch.validationLoss')}</div><LineChart x={history.epoch} xLabel="epoch" series={[{ name: t('torch.trainLoss'), values: history.train_loss }, { name: t('torch.validationLoss'), values: history.val_loss, color: PALETTE[1] }]} /></div>
        <div><div className="field-l">{t('torch.trainAccuracy')} / {t('torch.validationAccuracy')}</div><LineChart x={history.epoch} xLabel="epoch" yDomain={[0, 1]} series={[{ name: t('torch.trainAccuracy'), values: history.train_acc }, { name: t('torch.validationAccuracy'), values: history.val_acc, color: PALETTE[2] }]} /></div>
      </div>
      <div className="metrics">
        <div className="metric"><div className="metric-l">{t('torch.trainAccuracy')}</div><div className="metric-v">{pct(result.final.train_accuracy)}</div></div>
        <div className="metric"><div className="metric-l">{t('torch.validationAccuracy')}</div><div className="metric-v">{pct(result.final.validation_accuracy)}</div></div>
        <div className="metric"><div className="metric-l">{t('common.device')}</div><div className="metric-v"><Ltr>{result.device}</Ltr></div></div>
        <div className="metric"><div className="metric-l">{t('common.duration')}</div><div className="metric-v">{fmt(result.duration_seconds, 2)}s</div></div>
        <div className="metric"><div className="metric-l">parameters</div><div className="metric-v">{result.parameters.toLocaleString()}</div></div>
      </div>
      {result.warnings?.map((warning: string, index: number) => <div key={index} className="warn-box">{warning}</div>)}
      {result.checkpoint && <div className="info-box">💾 {t('torch.checkpoint')}: <Ltr>{result.checkpoint.id}</Ltr> <Pill>{result.checkpoint.format}</Pill></div>}
    </Card>
  )
}

function MlpLab() {
  const { t } = useI18n()
  const { pid, reward } = useGame()
  const [config, setConfig] = useState<Any>({ dataset: 'moons', hidden: [32, 16], activation: 'relu', optimizer: 'adam', learning_rate: 0.001, batch_size: 32, epochs: 30, dropout: 0.1, device: 'auto', seed: 42 })
  const [busy, setBusy] = useState(false)
  const [output, setOutput] = useState<Any>(null)
  const [error, setError] = useState<string | null>(null)
  const set = (key: string, value: Any) => setConfig((current: Any) => ({ ...current, [key]: value }))
  const run = async () => {
    setBusy(true); setError(null); setOutput(null)
    try { const response = await post(`/api/p/${pid}/pytorch/train`, { config }); setOutput(response); reward(response) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  return (
    <div className="grid g-side">
      <div className="col">
        <Card title={t('torch.architecture')} icon="🧬">
          <Select label="Dataset" value={config.dataset} onChange={(value) => set('dataset', value)} options={['moons', 'circles', 'spiral', 'xor', 'linear2d', 'blobs', 'student_success', 'digits', 'overfit_lab']} />
          <div className="field-l">{t('torch.hidden')}</div>
          {config.hidden.map((neurons: number, index: number) => <div className="row" key={index}><div style={{ flex: 1 }}><Slider label={`layer ${index + 1}`} value={neurons} min={2} max={256} onChange={(value) => set('hidden', config.hidden.map((item: number, i: number) => i === index ? value : item))} /></div><button className="link" onClick={() => set('hidden', config.hidden.filter((_: number, i: number) => i !== index))}>✕</button></div>)}
          {config.hidden.length < 4 && <button className="link" onClick={() => set('hidden', [...config.hidden, 16])}>+ layer</button>}
          <Select label={t('torch.activation')} value={config.activation} onChange={(value) => set('activation', value)} options={['relu', 'tanh', 'sigmoid', 'gelu']} />
          <Slider label={t('torch.dropout')} value={config.dropout} min={0} max={0.8} step={0.05} onChange={(value) => set('dropout', value)} />
        </Card>
        <Card title={t('common.train')} icon="🏋️">
          <Select label={t('torch.optimizer')} value={config.optimizer} onChange={(value) => set('optimizer', value)} options={['adam', 'adamw', 'sgd', 'momentum']} />
          <Slider label={t('torch.learningRate')} value={Math.log10(config.learning_rate)} min={-5} max={-0.3} step={0.1} onChange={(value) => set('learning_rate', Number(10 ** value))} fmt={(value) => (10 ** value).toExponential(1)} />
          <Select label={t('torch.batchSize')} value={String(config.batch_size)} onChange={(value) => set('batch_size', Number(value))} options={['8', '16', '32', '64', '128']} />
          <Slider label={t('torch.epochs')} value={config.epochs} min={1} max={100} onChange={(value) => set('epochs', value)} />
          <Select label={t('torch.preferredDevice')} value={config.device} onChange={(value) => set('device', value)} options={[{ value: 'auto', label: t('torch.autoDevice') }, { value: 'cpu', label: 'CPU' }, { value: 'cuda', label: 'CUDA GPU' }]} />
          <Btn onClick={run} disabled={busy}>{busy ? t('torch.training') : `▶ ${t('torch.train')}`}</Btn>
        </Card>
      </div>
      <div className="col">
        {busy && <Loading text={t('torch.training')} />}
        <ErrorBox error={error} />
        {output?.result && <><Curves result={output.result} /><Card title="Configuration" icon="🧾"><pre className="json-view" dir="ltr">{JSON.stringify(output.result.config, null, 2)}</pre></Card></>}
        {!output && !busy && !error && <Card><div className="empty">optimizer.zero_grad() → loss.backward() → optimizer.step()</div></Card>}
      </div>
    </div>
  )
}

function CnnLab() {
  const { t } = useI18n()
  const { pid, reward } = useGame()
  const [config, setConfig] = useState<Any>({ optimizer: 'adam', learning_rate: 0.001, batch_size: 64, epochs: 5, dropout: 0.1, augmentation: 'none', device: 'auto', seed: 42 })
  const [busy, setBusy] = useState(false)
  const [output, setOutput] = useState<Any>(null)
  const [error, setError] = useState<string | null>(null)
  const set = (key: string, value: Any) => setConfig((current: Any) => ({ ...current, [key]: value }))
  const run = async () => {
    setBusy(true); setError(null); setOutput(null)
    try { const response = await post(`/api/p/${pid}/cnn/train`, { config }); setOutput(response); reward(response) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const result = output?.result
  return (
    <div className="stack">
      <Card title="Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Linear" icon="👁️">
        <div className="grid g4">
          <Select label={t('torch.optimizer')} value={config.optimizer} onChange={(value) => set('optimizer', value)} options={['adam', 'adamw', 'sgd', 'momentum']} />
          <Select label={t('torch.batchSize')} value={String(config.batch_size)} onChange={(value) => set('batch_size', Number(value))} options={['16', '32', '64', '128']} />
          <Slider label={t('torch.epochs')} value={config.epochs} min={1} max={20} onChange={(value) => set('epochs', value)} />
          <Select label={t('torch.preferredDevice')} value={config.device} onChange={(value) => set('device', value)} options={['auto', 'cpu', 'cuda']} />
          <Select label={t('torch.augmentation')} value={config.augmentation} onChange={(value) => set('augmentation', value)} options={[{ value: 'none', label: t('torch.none') }, { value: 'shift', label: t('torch.shift') }, { value: 'noise', label: t('torch.noise') }, { value: 'shift_noise', label: t('torch.shiftNoise') }]} />
        </div>
        <Btn onClick={run} disabled={busy}>{busy ? t('torch.training') : `▶ ${t('torch.cnnTrain')}`}</Btn>
      </Card>
      {busy && <Loading text={t('torch.training')} />}
      <ErrorBox error={error} />
      {result && <>
        <Curves result={result} />
        <div className="grid g2">
          <Card title={t('torch.confusion')} icon="🔢"><ConfusionMatrix labels={result.confusion_matrix.labels} matrix={result.confusion_matrix.matrix} /></Card>
          <Card title={t('torch.mistakes')} icon="🔎"><div className="digit-gallery">{result.misclassified.length ? result.misclassified.map((item: Any, index: number) => <PixelImage key={index} pixels={item.image} label={`${item.actual} → ${item.predicted} (${pct(item.confidence, 0)})`} />) : <div className="ok-box">No validation mistakes in this run.</div>}</div></Card>
          <Card title={t('torch.correct')} icon="✅"><div className="digit-gallery">{result.correct_predictions.map((item: Any, index: number) => <PixelImage key={index} pixels={item.image} label={`${item.predicted} · ${pct(item.confidence, 0)}`} />)}</div></Card>
          <Card title={t('torch.augmentationPreview')} icon="🔄"><div className="digit-gallery"><PixelImage pixels={result.augmentation_preview.original} label={t('torch.original')} /><PixelImage pixels={result.augmentation_preview.transformed} label={t('torch.transformed')} /></div><small className="muted">{result.augmentation_preview.note}</small></Card>
          <Card title={t('torch.featureMaps')} icon="🧠"><div className="digit-gallery">{result.feature_maps.map((map: number[][], index: number) => { const min = Math.min(...map.flat()); const max = Math.max(...map.flat()); const normalized = map.map((row) => row.map((value) => (value - min) / (max - min || 1))); return <PixelImage key={index} pixels={normalized} label={`channel ${index + 1}`} /> })}</div><small className="muted">{result.feature_map_note}</small></Card>
        </div>
      </>}
    </div>
  )
}

export default function PyTorchLab({ initialTab = 'mlp' }: { initialTab?: 'mlp' | 'cnn' }) {
  const { t } = useI18n()
  const [tab, setTab] = useState(initialTab)
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('torch.kicker')}</div><h1>{t('torch.title')}</h1></div></div>
      <div className="info-box">{t('torch.numpyPreserved')}</div>
      <Tabs value={tab} onChange={(value) => setTab(value as 'mlp' | 'cnn')} tabs={[{ id: 'mlp', label: `🔥 ${t('torch.pytorchTab')}` }, { id: 'cnn', label: `👁️ ${t('torch.cnnTab')}` }]} />
      {tab === 'mlp' ? <MlpLab /> : <CnnLab />}
    </div>
  )
}
