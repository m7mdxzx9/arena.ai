import { useState } from 'react'
import { Btn, Card, Tabs, go } from '../ui'
import { Widget } from '../widgets'

export default function LanguageLab() {
  const [tab, setTab] = useState('tokens')
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">Language Intelligence Center · Tokenizer Press</div><h1 style={{ margin: 0 }}>Language Lab</h1></div><div className="spacer" /><Btn kind="ghost" onClick={() => go('/workbench/sentiment')}>Train a sentiment classifier →</Btn></div>
      <p className="muted">How do machines read? Text → tokens → vectors → attention → next-word probabilities. Each tool below is a small but real implementation trained on the campus corpus.</p>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'tokens', label: '🔤 Tokenizer' }, { id: 'embed', label: '🌌 Embeddings' }, { id: 'attn', label: '👀 Attention' }, { id: 'lm', label: '🗣️ Language model' }]} />
      <Card>
        {tab === 'tokens' && <Widget name="tokenizer" />}
        {tab === 'embed' && <Widget name="embedding_map" />}
        {tab === 'attn' && <Widget name="attention_heatmap" />}
        {tab === 'lm' && <Widget name="ngram_lm" />}
      </Card>
    </div>
  )
}
