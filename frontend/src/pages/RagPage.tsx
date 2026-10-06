import { useState } from 'react'
import RagLab from '../labs/RagLab'
import { Card, Tabs } from '../ui'
import { Widget } from '../widgets'

export default function RagPage() {
  const [tab, setTab] = useState('retrieval')
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">RAG Archives · Vector Vault</div><h1 style={{ margin: 0 }}>RAG Lab</h1></div></div>
      <p className="muted">Retrieval-Augmented Generation: find the relevant passages first, then answer <i>only</i> from them. Everything here runs on 14 campus documents with real embeddings, real BM25 and automatic answer checking.</p>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'retrieval', label: '🔎 Retrieval' }, { id: 'answers', label: '💬 Answering & hallucination' }, { id: 'explore', label: '🗺️ Chunks & embeddings' }]} />
      {tab === 'retrieval' && <RagLab key="r" context="rag_lab" />}
      {tab === 'answers' && <RagLab key="a" answers context="rag_lab" />}
      {tab === 'explore' && <div className="grid g2"><Card title="Chunking" icon="✂️"><Widget name="chunker" /></Card><Card title="Embedding space" icon="🌌"><Widget name="embedding_map" /></Card></div>}
    </div>
  )
}
