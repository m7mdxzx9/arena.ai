import { useState } from 'react'
import RagLab from '../labs/RagLab'
import { useI18n } from '../i18n'
import { Card, Tabs } from '../ui'
import { Widget } from '../widgets'

export default function RagPage() {
  const { t } = useI18n()
  const [tab, setTab] = useState('retrieval')
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('ragSimulation.pageKicker')}</div><h1 style={{ margin: 0 }}>{t('ragSimulation.pageTitle')}</h1></div></div>
      <p className="muted">{t('ragSimulation.pageSubtitle')}</p>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'retrieval', label: t('ragSimulation.tabRetrieval') }, { id: 'answers', label: t('ragSimulation.tabAnswers') }, { id: 'explore', label: t('ragSimulation.tabExplore') }]} />
      {tab === 'retrieval' && <RagLab key="r" context="rag_lab" />}
      {tab === 'answers' && <RagLab key="a" answers context="rag_lab" />}
      {tab === 'explore' && <div className="grid g2"><Card title={t('ragSimulation.chunking')} icon="✂️"><Widget name="chunker" /></Card><Card title={t('ragSimulation.vectorSpace')} icon="🌌"><Widget name="embedding_map" /></Card></div>}
    </div>
  )
}
