import { useState } from 'react'
import { del, post, upload, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Select, Slider, useApi, useGame } from '../ui'

export default function PersonalRagPage() {
  const { t } = useI18n()
  const { pid, ov } = useGame()
  const documents = useApi<Any[]>(`/api/p/${pid}/documents`)
  const models = useApi<Any>('/api/models')
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [question, setQuestion] = useState('')
  const [method, setMethod] = useState('hybrid')
  const [topK, setTopK] = useState(5)
  const [generation, setGeneration] = useState('extractive')
  const [model, setModel] = useState(ov.player.settings.default_model || '')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<Any>(null)
  const [error, setError] = useState<string | null>(null)
  const modelList = models.data?.models || []
  const activeModel = model || modelList[0]?.name || ''
  const add = async () => {
    if (!file) return
    setUploading(true); setError(null)
    try { await upload(`/api/p/${pid}/documents`, file, { chunk_size: 180, overlap: 30 }); setFile(null); documents.reload() } catch (e: Any) { setError(e.message) } finally { setUploading(false) }
  }
  const remove = async (id: string) => {
    try { await del(`/api/p/${pid}/documents/${id}`); documents.reload() } catch (e: Any) { setError(e.message) }
  }
  const query = async () => {
    setBusy(true); setError(null); setResult(null)
    try { setResult(await post(`/api/p/${pid}/personal-rag/query`, { question, method, top_k: topK, generation, model: generation === 'ollama' ? activeModel : null })) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('personalRag.kicker')}</div><h1>{t('personalRag.title')}</h1></div></div>
      <p className="muted">{t('personalRag.subtitle')}</p>
      <div className="grid g-side">
        <div className="col">
          <Card title={t('personalRag.upload')} icon="⬆️">
            <p className="muted"><small>{t('personalRag.formats')}</small></p>
            <input type="file" accept=".pdf,.txt,.md,.markdown,.docx" onChange={(event) => setFile(event.target.files?.[0] || null)} />
            {file && <div><Ltr>{file.name}</Ltr></div>}
            <Btn onClick={add} disabled={!file || uploading}>{uploading ? t('datasets.uploading') : t('common.upload')}</Btn>
          </Card>
          <Card title={t('personalRag.documents')} icon="📚">
            {documents.loading && <Loading />}
            {documents.data?.length === 0 && <p className="muted">{t('personalRag.noDocuments')}</p>}
            {documents.data?.map((document) => <div className="document-row" key={document.id}><div><b><Ltr>{document.name}</Ltr></b><br /><small>{document.format.toUpperCase()} · {document.chunks} chunks · {(document.size_bytes / 1024).toFixed(1)} KB</small></div><Btn small kind="danger" onClick={() => remove(document.id)}>{t('common.delete')}</Btn></div>)}
          </Card>
          <Card title={t('personalRag.question')} icon="🔎">
            <textarea rows={5} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={t('personalRag.question')} />
            <Select label={t('personalRag.retrieval')} value={method} onChange={setMethod} options={[{ value: 'dense', label: 'Dense / LSA' }, { value: 'bm25', label: 'BM25' }, { value: 'hybrid', label: 'Hybrid RRF' }]} />
            <Slider label={t('personalRag.topK')} value={topK} min={1} max={10} onChange={setTopK} />
            <Select label={t('personalRag.generation')} value={generation} onChange={setGeneration} options={[{ value: 'extractive', label: t('personalRag.extractive') }, { value: 'ollama', label: t('personalRag.ollama') }]} />
            {generation === 'ollama' && <Select label={t('common.model')} value={activeModel} onChange={setModel} options={modelList.map((item: Any) => ({ value: item.name, label: item.name }))} />}
            <Btn onClick={query} disabled={busy || !question.trim() || !documents.data?.length || (generation === 'ollama' && !activeModel)}>{busy ? t('common.loading') : t('personalRag.ask')}</Btn>
          </Card>
        </div>
        <div className="col">
          <ErrorBox error={error || documents.error} />
          {busy && <Loading />}
          {result && <>
            <Card title={t('personalRag.generation')} icon="💬" right={<Pill kind={result.generation_mode === 'extractive_fallback' ? 'amber' : 'violet'}><Ltr>{result.generation_mode}</Ltr></Pill>}>
              <p>{result.answer}</p>
              <div className="row"><small>{t('personalRag.citations')}: </small>{result.citations?.length ? result.citations.map((citation: string) => <Pill key={citation}><Ltr>{citation}</Ltr></Pill>) : <Pill kind="red">0</Pill>}<small>{t('personalRag.latency')}: {result.latency_ms} ms</small></div>
            </Card>
            <Card title={t('personalRag.evidence')} icon="🧾">
              <div className="chunk-list">{result.retrieved.map((chunk: Any) => <div className="retrieved-chunk" key={chunk.id}>
                <div className="row between"><b><Ltr>{chunk.id}</Ltr></b><span><Pill>{t('personalRag.score')} {chunk.score}</Pill>{chunk.possible_prompt_injection && <Pill kind="red">⚠ {t('personalRag.injection')}</Pill>}</span></div>
                <small className="muted"><Ltr>{chunk.document_name}</Ltr> · dense {chunk.dense_score} · BM25 {chunk.bm25_score} · {chunk.embedding}</small>
                <p>{chunk.text}</p><small className="untrusted-label">⚠ {t('personalRag.untrusted')}</small>
              </div>)}</div>
            </Card>
          </>}
          {!result && !busy && <Card><div className="empty">{t('personalRag.evidence')}</div></Card>}
        </div>
      </div>
    </div>
  )
}
