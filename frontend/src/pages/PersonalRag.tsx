import { useState } from 'react'
import { del, get, post, upload, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Rich, Select, Slider, Tabs, useApi, useGame } from '../ui'

const DEFAULT_SENTENCE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
const DEFAULT_OLLAMA_EMBEDDING = 'nomic-embed-text'
const DEFAULT_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
const DEFAULT_CASES = `[
  {
    "question": "What does the uploaded document say about its main topic?",
    "expected_document": "replace-with-document-name",
    "reference_answer": "",
    "relevant_chunk_ids": [],
    "tags": ["smoke-test"],
    "difficulty": "beginner"
  }
]`

export default function PersonalRagPage() {
  const { t, language } = useI18n()
  const { pid, ov } = useGame()
  const documents = useApi<Any[]>(`/api/p/${pid}/documents`)
  const models = useApi<Any>('/api/models')
  const providers = useApi<Any>('/api/rag/embedding-providers')
  const experiments = useApi<Any[]>(`/api/p/${pid}/rag/experiments`)
  const evaluationSets = useApi<Any[]>(`/api/p/${pid}/rag/evaluation-datasets`)

  const [tab, setTab] = useState('setup')
  const [file, setFile] = useState<File | null>(null)
  const [uploadName, setUploadName] = useState('')
  const [uploading, setUploading] = useState(false)
  const [reindexing, setReindexing] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [method, setMethod] = useState('hybrid')
  const [topK, setTopK] = useState(5)
  const [chunkSize, setChunkSize] = useState(180)
  const [overlap, setOverlap] = useState(30)
  const [embeddingProvider, setEmbeddingProvider] = useState('statistical_lsa')
  const [embeddingModel, setEmbeddingModel] = useState('')
  const [reranker, setReranker] = useState('none')
  const [rerankerModel, setRerankerModel] = useState(DEFAULT_RERANKER)
  const [rerankingDepth, setRerankingDepth] = useState(20)
  const [contextBudget, setContextBudget] = useState(1200)
  const [rrfK, setRrfK] = useState(60)
  const [denseWeight, setDenseWeight] = useState(1)
  const [lexicalWeight, setLexicalWeight] = useState(1)
  const [generation, setGeneration] = useState('extractive')
  const [generationModel, setGenerationModel] = useState(ov.player.settings.default_model || '')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<Any>(null)
  const [error, setError] = useState<unknown | null>(null)
  const [notice, setNotice] = useState('')
  const [inspected, setInspected] = useState<Any>(null)
  const [selectedExperimentIds, setSelectedExperimentIds] = useState<string[]>([])
  const [comparison, setComparison] = useState<Any>(null)
  const [compareBusy, setCompareBusy] = useState(false)
  const [datasetName, setDatasetName] = useState('RAG retrieval checks')
  const [casesJson, setCasesJson] = useState(DEFAULT_CASES)
  const [selectedDatasetId, setSelectedDatasetId] = useState('')
  const [evaluation, setEvaluation] = useState<Any>(null)
  const [evaluationBusy, setEvaluationBusy] = useState(false)

  const modelList: Any[] = models.data?.models || []
  const activeGenerationModel = generationModel || modelList[0]?.name || ''
  const providerList: Any[] = providers.data?.providers || []
  const sentenceProvider = providerList.find((item) => item.provider === 'sentence-transformers')
  const ollamaProvider = providerList.find((item) => item.provider === 'ollama')
  const canUseSentence = Boolean(sentenceProvider?.available)
  const canUseOllamaEmbedding = Boolean(ollamaProvider?.available)

  const embeddingModelValue = embeddingModel || (embeddingProvider === 'ollama' ? DEFAULT_OLLAMA_EMBEDDING : DEFAULT_SENTENCE_MODEL)
  const changeEmbeddingProvider = (provider: string) => {
    setEmbeddingProvider(provider)
    setEmbeddingModel('')
    if (provider === 'none') setMethod('bm25')
  }
  const changeChunkSize = (size: number) => { setChunkSize(size); setOverlap((value) => Math.min(value, size - 10)) }
  const apiSettings = () => ({
    method,
    top_k: topK,
    embedding_provider: embeddingProvider,
    embedding_model: ['sentence-transformers', 'ollama'].includes(embeddingProvider) ? embeddingModelValue : null,
    reranker,
    reranking_depth: rerankingDepth,
    reranker_model: reranker === 'cross_encoder' ? rerankerModel : null,
    rrf_k: rrfK,
    dense_weight: denseWeight,
    lexical_weight: lexicalWeight,
    context_budget: contextBudget,
  })

  const add = async () => {
    if (!file) return
    setUploading(true); setError(null); setNotice('')
    try {
      const indexed = await upload(`/api/p/${pid}/documents`, file, {
        name: uploadName.trim(),
        chunk_size: chunkSize,
        overlap,
        embedding_provider: embeddingProvider,
        embedding_model: ['sentence-transformers', 'ollama'].includes(embeddingProvider) ? embeddingModelValue : '',
      })
      setFile(null); setUploadName(''); documents.reload(); setInspected(indexed)
      if (indexed.index_error) setError({ code: indexed.index_error.code, message: indexed.index_error.message })
      else setNotice(t('advancedRag.ingestionComplete'))
    } catch (e) { setError(e) } finally { setUploading(false) }
  }

  const remove = async (id: string) => {
    setError(null); setNotice('')
    try {
      await del(`/api/p/${pid}/documents/${id}`)
      documents.reload()
      if (inspected?.id === id) setInspected(null)
      setNotice(t('common.delete') + ' ✓')
    } catch (e) { setError(e) }
  }

  const inspect = async (id: string) => {
    setError(null)
    try { setInspected(await get(`/api/p/${pid}/documents/${id}`)) } catch (e) { setError(e) }
  }

  const reindex = async (id: string) => {
    setReindexing(id); setError(null); setNotice('')
    try {
      const refreshed = await post(`/api/p/${pid}/documents/${id}/reindex`, {
        chunk_size: chunkSize,
        overlap,
        embedding_provider: embeddingProvider,
        embedding_model: ['sentence-transformers', 'ollama'].includes(embeddingProvider) ? embeddingModelValue : null,
      })
      setInspected(refreshed); documents.reload()
      if (refreshed.index_error) setError({ code: refreshed.index_error.code, message: refreshed.index_error.message })
      else setNotice(t('advancedRag.reindexed'))
    } catch (e) { setError(e) } finally { setReindexing(null) }
  }

  const query = async () => {
    if (!question.trim()) return
    setBusy(true); setError(null); setResult(null); setNotice('')
    try {
      const value = await post(`/api/p/${pid}/personal-rag/query`, {
        question,
        document_ids: (documents.data || []).map((document: Any) => document.id),
        ...apiSettings(),
        generation,
        model: generation === 'ollama' ? activeGenerationModel : null,
        language,
      })
      setResult(value); experiments.reload()
    } catch (e) { setError(e) } finally { setBusy(false) }
  }

  const compareExperiments = async () => {
    setCompareBusy(true); setError(null); setComparison(null)
    try { setComparison(await post(`/api/p/${pid}/rag/experiments/compare`, { experiment_ids: selectedExperimentIds })) }
    catch (e) { setError(e) } finally { setCompareBusy(false) }
  }

  const createEvaluationSet = async () => {
    setError(null); setEvaluation(null)
    try {
      const cases = JSON.parse(casesJson)
      if (!Array.isArray(cases) || !cases.length) throw new Error('Enter a JSON array containing at least one evaluation case.')
      const created = await post(`/api/p/${pid}/rag/evaluation-datasets`, { name: datasetName, cases })
      setSelectedDatasetId(created.id); evaluationSets.reload(); setNotice(t('advancedRag.saved'))
    } catch (e) { setError(e) }
  }

  const runEvaluation = async () => {
    if (!selectedDatasetId) return
    setEvaluationBusy(true); setError(null); setEvaluation(null)
    try {
      setEvaluation(await post(`/api/p/${pid}/rag/evaluation-datasets/${selectedDatasetId}/run`, { config: apiSettings() }))
    } catch (e) { setError(e) } finally { setEvaluationBusy(false) }
  }

  const toggleExperiment = (id: string) => setSelectedExperimentIds((ids) => ids.includes(id) ? ids.filter((item) => item !== id) : ids.length < 8 ? [...ids, id] : ids)

  const hasNeuralModel = embeddingProvider === 'sentence-transformers' ? canUseSentence : embeddingProvider === 'ollama' ? canUseOllamaEmbedding : false
  const showEmbeddingModel = ['sentence-transformers', 'ollama'].includes(embeddingProvider)

  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('personalRag.kicker')}</div><h1>{t('advancedRag.title')}</h1></div></div>
      <p className="muted">{t('advancedRag.subtitle')}</p>
      <div className="info-box">🔒 {t('advancedRag.localOnly')}<br /><small>{t('advancedRag.pipeline')}</small></div>
      <div className="row wrap"><a className="btn btn-ghost btn-sm" href="#/rag">{t('advancedRag.openLearning')}</a><a className="btn btn-ghost btn-sm" href="#/tutor">{t('advancedRag.openTutor')}</a></div>
      <Tabs value={tab} onChange={setTab} tabs={[
        { id: 'setup', label: `📚 ${t('personalRag.documents')}` },
        { id: 'query', label: `🔎 ${t('advancedRag.run')}` },
        { id: 'experiments', label: `🧪 ${t('advancedRag.experiments')}` },
        { id: 'evaluation', label: `📏 ${t('advancedRag.evaluation')}` },
      ]} />
      <ErrorBox error={error || documents.error || providers.error || experiments.error || evaluationSets.error} />
      {notice && <div className="ok-box" role="status">{notice}</div>}

      {tab === 'setup' && <div className="grid g-side">
        <div className="col">
          <Card title={t('personalRag.upload')} icon="⬆️">
            <p className="muted"><small>{t('personalRag.formats')}</small></p>
            <label className="field"><span className="field-l">{t('advancedRag.embeddingProvider')}</span>
<Select value={embeddingProvider} onChange={changeEmbeddingProvider} options={[
                { value: 'statistical_lsa', label: t('advancedRag.statistical') },
                { value: 'sentence-transformers', label: t('advancedRag.sentenceTransformers') },
                { value: 'ollama', label: t('advancedRag.ollamaEmbedding') },
              ]} />
            </label>
            {showEmbeddingModel && <label className="field"><span className="field-l">{t('advancedRag.embeddingModel')}</span>
              <input dir="ltr" value={embeddingModelValue} onChange={(event) => setEmbeddingModel(event.target.value)} placeholder={embeddingProvider === 'ollama' ? DEFAULT_OLLAMA_EMBEDDING : DEFAULT_SENTENCE_MODEL} />
            </label>}
            {embeddingProvider === 'sentence-transformers' && <small className={canUseSentence ? 'success-text' : 'muted'}>{canUseSentence ? t('advancedRag.modelCacheNotice') : t('advancedRag.neuralUnavailable')} <Ltr>Sentence Transformers</Ltr> · {t('advancedRag.localOnly')}</small>}
            {embeddingProvider === 'ollama' && <small className={canUseOllamaEmbedding ? 'success-text' : 'muted'}>{canUseOllamaEmbedding ? t('advancedRag.neuralAvailable') : t('advancedRag.neuralUnavailable')} <Ltr>Ollama</Ltr> · {t('advancedRag.localOnly')}</small>}
            <Slider label={t('advancedRag.chunkSize')} value={chunkSize} min={40} max={500} step={10} onChange={changeChunkSize} />
            <Slider label={t('advancedRag.overlap')} value={overlap} min={0} max={Math.min(200, chunkSize - 10)} step={5} onChange={setOverlap} />
            <label className="field"><span className="field-l">{t('common.title')}</span><input value={uploadName} onChange={(event) => setUploadName(event.target.value)} maxLength={100} placeholder={file?.name || ''} /></label>
            <input type="file" accept=".pdf,.txt,.md,.markdown,.docx" onChange={(event) => setFile(event.target.files?.[0] || null)} />
            {file && <div><Ltr>{file.name}</Ltr></div>}
            <Btn onClick={add} disabled={!file || uploading || (showEmbeddingModel && !embeddingModelValue.trim())}>{uploading ? t('advancedRag.ingestionProgress') : t('personalRag.upload')}</Btn>
          </Card>
          <Card title={t('advancedRag.embeddingStatus')} icon="🧠">
            <div className="stack-sm">
              {providerList.map((provider: Any) => <div className="document-row" key={provider.provider}>
                <div><b><Ltr>{provider.provider}</Ltr></b><br /><small>{provider.note} {provider.model && <Ltr>{provider.model}</Ltr>}</small></div>
                <Pill kind={provider.available ? 'green' : 'amber'}>{provider.available ? t('common.available') : t('common.unavailable')}</Pill>
              </div>)}
            </div>
            <small className="muted">{t('advancedRag.statistical')}</small>
          </Card>
        </div>
        <div className="col">
          <Card title={t('advancedRag.title')} icon="🗂️" right={<Pill>{documents.data?.length || 0}</Pill>}>
            {documents.loading && <Loading />}
            {documents.data?.length === 0 && <p className="muted">{t('advancedRag.noDocuments')}</p>}
            {documents.data?.map((document: Any) => <div className="document-row" key={document.id}>
              <div className="document-row-main">
                <b>{document.name}</b><br />
                <small><Ltr>{document.format.toUpperCase()}</Ltr> · {document.chunks} {t('advancedRag.chunks')} · {(document.size_bytes / 1024).toFixed(1)} KB · {document.embedding_status}</small>
                {document.embedding_error && <small className="error-text">{document.embedding_error}</small>}
              </div>
              <div className="row wrap">
                <Btn small kind="ghost" onClick={() => inspect(document.id)}>{t('advancedRag.inspect')}</Btn>
                <Btn small kind="ghost" onClick={() => reindex(document.id)} disabled={reindexing === document.id}>{reindexing === document.id ? t('advancedRag.reindexing') : t('advancedRag.reindex')}</Btn>
                <Btn small kind="danger" onClick={() => remove(document.id)}>{t('common.delete')}</Btn>
              </div>
            </div>)}
          </Card>
          {inspected && <Card title={t('advancedRag.chunkPreview')} icon="🔬" right={<Pill>{inspected.chunks} {t('advancedRag.chunks')}</Pill>}>
            <div className="kv">
              <b>{t('advancedRag.indexStatus')}</b><span><Ltr>{inspected.embedding_status || 'lexical_ready'}</Ltr></span>
              <b>{t('advancedRag.chunkSettings')}</b><span><Ltr>{inspected.chunk_size}</Ltr> / <Ltr>{inspected.overlap}</Ltr></span>
              <b>{t('advancedRag.embeddingProvider')}</b><span><Ltr>{inspected.embedding_provider || t('advancedRag.lexicalReady')}</Ltr></span>
              {inspected.embedding_model && <><b>{t('advancedRag.embeddingModel')}</b><span><Ltr>{inspected.embedding_model}</Ltr></span></>}
              {inspected.page_count && <><b>{t('advancedRag.page')}</b><span>{inspected.page_count}</span></>}
            </div>
            {inspected.normalized_text_preview && <details><summary>{t('advancedRag.normalizedPreview')}</summary><pre className="evidence-pre">{inspected.normalized_text_preview}</pre></details>}
            <div className="chunk-list">{inspected.chunks_preview?.map((chunk: Any) => <div className="retrieved-chunk" key={chunk.id}>
              <div className="row between"><b><Ltr>{chunk.id}</Ltr></b><small>{chunk.page ? `${t('advancedRag.page')} ${chunk.page}` : t('advancedRag.noPage')}</small></div>
              <p>{chunk.text}</p>{chunk.possible_prompt_injection && <Pill kind="red">⚠ {t('personalRag.injection')}</Pill>}
            </div>)}</div>
          </Card>}
        </div>
      </div>}

      {tab === 'query' && <div className="grid g-side">
        <div className="col">
          <Card title={t('advancedRag.retrieval')} icon="🎛️">
            <label className="field"><span className="field-l">{t('advancedRag.embeddingProvider')}</span><Select value={embeddingProvider} onChange={changeEmbeddingProvider} options={[
              { value: 'statistical_lsa', label: t('advancedRag.statistical') },
              { value: 'sentence-transformers', label: t('advancedRag.sentenceTransformers') },
              { value: 'ollama', label: t('advancedRag.ollamaEmbedding') },
              { value: 'none', label: `${t('advancedRag.lexical')} · BM25 only` },
            ]} /></label>
            {showEmbeddingModel && <label className="field"><span className="field-l">{t('advancedRag.embeddingModel')}</span><input dir="ltr" value={embeddingModelValue} onChange={(event) => setEmbeddingModel(event.target.value)} /></label>}
            {showEmbeddingModel && !hasNeuralModel && <small className="muted">{t('advancedRag.neuralUnavailable')}</small>}
            {embeddingProvider === 'sentence-transformers' && canUseSentence && <small className="muted">{t('advancedRag.modelCacheNotice')}</small>}
            <label className="field"><span className="field-l">{t('advancedRag.retrieval')}</span><Select value={method} onChange={setMethod} options={embeddingProvider === 'none'
              ? [{ value: 'bm25', label: t('advancedRag.lexical') }]
              : [{ value: 'dense', label: t('advancedRag.dense') }, { value: 'bm25', label: t('advancedRag.lexical') }, { value: 'hybrid', label: t('advancedRag.hybrid') }]} /></label>
            <Slider label={t('advancedRag.topK')} value={topK} min={1} max={20} onChange={setTopK} />
            <Slider label={t('advancedRag.contextBudget')} value={contextBudget} min={30} max={4000} step={50} onChange={setContextBudget} />
            {method === 'hybrid' && <>
              <Slider label={t('advancedRag.rrfK')} value={rrfK} min={1} max={200} onChange={setRrfK} />
              <Slider label={t('advancedRag.denseWeight')} value={denseWeight} min={0} max={3} step={0.1} onChange={setDenseWeight} fmt={(value) => value.toFixed(1)} />
              <Slider label={t('advancedRag.lexicalWeight')} value={lexicalWeight} min={0} max={3} step={0.1} onChange={setLexicalWeight} fmt={(value) => value.toFixed(1)} />
            </>}
            <label className="field"><span className="field-l">{t('advancedRag.reranker')}</span><Select value={reranker} onChange={setReranker} options={[
              { value: 'none', label: t('advancedRag.noReranker') }, { value: 'cross_encoder', label: t('advancedRag.neuralReranker') }, { value: 'lexical_overlap', label: t('advancedRag.heuristicReranker') },
            ]} /></label>
            {reranker === 'cross_encoder' && <>
              <label className="field"><span className="field-l">{t('advancedRag.rerankerModel')}</span><input dir="ltr" value={rerankerModel} onChange={(event) => setRerankerModel(event.target.value)} /></label>
              {!canUseSentence ? <small className="muted">{t('errors.reranker_dependency_missing')}</small> : <small className="muted">{t('advancedRag.modelCacheNotice')}</small>}
            </>}
            {reranker !== 'none' && <Slider label={t('advancedRag.rerankerDepth')} value={rerankingDepth} min={1} max={100} onChange={setRerankingDepth} />}
            <label className="field"><span className="field-l">{t('advancedRag.answerMode')}</span><Select value={generation} onChange={setGeneration} options={[
              { value: 'extractive', label: t('advancedRag.extractive') }, { value: 'ollama', label: t('advancedRag.localGeneration') },
            ]} /></label>
            {generation === 'ollama' && <Select label={t('common.model')} value={activeGenerationModel} onChange={setGenerationModel} options={modelList.map((item: Any) => ({ value: item.name, label: item.name }))} />}
            {generation === 'ollama' && !models.data?.reachable && <div className="warn-box">{t('models.ollamaUnavailable')}</div>}
          </Card>
          <Card title={t('personalRag.question')} icon="🔎">
            <textarea rows={5} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={t('personalRag.question')} />
            <Btn onClick={query} disabled={busy || !question.trim() || !documents.data?.length || (generation === 'ollama' && !activeGenerationModel) || (showEmbeddingModel && !embeddingModelValue.trim())}>
              {busy ? t('advancedRag.running') : t('advancedRag.ask')}
            </Btn>
          </Card>
        </div>
        <div className="col">
          {busy && <Loading text={t('advancedRag.running')} />}
          {result && <>
            <Card title={t('advancedRag.answer')} icon="💬" right={<Pill kind={result.generation_mode === 'extractive_fallback' ? 'amber' : 'violet'}><Ltr>{result.generation_mode}</Ltr></Pill>}>
              <Rich text={result.answer} />
              <div className="row wrap" style={{ marginTop: 12 }}>
                <Pill><Ltr>{result.retrieval_method}</Ltr></Pill><Pill>{t('advancedRag.latency')}: {result.latency_ms} ms</Pill>
                <Pill>{t('advancedRag.citations')}: {result.citations?.length || 0}</Pill>
                <Pill>{t('advancedRag.contextBudget')}: {result.context_words} / {contextBudget}</Pill>
              </div>
              {result.citation_details?.length > 0 && <div className="citation-list"><h4>{t('advancedRag.citations')}</h4>{result.citation_details.map((citation: Any) => <div className="citation-card" key={citation.chunk_id}>
                <div className="row between"><b>{citation.document}{citation.page ? ` · ${t('advancedRag.page')} ${citation.page}` : ''}</b><Pill><Ltr>{citation.chunk_id}</Ltr></Pill></div>
                <p>{citation.excerpt}</p>
              </div>)}</div>}
            </Card>
            <Card title={t('advancedRag.evidence')} icon="🧾">
              <div className="chunk-list">{result.retrieved?.map((chunk: Any) => <div className="retrieved-chunk" key={chunk.id}>
                <div className="row between"><b><Ltr>{chunk.id}</Ltr></b><span className="row wrap">
                  <Pill>{t('advancedRag.score')} {chunk.score}</Pill>
                  <Pill>{t('advancedRag.firstRank')} {chunk.first_rank}</Pill>
                  {chunk.reranked_position && <Pill>{t('advancedRag.rerankedPosition')} {chunk.reranked_position}</Pill>}
                  {chunk.possible_prompt_injection && <Pill kind="red">⚠ {t('personalRag.injection')}</Pill>}
                </span></div>
                <small className="muted">{chunk.document_name} · {chunk.page ? `${t('advancedRag.page')} ${chunk.page}` : t('advancedRag.noPage')} · <Ltr>{chunk.embedding}</Ltr> · dense {chunk.dense_score} · BM25 {chunk.bm25_score}</small>
                <p>{chunk.text}</p><small className="untrusted-label">⚠ {t('personalRag.untrusted')}</small>
              </div>)}</div>
            </Card>
            <Card title={t('advancedRag.selectedContext')} icon="🧩">
              <p className="muted">{t('advancedRag.contextBudget')} · {result.context_words || 0} words</p>
              <pre className="evidence-pre">{result.selected_context || ''}</pre>
              <div className="row wrap">{result.included_chunks?.map((item: Any) => <Pill key={item.id}><Ltr>{item.id}</Ltr>{item.truncated ? ' · …' : ''}</Pill>)}</div>
            </Card>
          </>}
          {!result && !busy && <Card><div className="empty">{t('advancedRag.evidence')}</div></Card>}
        </div>
      </div>}

      {tab === 'experiments' && <div className="stack">
        <div className="row between wrap"><div><h2>{t('advancedRag.experiments')}</h2><p className="muted">{t('advancedRag.saved')} · {t('advancedRag.localOnly')}</p></div>
          <Btn onClick={compareExperiments} disabled={compareBusy || selectedExperimentIds.length < 2}>{compareBusy ? t('common.loading') : t('advancedRag.compare')} ({selectedExperimentIds.length})</Btn>
        </div>
        {experiments.data?.length ? <div className="table-wrap"><table><thead><tr><th></th><th>{t('advancedRag.experiment')}</th><th>{t('personalRag.question')}</th><th>{t('advancedRag.retrieval')}</th><th>{t('advancedRag.retrievedCount')}</th><th>{t('advancedRag.citationCount')}</th><th>{t('advancedRag.latency')}</th></tr></thead><tbody>
          {experiments.data.map((item: Any) => <tr key={item.id}>
            <td><input type="checkbox" aria-label={`${t('advancedRag.experiment')} ${item.query}`} checked={selectedExperimentIds.includes(item.id)} onChange={() => toggleExperiment(item.id)} /></td>
            <td><Ltr>{item.id.slice(0, 8)}</Ltr></td><td>{item.query}</td><td><Ltr>{item.config?.method} · {item.config?.embedding_provider}</Ltr></td>
            <td>{item.metrics?.retrieved_chunks ?? item.retrieved?.length ?? 0}</td><td>{item.metrics?.citation_count ?? item.citations?.length ?? 0}</td><td>{item.latency_ms} ms</td>
          </tr>)}
        </tbody></table></div> : <Card><p className="muted">{t('advancedRag.noExperiments')}</p></Card>}
        {comparison && <Card title={t('advancedRag.compare')} icon="📊"><div className="stack-sm">{comparison.experiments?.map((item: Any) => <div className="retrieved-chunk" key={item.id}>
          <div className="row between"><b>{item.query}</b><Pill><Ltr>{item.config.method} · {item.config.embedding_provider} · {item.config.reranker}</Ltr></Pill></div>
          <div className="row wrap"><Pill>{t('advancedRag.retrievedCount')}: {item.metrics?.retrieved_chunks}</Pill><Pill>{t('advancedRag.citationCount')}: {item.metrics?.citation_count}</Pill><Pill>{t('advancedRag.latency')}: {item.metrics?.latency_ms} ms</Pill></div>
          <Rich text={item.answer} /><small><Ltr>{item.retrieved?.map((chunk: Any) => `${chunk.rank}. ${chunk.document_name}#${chunk.chunk_index}`).join(' · ')}</Ltr></small>
        </div>)}</div></Card>}
      </div>}

      {tab === 'evaluation' && <div className="grid g-side">
        <div className="col">
          <Card title={t('advancedRag.evaluation')} icon="📏">
            <label className="field"><span className="field-l">{t('advancedRag.datasetName')}</span><input value={datasetName} onChange={(event) => setDatasetName(event.target.value)} maxLength={100} /></label>
            <label className="field"><span className="field-l">{t('advancedRag.datasetCases')}</span><textarea className="code-input" dir="ltr" rows={13} value={casesJson} onChange={(event) => setCasesJson(event.target.value)} spellCheck={false} /></label>
            <small className="muted">{t('advancedRag.noGroundTruth')}</small>
            <Btn onClick={createEvaluationSet} disabled={!datasetName.trim()}>{t('advancedRag.createDataset')}</Btn>
          </Card>
          <Card title={t('advancedRag.evaluation')} icon="🧪">
            <Select label={t('advancedRag.datasetName')} value={selectedDatasetId} onChange={setSelectedDatasetId} options={[
              { value: '', label: t('advancedRag.datasetName') }, ...(evaluationSets.data || []).map((item: Any) => ({ value: item.id, label: `${item.name} (${item.case_count})` })),
            ]} />
            <p className="muted">{t('advancedRag.retrieval')} · <Ltr>{method} / {embeddingProvider}</Ltr> · Top-K {topK}</p>
            <Btn onClick={runEvaluation} disabled={!selectedDatasetId || evaluationBusy}>{evaluationBusy ? t('advancedRag.running') : t('advancedRag.runEvaluation')}</Btn>
          </Card>
        </div>
        <div className="col">
          {evaluationBusy && <Loading text={t('advancedRag.running')} />}
          {evaluation && <>
            <Card title={t('advancedRag.evaluation')} icon="📊" right={<Pill>{evaluation.n_cases} cases</Pill>}>
              <p className="muted">{t('advancedRag.noGroundTruth')}</p>
              <div className="metric-grid">
                <Metric label={t('advancedRag.hitRate')} value={evaluation.metrics?.hit_rate} />
                <Metric label={t('advancedRag.precisionAtK')} value={evaluation.metrics?.precision_at_k} />
                <Metric label={t('advancedRag.recallAtK')} value={evaluation.metrics?.recall_at_k} />
                <Metric label={t('advancedRag.mrr')} value={evaluation.metrics?.mrr} />
                <Metric label={t('advancedRag.ndcg')} value={evaluation.metrics?.ndcg_at_k} />
                <Metric label={t('advancedRag.exactMatch')} value={evaluation.metrics?.answer_exact_match} />
                <Metric label={t('advancedRag.containsReference')} value={evaluation.metrics?.answer_contains_reference} />
                <Metric label={t('advancedRag.citationPresence')} value={evaluation.metrics?.citation_presence} />
              </div>
              <small className="muted">Retrieval cases: {evaluation.metrics?.retrieval_cases ?? 0} · answer/reference cases: {evaluation.metrics?.answer_cases ?? 0}</small>
            </Card>
            <Card title={t('advancedRag.evidence')} icon="📝"><div className="stack-sm">{evaluation.rows?.map((row: Any, index: number) => <div className="retrieved-chunk" key={`${row.question}-${index}`}>
              <b>{row.question}</b><div className="row wrap"><Pill>{t('advancedRag.hitRate')}: {row.hit_rank ? `#${row.hit_rank}` : '—'}</Pill><Pill>{t('advancedRag.citationPresence')}: {row.citation_present ? '✓' : '—'}</Pill>{row.difficulty && <Pill>{row.difficulty}</Pill>}</div>
              {row.tags?.length > 0 && <small>{row.tags.join(', ')}</small>}
              <Rich text={row.answer} />
              {row.retrieved?.map((chunk: Any) => <small key={chunk.id} className="muted"><Ltr>{chunk.rank}. {chunk.document}#{chunk.id.split('#').at(-1)} · {chunk.score}</Ltr></small>)}
            </div>)}</div></Card>
          </>}
          {!evaluation && !evaluationBusy && <Card><div className="empty">{t('advancedRag.noGroundTruth')}</div></Card>}
        </div>
      </div>}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: number | null | undefined }) {
  const formatted = typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'
  return <div className="metric-card"><span>{label}</span><strong>{formatted}</strong></div>
}
