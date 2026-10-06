import { useMemo, useState } from 'react'
import { post, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Select, Slider, Toggle, useApi, useGame } from '../ui'

function Trace({ events }: { events: Any[] }) {
  const { t } = useI18n()
  return <div className="agent-event-trace">{events.map((event, index) => <div key={index} className={`agent-event event-${event.event}`}>
    <div className="row between"><Pill kind={event.allowed === false ? 'red' : event.event === 'final' ? 'green' : 'cyan'}><Ltr>{event.event}</Ltr></Pill>{event.step && <small>step {event.step}</small>}{event.latency_ms != null && <small>{event.latency_ms} ms</small>}</div>
    {event.tool && <b><Ltr>{event.tool}</Ltr></b>}
    {event.event === 'permission' && <div>{event.allowed ? t('realAgent.permissionAllowed') : t('realAgent.permissionDenied')} · <Ltr>{event.permission || event.reason}</Ltr></div>}
    {event.arguments && <pre className="json-view" dir="ltr">{JSON.stringify(event.arguments, null, 2)}</pre>}
    {event.result && <pre className="json-view" dir="ltr">{JSON.stringify(event.result, null, 2)}</pre>}
    {event.answer && <p>{event.answer}</p>}
    {event.error && <div className="error-box">{event.error}</div>}
  </div>)}</div>
}

export default function RealAgentPage() {
  const { t } = useI18n()
  const { pid, ov } = useGame()
  const tools = useApi<Any[]>('/api/agent-tools')
  const models = useApi<Any>('/api/models')
  const configs = useApi<Any[]>(`/api/p/${pid}/agent-configurations`)
  const [selectedId, setSelectedId] = useState('')
  const [name, setName] = useState('Research Assistant')
  const [model, setModel] = useState(ov.player.settings.default_model || '')
  const [systemPrompt, setSystemPrompt] = useState('You are a careful AI learning assistant. Use tools only when needed and explain the result concisely.')
  const [selectedTools, setSelectedTools] = useState<string[]>(['calculator', 'statistics'])
  const [permissions, setPermissions] = useState<string[]>(['calculate'])
  const [maxSteps, setMaxSteps] = useState(6)
  const [request, setRequest] = useState('')
  const [busy, setBusy] = useState(false)
  const [run, setRun] = useState<Any>(null)
  const [arenaFirst, setArenaFirst] = useState('')
  const [arenaSecond, setArenaSecond] = useState('')
  const [arenaTasks, setArenaTasks] = useState('Calculate 12 * 8. || 96\nExplain overfitting in one sentence. || generalization')
  const [arenaResult, setArenaResult] = useState<Any>(null)
  const [error, setError] = useState<string | null>(null)
  const modelList = models.data?.models || []
  const activeModel = model || modelList[0]?.name || ''
  const selected = configs.data?.find((config) => config.id === selectedId) || configs.data?.[0]
  const firstArenaId = arenaFirst || configs.data?.[0]?.id || ''
  const secondArenaId = arenaSecond || configs.data?.find((config) => config.id !== firstArenaId)?.id || ''
  const permissionOptions = useMemo(() => Array.from(new Set((tools.data || []).map((tool) => tool.permission))), [tools.data])
  const toggle = (list: string[], value: string, update: (items: string[]) => void) => update(list.includes(value) ? list.filter((item) => item !== value) : [...list, value])
  const create = async () => {
    setBusy(true); setError(null)
    try {
      const config = await post(`/api/p/${pid}/agent-configurations`, { name, model: activeModel, system_prompt: systemPrompt, tools: selectedTools, permissions, max_steps: maxSteps, timeout_seconds: 60 })
      await configs.reload(); setSelectedId(config.id)
    } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const execute = async () => {
    if (!selected || !request.trim()) return
    setBusy(true); setError(null); setRun(null)
    try { setRun(await post(`/api/p/${pid}/agent-configurations/${selected.id}/run`, { request })) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const clearMemory = async () => {
    if (!selected) return
    await post(`/api/p/${pid}/agent-configurations/${selected.id}/clear-memory`)
    configs.reload()
  }
  const runArena = async () => {
    const tasks = arenaTasks.split('\n').map((line, index) => {
      const [prompt, expected = ''] = line.split('||', 2)
      return { id: `task-${index + 1}`, prompt: prompt.trim(), expected_contains: expected.split(',').map((value) => value.trim()).filter(Boolean) }
    }).filter((task) => task.prompt)
    setBusy(true); setError(null); setArenaResult(null)
    try { setArenaResult(await post(`/api/p/${pid}/agent-arena`, { configuration_ids: [firstArenaId, secondArenaId], tasks })) } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('realAgent.kicker')}</div><h1>{t('realAgent.title')}</h1></div></div>
      <p className="muted">{t('realAgent.subtitle')}</p>
      {!models.data?.reachable && <div className="warn-box">{t('realAgent.requiresOllama')}</div>}
      <div className="grid g-side">
        <div className="col">
          <Card title={t('realAgent.create')} icon="🤖">
            <label className="field"><span className="field-l">{t('realAgent.name')}</span><input value={name} onChange={(event) => setName(event.target.value)} /></label>
            <Select label={t('common.model')} value={activeModel} onChange={setModel} options={modelList.map((item: Any) => ({ value: item.name, label: item.name }))} />
            <label className="field"><span className="field-l">{t('realAgent.systemPrompt')}</span><textarea rows={5} value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} /></label>
            <div className="field-l">{t('realAgent.tools')}</div>
            {(tools.data || []).map((tool) => <div key={tool.name}><Toggle label={<><Ltr>{tool.name}</Ltr> — {tool.description}</>} checked={selectedTools.includes(tool.name)} onChange={() => toggle(selectedTools, tool.name, setSelectedTools)} /><small className="muted" style={{ display: 'block', paddingInlineStart: 28 }}><Ltr>{tool.permission}</Ltr></small></div>)}
            <div className="field-l">{t('realAgent.permissions')}</div>
            {permissionOptions.map((permission) => <Toggle key={permission} label={<Ltr>{permission}</Ltr>} checked={permissions.includes(permission)} onChange={() => toggle(permissions, permission, setPermissions)} />)}
            <Slider label={t('realAgent.maxSteps')} value={maxSteps} min={1} max={12} onChange={setMaxSteps} />
            <Btn onClick={create} disabled={busy || !activeModel}>{t('realAgent.create')}</Btn>
          </Card>
        </div>
        <div className="col">
          <Card title={t('realAgent.configs')} icon="🧰">
            {configs.loading && <Loading />}
            {!configs.data?.length && <p className="muted">{t('realAgent.noConfigs')}</p>}
            {configs.data?.length ? <Select value={selected?.id || ''} onChange={setSelectedId} options={configs.data.map((config) => ({ value: config.id, label: `${config.name} · ${config.model}` }))} /> : null}
            {selected && <>
              <div className="row"><Pill><Ltr>{selected.model}</Ltr></Pill><Pill>{selected.max_steps} steps</Pill>{selected.tools.map((tool: string) => <Pill key={tool} kind="cyan"><Ltr>{tool}</Ltr></Pill>)}</div>
              <div className="row between"><b>{t('realAgent.memory')} ({selected.memory.length}/20)</b><Btn small kind="ghost" onClick={clearMemory}>{t('realAgent.clearMemory')}</Btn></div>
              {selected.memory.length ? selected.memory.map((item: Any, index: number) => <div className="info-box" key={index}>{item.note}</div>) : <small className="muted">—</small>}
              <label className="field"><span className="field-l">{t('realAgent.request')}</span><textarea rows={4} value={request} onChange={(event) => setRequest(event.target.value)} /></label>
              <Btn onClick={execute} disabled={busy || !request.trim() || !models.data?.reachable}>{busy ? t('realAgent.running') : t('realAgent.run')}</Btn>
            </>}
          </Card>
          <ErrorBox error={error || configs.error || tools.error} />
          {busy && <Loading text={t('realAgent.running')} />}
          {run && <Card title={t('realAgent.trace')} icon="🔍" right={<><Pill kind={run.status === 'completed' ? 'green' : 'amber'}><Ltr>{run.status}</Ltr></Pill><small>{run.duration_ms} ms</small></>}><Trace events={run.trace} /></Card>}
        </div>
      </div>
      <Card title={t('realAgent.arena')} icon="⚔️">
        <p className="muted">{t('realAgent.arenaHelp')}</p>
        {(configs.data?.length || 0) < 2 ? <div className="warn-box">{t('realAgent.noConfigs')}</div> : <>
          <div className="grid g2"><Select label={t('realAgent.firstAgent')} value={firstArenaId} onChange={setArenaFirst} options={(configs.data || []).map((config) => ({ value: config.id, label: config.name }))} /><Select label={t('realAgent.secondAgent')} value={secondArenaId} onChange={setArenaSecond} options={(configs.data || []).filter((config) => config.id !== firstArenaId).map((config) => ({ value: config.id, label: config.name }))} /></div>
          <label className="field"><span className="field-l">{t('realAgent.tasks')}</span><textarea rows={5} value={arenaTasks} onChange={(event) => setArenaTasks(event.target.value)} /></label>
          <Btn onClick={runArena} disabled={busy || !models.data?.reachable || !firstArenaId || !secondArenaId || firstArenaId === secondArenaId}>{t('realAgent.runArena')}</Btn>
        </>}
        {arenaResult && <div className="table-wrap"><table><thead><tr><th>#</th><th>{t('realAgent.leaderboard')}</th><th>{t('evaluationLab.score')}</th><th>{t('realAgent.completion')}</th><th>{t('realAgent.averageSteps')}</th><th>{t('realAgent.averageLatency')}</th></tr></thead><tbody>{arenaResult.leaderboard.map((item: Any) => <tr key={item.configuration_id}><td>{item.position}</td><td>{item.name}</td><td>{Math.round(item.deterministic_score * 100)}%</td><td>{Math.round(item.completion_rate * 100)}%</td><td>{item.average_steps}</td><td>{item.average_duration_ms} ms</td></tr>)}</tbody></table></div>}
      </Card>
    </div>
  )
}
