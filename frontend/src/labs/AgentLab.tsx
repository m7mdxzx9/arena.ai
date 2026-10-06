import { useState } from 'react'
import { post, type Any } from '../api'
import { Btn, Card, ErrorBox, Loading, Pill, Toggle, pct, useGame } from '../ui'

export function AgentTrace({ trace }: { trace: Any[] }) {
  return (
    <div className="agent-trace">
      {trace.map((t, i) => (
        <div key={i} className={`t-${t.step} ${t.blocked ? 't-blocked' : ''} ${t.channel ? `t-ch-${t.channel}` : ''}`}>
          <span className="t-tag">{t.channel ? `${t.channel}${t.trusted === false ? ' · UNTRUSTED' : ''}` : t.step}</span>
          <span className="t-text">{t.text}</span>
        </div>
      ))}
    </div>
  )
}

export function AgentResult({ res }: { res: Any }) {
  const [open, setOpen] = useState<string | null>(res.results.find((r: Any) => r.attack_succeeded)?.id || null)
  const m = res.metrics
  return (
    <div className="stack">
      <div className="metrics">
        <div className="metric"><div className="metric-l">attack success rate</div><div className="metric-v" style={{ color: m.attack_success_rate > 0 ? 'var(--red)' : 'var(--green)' }}>{pct(m.attack_success_rate, 0)}</div></div>
        <div className="metric" title="Share of all tasks (including legitimate ones) completed correctly"><div className="metric-l">utility</div><div className="metric-v">{pct(m.utility, 0)}</div></div>
        <div className="metric" title="Harmless documents wrongly blocked"><div className="metric-l">false positives</div><div className="metric-v">{m.false_positives}</div></div>
        <div className="metric" title="How often the human had to approve something"><div className="metric-l">confirmations</div><div className="metric-v">{m.confirmations}</div></div>
      </div>
      {res.advice.map((a: string, i: number) => <div key={i} className="warn-box">{a}</div>)}
      {res.results.map((r: Any) => (
        <div key={r.id} className="card inner">
          <div className="row between" style={{ cursor: 'pointer' }} onClick={() => setOpen(open === r.id ? null : r.id)}>
            <div><b>{r.user}</b><br /><small className="muted">{r.attack ? `attack: ${r.attack}` : 'legitimate task (no attack)'}</small></div>
            <div className="row">
              {r.attack && (r.attack_succeeded ? <Pill kind="red">attack succeeded</Pill> : <Pill kind="green">attack blocked</Pill>)}
              {r.success ? <Pill kind="cyan">task done</Pill> : <Pill kind="amber">task failed</Pill>}
              {r.false_positive && <Pill kind="amber">false positive</Pill>}
            </div>
          </div>
          {open === r.id && <div style={{ marginTop: 8 }}><AgentTrace trace={r.trace} />{r.answer && <div className="info-box" style={{ marginTop: 6 }}><b>Final answer:</b> {r.answer}</div>}</div>}
        </div>
      ))}
    </div>
  )
}

export default function AgentLab({ context, onRun }: { context?: string; onRun?: (r: Any) => void }) {
  const { pid, meta, reward } = useGame()
  const [def, setDef] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [out, setOut] = useState<Any>(null)
  const run = async () => {
    setBusy(true)
    try { const r = await post(`/api/p/${pid}/run/agent`, { config: { defences: def }, context }); setOut(r); reward(r); onRun?.(r) } catch (e: Any) { setOut({ ok: false, error: e.message }) } finally { setBusy(false) }
  }
  return (
    <div className="grid g-side">
      <div className="col">
        <Card title="Defence layers" icon="🛡️">
          {Object.entries(meta.agent.defences).map(([id, d]: [string, Any]) => (
            <div key={id} style={{ marginBottom: 8 }}>
              <Toggle label={<b>{d.name}</b>} checked={def.includes(id)} onChange={(on) => setDef(on ? [...def, id] : def.filter((x) => x !== id))} />
              <small className="muted" style={{ display: 'block', marginLeft: 28 }}>{d.desc}</small>
            </div>
          ))}
          <Btn onClick={run} disabled={busy}>{busy ? 'Running…' : '▶ Run the 8 agent tasks'}</Btn>
        </Card>
        <Card title="The agent's tools" icon="🧰">
          {Object.entries(meta.agent.tools).map(([id, t]: [string, Any]) => <div key={id} className="row between"><span className="mono">{id}</span><small className="muted">{t.risk}</small></div>)}
        </Card>
      </div>
      <div className="col">
        <div className="info-box">This is a <b>deterministic simulation</b> of a tool-using assistant reading documents that may contain hidden instructions. It models the documented behaviour of LLM agents (instructions in data can be adopted; model-level defences are probabilistic; code-level gateways are reliable) so you can see each defence's real effect, trace by trace. No real emails or files are touched.</div>
        {busy && <Loading />}
        {out && !out.ok && <ErrorBox error={out.error} />}
        {out?.ok && <Card title={`Run #${out.run_id}`} icon="📊"><AgentResult res={out.result} /></Card>}
      </div>
    </div>
  )
}
