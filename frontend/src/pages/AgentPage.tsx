import AgentLab from '../labs/AgentLab'

export default function AgentPage() {
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">Agent Arena · Agent Harness</div><h1 style={{ margin: 0 }}>Agent Security Lab</h1></div></div>
      <p className="muted">An agent = a language model in a loop that can call tools. When it reads untrusted text (web pages, emails, documents), that text can try to give it orders — <b>prompt injection</b>. Combine defences and measure two things at once: attacks blocked <i>and</i> legitimate work still done.</p>
      <AgentLab context="agent_lab" />
    </div>
  )
}
