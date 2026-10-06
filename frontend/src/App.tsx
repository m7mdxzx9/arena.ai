import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { get, storedPlayer, storePlayer, type Any } from './api'
import { Bar, GameCtx, Loading, Toasts, go, useGame, useRoute, useToasts } from './ui'
import Start from './pages/Start'
import Campus from './pages/Campus'
import AreaPage from './pages/Area'
import Tree from './pages/Tree'
import Lesson from './pages/Lesson'
import DataLab from './pages/DataLab'
import WorkbenchPage from './pages/WorkbenchPage'
import History from './pages/History'
import NNLabPage from './pages/NNLabPage'
import PredictionLab from './pages/PredictionLab'
import LanguageLab from './pages/LanguageLab'
import RagPage from './pages/RagPage'
import AgentPage from './pages/AgentPage'
import Dojo from './pages/Dojo'
import Missions from './pages/Missions'
import MissionPage from './pages/MissionPage'
import Bosses from './pages/Bosses'
import BossPage from './pages/BossPage'
import Research from './pages/Research'
import Profile from './pages/Profile'

type NavItem = { path: string; label: string; icon: string; equip?: string }
const NAV: { section: string; items: NavItem[] }[] = [
  { section: 'Campus', items: [
    { path: '', label: 'Campus Map', icon: '🗺️' },
    { path: 'tree', label: 'Knowledge Tree', icon: '🌳' },
    { path: 'missions', label: 'Missions', icon: '🎯' },
    { path: 'bosses', label: 'Boss Battles', icon: '⚔️' },
  ] },
  { section: 'Labs', items: [
    { path: 'data', label: 'Data Lab', icon: '🔍', equip: 'data_lens' },
    { path: 'workbench', label: 'ML Workbench', icon: '⚙️', equip: 'workbench' },
    { path: 'history', label: 'Experiments', icon: '📓', equip: 'workbench' },
    { path: 'predict', label: 'Prediction Lab', icon: '🔮' },
    { path: 'nn', label: 'Neural Net Lab', icon: '🧠', equip: 'neural_forge' },
    { path: 'language', label: 'Language Lab', icon: '💬', equip: 'tokenizer_press' },
    { path: 'rag', label: 'RAG Archives', icon: '📚', equip: 'vector_vault' },
    { path: 'agent', label: 'Agent Arena', icon: '🤖', equip: 'agent_harness' },
    { path: 'dojo', label: 'Code Dojo', icon: '⌨️', equip: 'code_terminal' },
  ] },
  { section: 'Career', items: [
    { path: 'research', label: 'Research Institute', icon: '🔭' },
    { path: 'profile', label: 'Profile & Settings', icon: '🪪' },
  ] },
]

function EquipGate({ equip, children }: { equip?: string; children: ReactNode }) {
  const { ov, meta } = useGame()
  if (!equip) return <>{children}</>
  const e = ov.equipment[equip]
  if (e?.unlocked) return <>{children}</>
  const reqs: string[] = e?.requires || []
  return (
    <div className="card" style={{ maxWidth: 640, margin: '3rem auto', textAlign: 'center' }}>
      <div className="big-emoji">🔒</div>
      <h2>{e?.name} is locked</h2>
      <p className="muted">{e?.desc}</p>
      <p>Reach 60% mastery in {reqs.map((r, i) => <span key={r}>{i > 0 && ', '}<a href={`#/lesson/${r}`}>{meta.concepts[r]?.name || r}</a></span>)} to unlock this equipment.</p>
      <p className="muted"><small>Prefer to explore freely? Enable <b>Free Play</b> in <a href="#/profile">Profile & Settings</a>.</small></p>
    </div>
  )
}

function Shell() {
  const route = useRoute()
  const { ov } = useGame()
  const [page, ...rest] = route
  const cur = page || ''
  let content: ReactNode
  const allItems = NAV.flatMap((s) => s.items)
  const item = allItems.find((i) => i.path === cur)
  switch (cur) {
    case '': content = <Campus />; break
    case 'area': content = <AreaPage id={rest[0]} />; break
    case 'tree': content = <Tree focus={rest[0]} />; break
    case 'lesson': content = <Lesson key={rest[0]} cid={rest[0]} />; break
    case 'data': content = <DataLab initial={rest[0]} />; break
    case 'workbench': content = <WorkbenchPage dataset={rest[0]} />; break
    case 'history': content = <History />; break
    case 'predict': content = <PredictionLab focus={rest[0]} />; break
    case 'nn': content = <NNLabPage />; break
    case 'language': content = <LanguageLab />; break
    case 'rag': content = <RagPage />; break
    case 'agent': content = <AgentPage />; break
    case 'dojo': content = <Dojo key={rest[0] || 'list'} exId={rest[0]} />; break
    case 'missions': content = <Missions />; break
    case 'mission': content = <MissionPage key={rest[0]} id={rest[0]} />; break
    case 'bosses': content = <Bosses />; break
    case 'boss': content = <BossPage key={rest[0]} id={rest[0]} />; break
    case 'research': content = <Research />; break
    case 'profile': content = <Profile />; break
    default: content = <div className="empty">Page not found. <a href="#/">Back to campus</a></div>
  }
  const rank = ov.rank
  const nextXp = rank.next?.xp
  return (
    <div className="app">
      <nav className="side">
        <div className="brand" onClick={() => go('/')}><img src="/favicon.svg" alt="" /><span>NEURAL FORGE</span></div>
        {NAV.map((sec) => (
          <div key={sec.section} style={{ display: 'contents' }}>
            <div className="nav-sec">{sec.section}</div>
            {sec.items.map((it) => {
              const locked = it.equip && !ov.equipment[it.equip]?.unlocked
              return (
                <a key={it.path} href={`#/${it.path}`} className={`nav ${cur === it.path || (cur === 'mission' && it.path === 'missions') || (cur === 'boss' && it.path === 'bosses') || (cur === 'area' && it.path === '') || (cur === 'lesson' && it.path === 'tree') ? 'active' : ''}`}>
                  <span>{it.icon}</span>{it.label}{locked && <span className="lock">🔒</span>}
                </a>
              )
            })}
          </div>
        ))}
        <div className="side-player">
          <div className="row between"><b>{ov.player.name}</b><span className="pill violet">{ov.mode.name}</span></div>
          <div className="rank" style={{ margin: '0.3rem 0' }}>{rank.title}</div>
          <Bar value={ov.player.xp} max={nextXp || ov.player.xp || 1} thin />
          <small>{ov.player.xp} XP{nextXp ? ` / ${nextXp} for ${rank.next.title}` : ' — max rank'}</small>
        </div>
      </nav>
      <main className="main">
        <EquipGate equip={item?.equip}>{content}</EquipGate>
      </main>
    </div>
  )
}

export default function App() {
  const [pid, setPidState] = useState<number | null>(storedPlayer())
  const [ov, setOv] = useState<Any>(null)
  const [meta, setMeta] = useState<Any>(null)
  const [fatal, setFatal] = useState<string | null>(null)
  const { toasts, toast } = useToasts()
  const lastRank = useRef<number | null>(null)

  const setPid = useCallback((id: number | null) => {
    storePlayer(id)
    setPidState(id)
    setOv(null)
    lastRank.current = null
  }, [])

  const refresh = useCallback(async () => {
    if (!pid) return
    try {
      const o = await get(`/api/p/${pid}`)
      if (lastRank.current !== null && o.rank.index > lastRank.current) toast({ kind: 'level', icon: '🎖️', text: `Promoted to ${o.rank.title}!` })
      lastRank.current = o.rank.index
      setOv(o)
    } catch (e: Any) {
      if (e.status === 404) setPid(null)
      else setFatal(e.message)
    }
  }, [pid, setPid, toast])

  useEffect(() => {
    get('/api/meta').then(setMeta).catch((e) => setFatal(e.message))
  }, [])
  useEffect(() => {
    refresh()
  }, [refresh])

  const reward = useCallback((res: Any) => {
    if (!res) return
    if (res.xp) toast({ kind: 'xp', icon: '⭐', text: `+${res.xp} XP` })
    ;(res.events || []).forEach((e: Any) => toast({ kind: 'level', icon: e.kind === 'mastered' ? '🏆' : '📗', text: e.text }))
    ;(res.unlocked || []).slice(0, 3).forEach((u: Any) => toast({ kind: 'unlock', icon: '🔓', text: `Unlocked: ${u.name}` }))
    if ((res.unlocked || []).length > 3) toast({ kind: 'unlock', icon: '🔓', text: `…and ${res.unlocked.length - 3} more concepts` })
    ;(res.achievements || []).forEach((a: Any) => toast({ kind: 'ach', icon: a.icon, text: `Achievement: ${a.name}` }))
    refresh()
  }, [toast, refresh])

  if (fatal) return <div className="main"><div className="error-box">Could not reach the NEURAL FORGE server: {fatal}</div></div>
  if (!meta) return <Loading text="Booting the campus…" />
  if (!pid) return <><Start meta={meta} onPlayer={(id) => setPid(id)} /><Toasts toasts={toasts} /></>
  if (!ov) return <Loading text="Loading your progress…" />
  return (
    <GameCtx.Provider value={{ pid, ov, meta, refresh, toast, reward, setPid }}>
      <Shell />
      <Toasts toasts={toasts} />
    </GameCtx.Provider>
  )
}
