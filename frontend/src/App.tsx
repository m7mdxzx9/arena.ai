import { lazy, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { formatApiError, get, storedPlayer, storePlayer, type Any } from './api'
import { useI18n } from './i18n'
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
const OpenLab = lazy(() => import('./pages/OpenLab'))
const DatasetsPage = lazy(() => import('./pages/Datasets'))
const PyTorchLab = lazy(() => import('./pages/PyTorchLab'))
const CheckpointsPage = lazy(() => import('./pages/Checkpoints'))
const ModelHub = lazy(() => import('./pages/ModelHub'))
const TutorPage = lazy(() => import('./pages/Tutor'))
const MistakesPage = lazy(() => import('./pages/Mistakes'))
const PersonalRagPage = lazy(() => import('./pages/PersonalRag'))
const RealAgentPage = lazy(() => import('./pages/RealAgent'))
const PromptLabPage = lazy(() => import('./pages/PromptLab'))
const EvaluationLabPage = lazy(() => import('./pages/EvaluationLab'))
const PortfolioPage = lazy(() => import('./pages/Portfolio'))
const BackupRestorePage = lazy(() => import('./pages/BackupRestore'))

type NavItem = { path: string; labelKey: string; icon: string; equip?: string }
const NAV: { sectionKey: string; items: NavItem[] }[] = [
  { sectionKey: 'nav.campus', items: [
    { path: '', labelKey: 'nav.campusMap', icon: '🗺️' },
    { path: 'tree', labelKey: 'nav.knowledgeTree', icon: '🌳' },
    { path: 'missions', labelKey: 'nav.missions', icon: '🎯' },
    { path: 'bosses', labelKey: 'nav.bosses', icon: '⚔️' },
  ] },
  { sectionKey: 'nav.labs', items: [
    { path: 'open-lab', labelKey: 'nav.openLab', icon: '🚀' },
    { path: 'datasets', labelKey: 'nav.datasets', icon: '🗃️' },
    { path: 'data', labelKey: 'nav.dataLab', icon: '🔍', equip: 'data_lens' },
    { path: 'workbench', labelKey: 'nav.workbench', icon: '⚙️', equip: 'workbench' },
    { path: 'history', labelKey: 'nav.experiments', icon: '📓', equip: 'workbench' },
    { path: 'predict', labelKey: 'nav.prediction', icon: '🔮' },
    { path: 'nn', labelKey: 'nav.neural', icon: '🧠', equip: 'neural_forge' },
    { path: 'pytorch', labelKey: 'nav.pytorch', icon: '🔥' },
    { path: 'checkpoints', labelKey: 'nav.checkpoints', icon: '💾' },
    { path: 'language', labelKey: 'nav.language', icon: '💬', equip: 'tokenizer_press' },
    { path: 'rag', labelKey: 'nav.rag', icon: '📘', equip: 'vector_vault' },
    { path: 'personal-rag', labelKey: 'nav.personalRag', icon: '📚' },
    { path: 'models', labelKey: 'nav.modelHub', icon: '🧩' },
    { path: 'tutor', labelKey: 'nav.tutor', icon: '🧑‍🏫' },
    { path: 'mistakes', labelKey: 'nav.mistakes', icon: '📝' },
    { path: 'real-agent', labelKey: 'nav.realAgent', icon: '🦾' },
    { path: 'prompt-lab', labelKey: 'nav.promptLab', icon: '✍️' },
    { path: 'evaluation-lab', labelKey: 'nav.evaluationLab', icon: '📏' },
    { path: 'agent', labelKey: 'nav.agent', icon: '🤖', equip: 'agent_harness' },
    { path: 'dojo', labelKey: 'nav.dojo', icon: '⌨️', equip: 'code_terminal' },
  ] },
  { sectionKey: 'nav.career', items: [
    { path: 'research', labelKey: 'nav.research', icon: '🔭' },
    { path: 'portfolio', labelKey: 'nav.portfolio', icon: '💼' },
    { path: 'backup', labelKey: 'nav.backup', icon: '💾' },
    { path: 'profile', labelKey: 'nav.profile', icon: '🪪' },
  ] },
]

function EquipGate({ equip, children }: { equip?: string; children: ReactNode }) {
  const { t } = useI18n()
  const { ov, meta } = useGame()
  if (!equip) return <>{children}</>
  const e = ov.equipment[equip]
  if (e?.unlocked) return <>{children}</>
  const reqs: string[] = e?.requires || []
  return (
    <div className="card" style={{ maxWidth: 640, margin: '3rem auto', textAlign: 'center' }}>
      <div className="big-emoji">🔒</div>
      <h2>{t('shell.locked', { name: e?.name || '' })}</h2>
      <p className="muted">{e?.desc}</p>
      <p>{t('shell.reachMastery')} {reqs.map((r, i) => <span key={r}>{i > 0 && ', '}<a href={`#/lesson/${r}`}>{meta.concepts[r]?.name || r}</a></span>)}</p>
      <p className="muted"><small>{t('shell.freePlay')}</small></p>
    </div>
  )
}

function Shell() {
  const { t } = useI18n()
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
    case 'open-lab': content = <OpenLab />; break
    case 'datasets': content = <DatasetsPage />; break
    case 'data': content = <DataLab initial={rest[0]} />; break
    case 'workbench': content = <WorkbenchPage dataset={rest[0]} />; break
    case 'history': content = <History />; break
    case 'predict': content = <PredictionLab focus={rest[0]} />; break
    case 'nn': content = <NNLabPage />; break
    case 'pytorch': content = <PyTorchLab initialTab={rest[0] === 'cnn' ? 'cnn' : 'mlp'} />; break
    case 'checkpoints': content = <CheckpointsPage />; break
    case 'language': content = <LanguageLab />; break
    case 'rag': content = <RagPage />; break
    case 'personal-rag': content = <PersonalRagPage />; break
    case 'models': content = <ModelHub />; break
    case 'tutor': content = <TutorPage />; break
    case 'mistakes': content = <MistakesPage />; break
    case 'real-agent': content = <RealAgentPage />; break
    case 'prompt-lab': content = <PromptLabPage />; break
    case 'evaluation-lab': content = <EvaluationLabPage />; break
    case 'agent': content = <AgentPage />; break
    case 'dojo': content = <Dojo key={rest[0] || 'list'} exId={rest[0]} />; break
    case 'missions': content = <Missions />; break
    case 'mission': content = <MissionPage key={rest[0]} id={rest[0]} />; break
    case 'bosses': content = <Bosses />; break
    case 'boss': content = <BossPage key={rest[0]} id={rest[0]} />; break
    case 'research': content = <Research />; break
    case 'portfolio': content = <PortfolioPage />; break
    case 'backup': content = <BackupRestorePage />; break
    case 'profile': content = <Profile />; break
    default: content = <div className="empty">{t('shell.pageNotFound')} <a href="#/">{t('shell.backCampus')}</a></div>
  }
  const rank = ov.rank
  const nextXp = rank.next?.xp
  return (
    <div className="app">
      <nav className="side">
        <div className="brand" onClick={() => go('/')}><img src="/favicon.svg" alt="" /><span>NEURAL FORGE</span></div>
        {NAV.map((sec) => (
          <div key={sec.sectionKey} style={{ display: 'contents' }}>
            <div className="nav-sec">{t(sec.sectionKey)}</div>
            {sec.items.map((it) => {
              const locked = it.equip && !ov.equipment[it.equip]?.unlocked
              return (
                <a key={it.path} href={`#/${it.path}`} className={`nav ${cur === it.path || (cur === 'mission' && it.path === 'missions') || (cur === 'boss' && it.path === 'bosses') || (cur === 'area' && it.path === '') || (cur === 'lesson' && it.path === 'tree') ? 'active' : ''}`}>
                  <span>{it.icon}</span>{t(it.labelKey)}{locked && <span className="lock">🔒</span>}
                </a>
              )
            })}
          </div>
        ))}
        <div className="side-player">
          <div className="row between"><b>{ov.player.name}</b><span className="pill violet">{ov.mode.name}</span></div>
          <div className="rank" style={{ margin: '0.3rem 0' }}>{rank.title}</div>
          <Bar value={ov.player.xp} max={nextXp || ov.player.xp || 1} thin />
          <small dir="ltr">{ov.player.xp} XP{nextXp ? ` / ${nextXp} for ${rank.next.title}` : ` — ${t('shell.maxRank')}`}</small>
        </div>
      </nav>
      <main className="main">
        <EquipGate equip={item?.equip}><Suspense fallback={<Loading />}>{content}</Suspense></EquipGate>
      </main>
    </div>
  )
}

export default function App() {
  const { t, language, setLanguage } = useI18n()
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
      if (o.player.settings?.language && o.player.settings.language !== language) setLanguage(o.player.settings.language)
      if (lastRank.current !== null && o.rank.index > lastRank.current) toast({ kind: 'level', icon: '🎖️', text: t('shell.promoted', { rank: o.rank.title }) })
      lastRank.current = o.rank.index
      setOv(o)
    } catch (e: Any) {
      if (e.status === 404) setPid(null)
      else setFatal(formatApiError(e))
    }
  }, [pid, setPid, toast, language, setLanguage, t])

  useEffect(() => {
    get('/api/meta').then(setMeta).catch((e) => setFatal(formatApiError(e)))
  }, [])
  useEffect(() => {
    refresh()
  }, [refresh])

  const reward = useCallback((res: Any) => {
    if (!res) return
    if (res.xp) toast({ kind: 'xp', icon: '⭐', text: `+${res.xp} XP` })
    ;(res.events || []).forEach((e: Any) => toast({ kind: 'level', icon: e.kind === 'mastered' ? '🏆' : '📗', text: e.text }))
    ;(res.unlocked || []).slice(0, 3).forEach((u: Any) => toast({ kind: 'unlock', icon: '🔓', text: t('shell.unlocked', { name: u.name }) }))
    if ((res.unlocked || []).length > 3) toast({ kind: 'unlock', icon: '🔓', text: t('shell.moreConcepts', { count: res.unlocked.length - 3 }) })
    ;(res.achievements || []).forEach((a: Any) => toast({ kind: 'ach', icon: a.icon, text: t('shell.achievement', { name: a.name }) }))
    refresh()
  }, [toast, refresh, t])

  if (fatal) return <div className="main"><div className="error-box">{t('shell.serverError', { error: fatal })}</div></div>
  if (!meta) return <Loading text={t('shell.booting')} />
  if (!pid) return <><Start meta={meta} onPlayer={(id) => setPid(id)} /><Toasts toasts={toasts} /></>
  if (!ov) return <Loading text={t('shell.loadingProgress')} />
  return (
    <GameCtx.Provider value={{ pid, ov, meta, refresh, toast, reward, setPid }}>
      <Shell />
      <Toasts toasts={toasts} />
    </GameCtx.Provider>
  )
}
