import type { Any } from '../api'
import { useI18n } from '../i18n'
import { Card, ErrorBox, Loading, Pill, go, useApi, useGame } from '../ui'

export default function Missions() {
  const { pid, meta } = useGame()
  const { t } = useI18n()
  const { data, error } = useApi<Any[]>(`/api/p/${pid}/missions`)
  const ds = (id: string) => meta.datasets.find((d: Any) => d.id === id)
  const sideMissionTitle = (id: string) => t(`learningMissions.titles.${id}`)
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('nav.campus')}</div><h1 style={{ margin: 0 }}>{t('nav.missions')}</h1></div></div>
      <p className="muted">{t('learningMissions.missionsSubtitle')}</p>
      <ErrorBox error={error} />
      {!data ? <Loading /> : (
        <div className="grid g3">
          {data.map((mission) => {
            const dataset = mission.dataset ? ds(mission.dataset) : null
            const learning = mission.kind === 'rag_compare' || mission.kind === 'rag_grounding' || mission.kind === 'tutor_hint_ladder'
            const statusText = mission.completed ? t('learningMissions.completed') : learning ? t('learningMissions.available') : mission.unlocked ? (mission.step > 0 ? t('learningMissions.inProgress') : t('learningMissions.available')) : t('learningMissions.locked')
            const statusKind = mission.completed ? 'green' : learning ? 'cyan' : mission.unlocked ? 'violet' : ''
            const requiredCount = mission.id === 'rag_compare' ? 2 : mission.id === 'tutor_hint_ladder' ? 3 : 1
            const progressLabel = mission.id === 'rag_compare' ? t('learningMissions.savedRuns') : mission.id === 'rag_grounding' ? t('learningMissions.citedRuns') : t('learningMissions.hintLevels')
            return (
              <div key={`${learning ? 'learning' : 'campaign'}:${mission.id}`} className={`card mission-card ${mission.unlocked ? 'clickable' : 'locked'} ${mission.completed ? 'done' : ''}`} onClick={() => mission.unlocked && go(learning ? `/learning-mission/${mission.id}` : `/mission/${mission.id}`)}>
                <div className="row between"><span style={{ fontSize: '1.8rem' }}>{learning ? mission.icon : dataset?.icon}</span><Pill kind={statusKind}>{mission.completed ? '✓ ' : ''}{statusText}</Pill></div>
                <h3 style={{ margin: '0.4rem 0 0.2rem' }}>{learning ? sideMissionTitle(mission.id) : mission.title}</h3>
                <small className="muted">{t(`campaign.areas.${mission.area}`)}{dataset ? ` · ${dataset.title}` : ''} · {mission.xp} XP</small>
                {learning ? <>
                  <p className="muted" style={{ marginTop: '.6rem' }}>{t(`learningMissions.descriptions.${mission.id}`)}</p>
                  {!mission.completed && <div className="row"><Pill>{t('learningMissions.progress')}: {Math.min(Number(mission.step || 0), requiredCount)} / {requiredCount}</Pill><small>{progressLabel}</small></div>}
                </> : <>
                  {!mission.unlocked && <div style={{ marginTop: 8 }}><small>{t('learningMissions.requires')}: {mission.requires.map((required: Any) => <a key={required.id} href={`#/lesson/${required.id}`} onClick={(event) => event.stopPropagation()} style={{ marginInlineEnd: 6 }}>{required.name}</a>)}</small></div>}
                  {mission.step > 0 && !mission.completed && <small>{t('learningMissions.inProgress')} · {mission.step}/4</small>}
                </>}
              </div>
            )
          })}
        </div>
      )}
      <Card title={t('learningMissions.title')} icon="📚">
        <p>{t('learningMissions.instructions')}</p>
        <div className="row wrap"><a className="btn btn-ghost btn-sm" href="#/personal-rag">{t('learningMissions.openAdvancedRag')}</a><a className="btn btn-ghost btn-sm" href="#/tutor">{t('learningMissions.openTutor')}</a></div>
      </Card>
      <Card title={t('learningMissions.openEndedTitle')} icon="🔬"><p>{t('learningMissions.openEndedDescription')} <a href="#/research">{t('nav.research')}</a></p></Card>
    </div>
  )
}
