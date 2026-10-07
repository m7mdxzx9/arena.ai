import { useState } from 'react'
import { post, type Any } from '../api'
import { useI18n, Ltr, type Language } from '../i18n'
import { Bar, Btn, Card, Pill, Select, Toggle, go, useApi, useGame } from '../ui'

export default function Profile() {
  const { t, language, setLanguage, locale } = useI18n()
  const { pid, ov, meta, refresh, setPid, toast } = useGame()
  const review = useApi<Any[]>(`/api/p/${pid}/review`)
  const refl = useApi<Any[]>(`/api/p/${pid}/reflections`)
  const [busy, setBusy] = useState(false)
  const save = async (body: Any) => {
    setBusy(true)
    try {
      await post(`/api/p/${pid}/settings`, body)
      if (body.language === 'en' || body.language === 'ar') setLanguage(body.language as Language)
      await refresh()
      toast({ kind: 'info', text: t('common.settingsSaved') })
    } finally { setBusy(false) }
  }
  const r = ov.rank
  const earned = ov.achievements.filter((achievement: Any) => achievement.earned_at)
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('settings.playerProfile')}</div><h1>{ov.player.name}</h1></div><div className="spacer" /><Btn kind="ghost" small onClick={() => setPid(null)}>{t('settings.switchPlayer')}</Btn></div>
      <div className="grid g2">
        <Card title={t('settings.career')} icon="🎖️">
          <div className="rank-ladder">
            {r.all.map((title: string, index: number) => <div key={title} className={`rung ${index < r.index ? 'done' : index === r.index ? 'cur' : ''}`}><span>{index + 1}</span>{title}</div>)}
          </div>
          {r.next && <div className="col" style={{ marginTop: 10 }}>
            <b>{t('campus.nextRank')}: {r.next.title}</b>
            <Bar value={ov.player.xp} max={r.next.xp} label={<Ltr>XP {ov.player.xp} / {r.next.xp}</Ltr>} />
            <Bar value={ov.counts.proficient} max={Math.max(1, r.next.proficient)} label={`${ov.counts.proficient} / ${r.next.proficient}`} />
            {r.next.bosses > 0 && <Bar value={ov.counts.bosses_defeated} max={r.next.bosses} label={`${ov.counts.bosses_defeated} / ${r.next.bosses}`} />}
          </div>}
        </Card>
        <div className="col">
          <Card title={t('settings.language')} icon="🌐">
            <Select value={language} onChange={(value) => save({ language: value })} options={[{ value: 'en', label: `🇬🇧 ${t('settings.english')}` }, { value: 'ar', label: `🇸🇦 ${t('settings.arabic')}` }]} />
            <p className="muted"><small>{t('settings.languageHelp')}</small></p>
            <div className="direction-preview"><span dir="rtl">التعلم الآلي (Machine Learning)</span><code dir="ltr">model.fit(X_train, y_train)</code></div>
          </Card>
          <Card title={t('settings.difficulty')} icon="🎚️">
            <div className="col">
              {Object.entries(meta.modes).map(([id, mode]: [string, Any]) => <button key={id} className={`option ${Number(id) === ov.player.mode ? 'sel' : ''}`} disabled={busy} onClick={() => save({ mode: Number(id) })}><span className="key">{id}</span><span style={{ flex: 1 }}><b>{mode.name}</b><br /><small className="muted">{mode.desc}</small></span></button>)}
              <Toggle label={t('settings.codeMode')} checked={!!ov.player.settings.show_code} onChange={(value) => save({ show_code: value })} />
              <Toggle label={t('settings.freePlay')} checked={!!ov.player.settings.free_play} onChange={(value) => save({ free_play: value })} />
            </div>
          </Card>
        </div>
      </div>
      <div className="grid g2">
        <Card title={t('settings.review', { count: review.data?.length ?? 0 })} icon="🔁">
          {review.data?.length ? review.data.map((concept) => <div key={concept.id} className="row between"><a href={`#/lesson/${concept.id}`}>{concept.name}</a><Pill>box {concept.box}</Pill></div>) : <p className="muted">{t('settings.nothingDue')}</p>}
        </Card>
        <Card title={t('settings.reflections')} icon="🪞">
          {refl.data?.length ? <div className="col" style={{ maxHeight: 260, overflowY: 'auto' }}>{refl.data.map((item, index) => <div key={index}><small className="muted">{meta.concepts[item.concept_id]?.name} · {new Date(item.ts * 1000).toLocaleDateString(locale)}</small><div>{item.text}</div></div>)}</div> : <p className="muted">{t('settings.noReflections')}</p>}
        </Card>
      </div>
      <Card title={t('settings.achievements', { earned: earned.length, total: ov.achievements.length })} icon="🏅">
        <div className="grid g4">{ov.achievements.map((achievement: Any) => <div key={achievement.id} className={`ach ${achievement.earned_at ? 'on' : ''}`} title={achievement.desc}><span className="ach-i">{achievement.earned_at ? achievement.icon : '🔒'}</span><div><b>{achievement.name}</b><br /><small className="muted">{achievement.desc}</small></div></div>)}</div>
      </Card>
      <Card title={t('settings.recent')} icon="📈">
        {ov.events.length ? ov.events.map((event: Any, index: number) => <div key={index} className="row between"><small>{event.reason}</small><Pill kind="violet">+{event.xp} XP</Pill></div>) : <p className="muted"><a href="#/" onClick={() => go('/')}>{t('shell.backCampus')} →</a></p>}
      </Card>
    </div>
  )
}
