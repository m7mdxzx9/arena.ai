import { useState } from 'react'
import { post, type Any } from '../api'
import { useI18n, Ltr } from '../i18n'
import { Btn, Card, ErrorBox, Loading, Pill, Tabs, useApi, useGame } from '../ui'

export default function MistakesPage() {
  const { t, locale } = useI18n()
  const { pid } = useGame()
  const [filter, setFilter] = useState('all')
  const records = useApi<Any[]>(`/api/p/${pid}/mistakes?status=${filter}`, [filter])
  const review = async (id: number, remembered: boolean) => {
    await post(`/api/p/${pid}/mistakes/${id}/review`, { remembered })
    records.reload()
  }
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('mistakes.kicker')}</div><h1>{t('mistakes.title')}</h1></div></div>
      <p className="muted">{t('mistakes.subtitle')}</p>
      <Tabs value={filter} onChange={setFilter} tabs={[{ id: 'all', label: t('mistakes.all') }, { id: 'unresolved', label: t('mistakes.unresolved') }, { id: 'due', label: t('mistakes.due') }, { id: 'mastered', label: t('mistakes.mastered') }]} />
      <ErrorBox error={records.error} />
      {records.loading && <Loading />}
      {records.data?.length === 0 && <Card><div className="empty">{t('mistakes.empty')}</div></Card>}
      <div className="grid g2">
        {records.data?.map((record) => (
          <Card key={record.id} title={<><Ltr>{record.concept}</Ltr> {record.resolved ? <Pill kind="green">{t('mistakes.mastered')}</Pill> : record.due ? <Pill kind="amber">{t('mistakes.due')}</Pill> : <Pill>{t('mistakes.unresolved')}</Pill>}</>} icon="📝">
            <small className="muted">{new Date(record.created_at * 1000).toLocaleDateString(locale)} · <Ltr>{record.category}</Ltr> · {t('mistakes.reviews', { count: record.review_count })}</small>
            <dl className="mistake-detail">
              <dt>{t('mistakes.what')}</dt><dd>{record.player_action}</dd>
              <dt>{t('mistakes.why')}</dt><dd>{record.explanation}</dd>
              <dt>{t('mistakes.principle')}</dt><dd>{record.correct_principle}</dd>
              {record.example && <><dt>{t('mistakes.example')}</dt><dd>{record.example}</dd></>}
            </dl>
            {!record.resolved && <div className="row"><Btn small kind="success" onClick={() => review(record.id, true)}>✓ {t('mistakes.remembered')}</Btn><Btn small kind="ghost" onClick={() => review(record.id, false)}>↻ {t('mistakes.reviewAgain')}</Btn></div>}
          </Card>
        ))}
      </div>
    </div>
  )
}
