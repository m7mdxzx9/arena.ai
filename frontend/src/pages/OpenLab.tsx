import { Card, Btn, go, useApi } from '../ui'
import { useI18n } from '../i18n'
import type { Any } from '../api'

const LABS = [
  { icon: '🗃️', title: 'openLab.datasetTitle', desc: 'openLab.datasetDesc', path: '/datasets' },
  { icon: '⚙️', title: 'openLab.mlTitle', desc: 'openLab.mlDesc', path: '/workbench' },
  { icon: '🔥', title: 'openLab.torchTitle', desc: 'openLab.torchDesc', path: '/pytorch' },
  { icon: '👁️', title: 'openLab.cnnTitle', desc: 'openLab.cnnDesc', path: '/pytorch/cnn' },
  { icon: '📖', title: 'openLab.ragTitle', desc: 'openLab.ragDesc', path: '/personal-rag' },
  { icon: '🧩', title: 'openLab.modelTitle', desc: 'openLab.modelDesc', path: '/models' },
  { icon: '🧑‍🏫', title: 'openLab.tutorTitle', desc: 'openLab.tutorDesc', path: '/tutor' },
  { icon: '📝', title: 'openLab.mistakesTitle', desc: 'openLab.mistakesDesc', path: '/mistakes' },
  { icon: '🦾', title: 'openLab.agentTitle', desc: 'openLab.agentDesc', path: '/real-agent' },
  { icon: '✍️', title: 'openLab.promptTitle', desc: 'openLab.promptDesc', path: '/prompt-lab' },
  { icon: '📏', title: 'openLab.evaluationTitle', desc: 'openLab.evaluationDesc', path: '/evaluation-lab' },
]

export default function OpenLab() {
  const { t } = useI18n()
  const caps = useApi<Any>('/api/system/capabilities')
  return (
    <div className="stack">
      <div className="topbar"><div><div className="kicker">{t('openLab.kicker')}</div><h1>{t('openLab.title')}</h1></div></div>
      <p className="muted">{t('openLab.subtitle')}</p>
      {caps.data && (
        <Card title={t('capabilities.title')} icon="🩺">
          <div className="capability-strip">
            <span className={caps.data.pytorch.installed ? 'cap-on' : 'cap-off'}>🔥 {t('capabilities.pytorch')}: {caps.data.pytorch.installed ? t('common.yes') : t('common.no')}</span>
            <span className={caps.data.pytorch.cuda_available ? 'cap-on' : 'cap-neutral'}>⚡ {t('capabilities.cuda')}: {caps.data.pytorch.cuda_available ? t('common.yes') : t('common.no')}</span>
            <span className={caps.data.ollama.reachable ? 'cap-on' : 'cap-off'}>🦙 {t('capabilities.ollama')}: {caps.data.ollama.reachable ? t('common.yes') : t('common.no')}</span>
            <span className="cap-on">ع {t('capabilities.arabic')}</span>
            <span className="cap-on">📚 {t('capabilities.rag')}</span>
          </div>
        </Card>
      )}
      <div className="grid g3 lab-grid">
        {LABS.map((lab) => (
          <Card key={lab.path} className="lab-card" icon={lab.icon} title={t(lab.title)}>
            <p className="muted">{t(lab.desc)}</p>
            <Btn kind="ghost" onClick={() => go(lab.path)}>{t('openLab.enter')} →</Btn>
          </Card>
        ))}
      </div>
    </div>
  )
}
