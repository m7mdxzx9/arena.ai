import { useRef, useState } from 'react'
import { get, post, type Any } from '../api'
import { useI18n } from '../i18n'
import { Btn, Card, ErrorBox, Pill, useGame } from '../ui'

const MAX_BYTES = 5 * 1024 * 1024

export default function BackupRestorePage() {
  const { t } = useI18n()
  const { pid } = useGame()
  const input = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Any>(null)
  const download = async () => {
    setBusy(true); setError(null)
    try {
      const value = await get(`/api/p/${pid}/backup`)
      const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `neural-forge-player-${pid}.json`; link.click(); URL.revokeObjectURL(url)
    } catch (e: Any) { setError(e.message) } finally { setBusy(false) }
  }
  const restore = async () => {
    if (!file) return
    setBusy(true); setError(null); setResult(null)
    try {
      if (file.size > MAX_BYTES) throw new Error('Backup exceeds the 5 MiB limit.')
      const value = JSON.parse(await file.text())
      setResult(await post(`/api/p/${pid}/restore`, { backup: value }))
    } catch (e: Any) { setError(e instanceof SyntaxError ? 'The selected file is not valid JSON.' : e.message) } finally { setBusy(false) }
  }
  return <div className="stack">
    <div className="topbar"><div><div className="kicker">{t('backup.kicker')}</div><h1>{t('backup.title')}</h1></div></div>
    <p className="muted">{t('backup.subtitle')}</p>
    <div className="grid g2">
      <Card title={t('backup.export')} icon="📤"><p>{t('backup.mergeNotice')}</p><Btn onClick={download} disabled={busy}>{t('backup.export')}</Btn></Card>
      <Card title={t('backup.restore')} icon="📥"><div className="warn-box">{t('backup.security')}</div><input ref={input} type="file" accept="application/json,.json" hidden onChange={(event) => setFile(event.target.files?.[0] || null)} /><Btn kind="ghost" onClick={() => input.current?.click()}>{t('backup.choose')}</Btn>{file && <div className="file-chip"><span>📄</span><b>{file.name}</b><small>{(file.size / 1024).toFixed(1)} KiB</small></div>}<Btn onClick={restore} disabled={busy || !file}>{t('backup.restore')}</Btn></Card>
    </div>
    <ErrorBox error={error} />
    {result && <Card title={t('backup.restored')} icon="✅" right={<Pill kind="green">{result.mode}</Pill>}><pre className="json-view" dir="ltr">{JSON.stringify(result, null, 2)}</pre></Card>}
  </div>
}
