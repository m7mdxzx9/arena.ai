import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { get, localizedApiError, type Any } from './api'
import { useI18n } from './i18n'

// ------------------------------------------------------------------ router (hash based, no dependency)
export function useRoute(): string[] {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const on = () => setHash(window.location.hash)
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent)
}
export const go = (path: string) => {
  window.location.hash = path.startsWith('/') ? path : `/${path}`
  window.scrollTo({ top: 0 })
}

// ------------------------------------------------------------------ game context
export type Toast = { id: number; kind: 'xp' | 'ach' | 'unlock' | 'info' | 'error' | 'level'; text: string; icon?: string }
type Game = {
  pid: number
  ov: Any
  meta: Any
  refresh: () => Promise<void>
  toast: (t: Omit<Toast, 'id'>) => void
  reward: (res: Any) => void
  setPid: (id: number | null) => void
}
export const GameCtx = createContext<Game | null>(null)
export const useGame = () => {
  const g = useContext(GameCtx)
  if (!g) throw new Error('GameCtx missing')
  return g
}

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([])
  const n = useRef(0)
  const toast = useCallback((t: Omit<Toast, 'id'>) => {
    const id = ++n.current
    setToasts((ts) => [...ts.slice(-4), { ...t, id }])
    setTimeout(() => setToasts((ts) => ts.filter((x) => x.id !== id)), t.kind === 'ach' ? 6000 : 3800)
  }, [])
  return { toasts, toast }
}

export function Toasts({ toasts }: { toasts: Toast[] }) {
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind}`}>
          {t.icon && <span className="toast-icon">{t.icon}</span>}
          <span>{t.text}</span>
        </div>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ data hook
export function useApi<T = Any>(url: string | null, deps: unknown[] = []): { data: T | null; error: unknown | null; loading: boolean; reload: () => void; setData: (d: T) => void } {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<unknown | null>(null)
  const [loading, setLoading] = useState(false)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    if (!url) return
    let alive = true
    setLoading(true)
    setError(null)
    get(url)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(e))
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, tick, ...deps])
  return { data, error, loading, reload: () => setTick((t) => t + 1), setData }
}

// ------------------------------------------------------------------ primitives
export function Card({ title, children, className = '', right, icon }: { title?: ReactNode; children: ReactNode; className?: string; right?: ReactNode; icon?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <header className="card-h">
          <h3>
            {icon && <span className="card-icon">{icon}</span>}
            {title}
          </h3>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

export function Btn({ children, onClick, kind = 'primary', disabled, small, title, type }: { children: ReactNode; onClick?: () => void; kind?: 'primary' | 'ghost' | 'danger' | 'success' | 'warn'; disabled?: boolean; small?: boolean; title?: string; type?: 'submit' | 'button' }) {
  return (
    <button type={type || 'button'} className={`btn btn-${kind} ${small ? 'btn-sm' : ''}`} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  )
}

export const Pill = ({ children, kind = '' }: { children: ReactNode; kind?: string }) => <span className={`pill ${kind}`}>{children}</span>

export function Bar({ value, max = 1, color, label, thin }: { value: number; max?: number; color?: string; label?: ReactNode; thin?: boolean }) {
  const pct = Math.max(0, Math.min(100, (100 * value) / (max || 1)))
  return (
    <div className={`bar ${thin ? 'bar-thin' : ''}`} title={label ? undefined : `${Math.round(pct)}%`}>
      <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      {label && <span className="bar-label">{label}</span>}
    </div>
  )
}

export const Loading = ({ text = 'Loading…' }: { text?: string }) => (
  <div className="loading">
    <span className="spinner" /> {text}
  </div>
)

export function ErrorBox({ error }: { error: unknown | null }) {
  const { t } = useI18n()
  return error ? <div className="error-box" role="alert">⚠️ {localizedApiError(error, t)}</div> : null
}

export function Mentor({ name, icon, children, color }: { name: string; icon?: string; children: ReactNode; color?: string }) {
  return (
    <div className="mentor" style={{ borderColor: color }}>
      <div className="mentor-avatar" style={{ background: color }}>
        {icon || '🧑‍🔬'}
      </div>
      <div className="mentor-body">
        <div className="mentor-name">{name}</div>
        <div>{children}</div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ text rendering (tiny markdown subset)
function inline(text: string, key: number): ReactNode {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g)
  return (
    <span key={key}>
      {parts.map((p, i) => {
        if (p.startsWith('`') && p.endsWith('`')) return <code key={i}>{p.slice(1, -1)}</code>
        if (p.startsWith('**') && p.endsWith('**')) return <strong key={i}>{p.slice(2, -2)}</strong>
        if (p.startsWith('*') && p.endsWith('*') && p.length > 2) return <em key={i}>{p.slice(1, -1)}</em>
        return p
      })}
    </span>
  )
}

export function Rich({ text, className = '' }: { text: string; className?: string }) {
  if (!text) return null
  const blocks = text.split(/```(?:python|py)?\n?/)
  return (
    <div className={`rich ${className}`}>
      {blocks.map((b, i) =>
        i % 2 === 1 ? (
          <CodeBlock key={i} code={b.replace(/\n$/, '')} />
        ) : (
          b
            .split(/\n{2,}/)
            .filter((x) => x.trim())
            .map((para, j) => (
              <p key={`${i}-${j}`}>
                {para.split('\n').map((line, k) => (
                  <span key={k}>
                    {k > 0 && <br />}
                    {inline(line, k)}
                  </span>
                ))}
              </p>
            ))
        ),
      )}
    </div>
  )
}

const PY_KW = new Set('def return if elif else for while in import from as class with try except finally raise not and or is None True False lambda pass break continue yield global print len range'.split(' '))

export function highlight(code: string): ReactNode[] {
  const out: ReactNode[] = []
  const re = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|(\b[A-Za-z_][A-Za-z0-9_]*\b)/gm
  let last = 0
  let m: RegExpExecArray | null
  let i = 0
  while ((m = re.exec(code))) {
    if (m.index > last) out.push(code.slice(last, m.index))
    const [tok] = m
    const cls = m[1] ? 'tk-com' : m[2] ? 'tk-str' : m[3] ? 'tk-num' : PY_KW.has(tok) ? 'tk-kw' : /^[A-Z]/.test(tok) ? 'tk-cls' : ''
    out.push(cls ? <span key={i++} className={cls}>{tok}</span> : tok)
    last = m.index + tok.length
  }
  out.push(code.slice(last))
  return out
}

export function CodeBlock({ code, title }: { code: string; title?: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="codeblock">
      <div className="codeblock-h">
        <span>{title || 'python'}</span>
        <button
          className="link"
          onClick={() => {
            navigator.clipboard?.writeText(code)
            setCopied(true)
            setTimeout(() => setCopied(false), 1200)
          }}
        >
          {copied ? 'copied ✓' : 'copy'}
        </button>
      </div>
      <pre>
        <code>{highlight(code)}</code>
      </pre>
    </div>
  )
}

export function Tabs({ tabs, value, onChange }: { tabs: { id: string; label: ReactNode }[]; value: string; onChange: (id: string) => void }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} className={`tab ${value === t.id ? 'active' : ''}`} onClick={() => onChange(t.id)}>
          {t.label}
        </button>
      ))}
    </div>
  )
}

export function Diagnosis({ items }: { items: Any[] }) {
  if (!items?.length) return null
  const icon: Record<string, string> = { danger: '🛑', warn: '⚠️', info: 'ℹ️', ok: '✅' }
  return (
    <div className="diagnosis">
      {items.map((d: Any, i: number) => {
        const item = typeof d === 'string' ? { level: 'info', text: d } : d
        return (
          <div key={i} className={`diag diag-${item.level}`}>
            <span>{icon[item.level] || 'ℹ️'}</span>
            <span>{item.text}</span>
          </div>
        )
      })}
    </div>
  )
}

export const fmt = (v: unknown, d = 3): string => {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') {
    if (!isFinite(v)) return '—'
    if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
    return Number(v.toFixed(d)).toString()
  }
  return String(v)
}

export const pct = (v: number | null | undefined, d = 1) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`)

export function Slider({ label, value, min, max, step = 1, onChange, fmt: f, hint }: { label: ReactNode; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; fmt?: (v: number) => string; hint?: string }) {
  return (
    <label className="slider" title={hint}>
      <span className="slider-l">
        {label} <b>{f ? f(value) : value}</b>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

export function Select({ label, value, options, onChange, hint }: { label?: ReactNode; value: string; options: { value: string; label: string }[] | string[]; onChange: (v: string) => void; hint?: string }) {
  const opts = (options as Any[]).map((o) => (typeof o === 'string' ? { value: o, label: o } : o))
  return (
    <label className="field" title={hint}>
      {label && <span className="field-l">{label}</span>}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {opts.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  )
}

export function Toggle({ label, checked, onChange, hint }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="toggle" title={hint}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-ui" />
      <span>{label}</span>
    </label>
  )
}

export function Modal({ children, onClose, wide }: { children: ReactNode; onClose: () => void; wide?: boolean }) {
  useEffect(() => {
    const on = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [onClose])
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className={`modal ${wide ? 'modal-wide' : ''}`} onClick={(e) => e.stopPropagation()}>
        <button className="modal-x" onClick={onClose} aria-label="Close">
          ✕
        </button>
        {children}
      </div>
    </div>
  )
}
