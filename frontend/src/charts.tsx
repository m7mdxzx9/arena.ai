// Dependency-free SVG charts. Every chart plots exactly the numbers it receives — no smoothing, no invented points.
import { useMemo, useState, type ReactNode } from 'react'
import type { Any } from './api'

export const PALETTE = ['#22d3ee', '#f472b6', '#a3e635', '#fbbf24', '#a78bfa', '#fb7185', '#34d399', '#60a5fa', '#f97316', '#e879f9']

const niceTicks = (lo: number, hi: number, n = 5): number[] => {
  if (!isFinite(lo) || !isFinite(hi)) return []
  if (hi === lo) return [lo]
  const span = hi - lo
  const step0 = span / n
  const mag = Math.pow(10, Math.floor(Math.log10(step0)))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || step0
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)))
  return out
}
const short = (v: number) => {
  const a = Math.abs(v)
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`
  if (a >= 1e4) return `${(v / 1e3).toFixed(0)}k`
  if (a >= 1000) return `${(v / 1e3).toFixed(1)}k`
  if (a > 0 && a < 0.01) return v.toExponential(0)
  return String(Number(v.toFixed(3)))
}

type Series = { name: string; values: (number | null)[]; color?: string; dashed?: boolean }
export function LineChart({ series, x, height = 220, yLabel, xLabel, yDomain, markX, logY }: { series: Series[]; x?: number[]; height?: number; yLabel?: string; xLabel?: string; yDomain?: [number, number]; markX?: number | null; logY?: boolean }) {
  const W = 560, H = height, L = 48, R = 12, T = 12, B = 34
  const [hover, setHover] = useState<number | null>(null)
  const n = Math.max(...series.map((s) => s.values.length), 1)
  const xs = x || Array.from({ length: n }, (_, i) => i + 1)
  const tf = (v: number) => (logY ? Math.log10(Math.max(v, 1e-6)) : v)
  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null && isFinite(v))).map(tf)
  let [lo, hi] = yDomain ? [tf(yDomain[0]), tf(yDomain[1])] : [Math.min(...all), Math.max(...all)]
  if (!all.length) [lo, hi] = [0, 1]
  if (lo === hi) [lo, hi] = [lo - 0.5, hi + 0.5]
  const pad = yDomain ? 0 : (hi - lo) * 0.06
  lo -= pad
  hi += pad
  const x0 = xs[0], x1 = xs[xs.length - 1] === x0 ? x0 + 1 : xs[xs.length - 1]
  const sx = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R)
  const sy = (v: number) => T + (1 - (tf(v) - lo) / (hi - lo)) * (H - T - B)
  const ticks = logY ? niceTicks(lo, hi, 4).map((t) => Math.pow(10, t)) : niceTicks(lo, hi, 4)
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
          const px = ((e.clientX - r.left) / r.width) * W
          const v = x0 + ((px - L) / (W - L - R)) * (x1 - x0)
          let best = 0
          xs.forEach((xv, i) => { if (Math.abs(xv - v) < Math.abs(xs[best] - v)) best = i })
          setHover(best)
        }}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={L - 6} y={sy(t) + 4} textAnchor="end" className="tick">{short(t)}</text>
          </g>
        ))}
        {niceTicks(x0, x1, 6).map((t) => (
          <text key={t} x={sx(t)} y={H - B + 16} textAnchor="middle" className="tick">{short(t)}</text>
        ))}
        {markX != null && <line x1={sx(markX)} x2={sx(markX)} y1={T} y2={H - B} className="mark" />}
        {series.map((s, si) => {
          let d = ''
          s.values.forEach((v, i) => {
            if (v === null || !isFinite(v)) return
            d += `${d && s.values[i - 1] !== null ? 'L' : 'M'}${sx(xs[i]).toFixed(1)},${sy(v).toFixed(1)}`
          })
          return <path key={si} d={d} fill="none" stroke={s.color || PALETTE[si]} strokeWidth={2} strokeDasharray={s.dashed ? '5 4' : undefined} />
        })}
        {hover !== null && (
          <g>
            <line x1={sx(xs[hover])} x2={sx(xs[hover])} y1={T} y2={H - B} className="hover-line" />
            {series.map((s, si) => s.values[hover] != null && isFinite(s.values[hover] as number) && (
              <circle key={si} cx={sx(xs[hover])} cy={sy(s.values[hover] as number)} r={3.5} fill={s.color || PALETTE[si]} />
            ))}
          </g>
        )}
        {yLabel && <text x={12} y={T + (H - T - B) / 2} transform={`rotate(-90 12 ${T + (H - T - B) / 2})`} textAnchor="middle" className="axis-label">{yLabel}</text>}
        {xLabel && <text x={L + (W - L - R) / 2} y={H - 4} textAnchor="middle" className="axis-label">{xLabel}</text>}
      </svg>
      <div className="legend">
        {series.map((s, si) => (
          <span key={si}><i style={{ background: s.color || PALETTE[si] }} />{s.name}{hover !== null && s.values[hover] != null ? `: ${short(s.values[hover] as number)}` : ''}</span>
        ))}
        {hover !== null && <span className="muted">{xLabel || 'x'} = {short(xs[hover])}</span>}
      </div>
    </div>
  )
}

export function BarChart({ items, max, height = 22, format = short, unit = '' }: { items: { label: string; value: number; color?: string; note?: string }[]; max?: number; height?: number; format?: (v: number) => string; unit?: string }) {
  const m = max ?? Math.max(...items.map((i) => Math.abs(i.value)), 1e-9)
  const hasNeg = items.some((i) => i.value < 0)
  return (
    <div className="barchart">
      {items.map((it, i) => (
        <div key={i} className="barchart-row" title={it.note}>
          <span className="barchart-label">{it.label}</span>
          <div className="barchart-track" style={{ height }}>
            {hasNeg && <div className="barchart-zero" />}
            <div className="barchart-fill" style={{
              width: `${(Math.abs(it.value) / m) * (hasNeg ? 50 : 100)}%`,
              left: hasNeg ? (it.value < 0 ? `${50 - (Math.abs(it.value) / m) * 50}%` : '50%') : 0,
              background: it.color || (it.value < 0 ? '#fb7185' : PALETTE[0]),
            }} />
          </div>
          <span className="barchart-val">{format(it.value)}{unit}</span>
        </div>
      ))}
    </div>
  )
}

export function Histogram({ counts, edges, height = 160, color = PALETTE[0], marks = [] }: { counts: number[]; edges: number[]; height?: number; color?: string; marks?: { x: number; label: string; color: string }[] }) {
  const W = 520, H = height, L = 36, B = 22, T = 8
  const m = Math.max(...counts, 1)
  const lo = edges[0], hi = edges[edges.length - 1]
  const sx = (v: number) => L + ((v - lo) / (hi - lo || 1)) * (W - L - 8)
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`}>
        {counts.map((c, i) => (
          <rect key={i} x={sx(edges[i]) + 0.5} width={Math.max(0, sx(edges[i + 1]) - sx(edges[i]) - 1)} y={T + (1 - c / m) * (H - T - B)} height={(c / m) * (H - T - B)} fill={color} opacity={0.85}>
            <title>{`${short(edges[i])} – ${short(edges[i + 1])}: ${c}`}</title>
          </rect>
        ))}
        {niceTicks(lo, hi, 5).map((t) => <text key={t} x={sx(t)} y={H - 6} textAnchor="middle" className="tick">{short(t)}</text>)}
        <text x={L - 4} y={T + 8} textAnchor="end" className="tick">{m}</text>
        {marks.map((mk, i) => (
          <g key={i}>
            <line x1={sx(mk.x)} x2={sx(mk.x)} y1={T} y2={H - B} stroke={mk.color} strokeWidth={2} strokeDasharray="4 3" />
            <text x={sx(mk.x) + 4} y={T + 12 + i * 12} className="tick" fill={mk.color}>{mk.label}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}

export function Heatmap({ matrix, rows, cols, color, showValues = true, cell = 34, onCell, highlight, fmtv = (v: number) => short(v), domain }: { matrix: (number | null)[][]; rows?: string[]; cols?: string[]; color?: (t: number, v: number) => string; showValues?: boolean; cell?: number; onCell?: (r: number, c: number) => void; highlight?: [number, number] | null; fmtv?: (v: number) => string; domain?: [number, number] }) {
  const flat = matrix.flat().filter((v): v is number => v !== null && isFinite(v))
  const [lo, hi] = domain || [Math.min(...flat, 0), Math.max(...flat, 1e-9)]
  const col = color || ((t: number) => `rgba(34,211,238,${0.08 + 0.85 * t})`)
  const labelW = rows ? Math.min(120, Math.max(...rows.map((r) => r.length)) * 7 + 8) : 0
  const labelH = cols ? Math.min(90, Math.max(...cols.map((c) => c.length)) * 6 + 10) : 0
  const W = labelW + (matrix[0]?.length || 0) * cell, H = labelH + matrix.length * cell
  return (
    <div className="chart heatmap" style={{ maxWidth: Math.max(W, 120) * 1.4 }}>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {cols?.map((c, j) => (
          <text key={j} x={labelW + j * cell + cell / 2} y={labelH - 6} transform={`rotate(-40 ${labelW + j * cell + cell / 2} ${labelH - 6})`} className="tick" textAnchor="start">{c}</text>
        ))}
        {matrix.map((row, i) => (
          <g key={i}>
            {rows && <text x={labelW - 6} y={labelH + i * cell + cell / 2 + 4} textAnchor="end" className="tick">{rows[i]}</text>}
            {row.map((v, j) => {
              const t = v === null ? 0 : (v - lo) / (hi - lo || 1)
              const hl = highlight && highlight[0] === i && highlight[1] === j
              return (
                <g key={j} onClick={() => onCell?.(i, j)} style={{ cursor: onCell ? 'pointer' : undefined }}>
                  <rect x={labelW + j * cell + 1} y={labelH + i * cell + 1} width={cell - 2} height={cell - 2} rx={4} fill={v === null ? '#1e293b' : col(Math.max(0, Math.min(1, t)), v)} stroke={hl ? '#fff' : 'none'} strokeWidth={2} />
                  {showValues && v !== null && <text x={labelW + j * cell + cell / 2} y={labelH + i * cell + cell / 2 + 4} textAnchor="middle" className="cell-val">{fmtv(v)}</text>}
                </g>
              )
            })}
          </g>
        ))}
      </svg>
    </div>
  )
}

export const divergent = (t: number, v: number) => (v >= 0 ? `rgba(34,211,238,${0.1 + 0.85 * Math.min(1, Math.abs(v))})` : `rgba(244,114,182,${0.1 + 0.85 * Math.min(1, Math.abs(v))})`)

export function Scatter({ x, y, c, line, xLabel, yLabel, size = 3.2, height = 260, classes, extra, residuals }: { x: number[]; y: number[]; c?: (number | string)[]; line?: { slope: number; intercept: number; color?: string } | null; xLabel?: string; yLabel?: string; size?: number; height?: number; classes?: string[]; extra?: (sx: (v: number) => number, sy: (v: number) => number) => ReactNode; residuals?: boolean }) {
  const W = 520, H = height, L = 44, R = 10, T = 10, B = 32
  const vx = x.filter(isFinite), vy = y.filter(isFinite)
  let [x0, x1] = [Math.min(...vx), Math.max(...vx)]
  let [y0, y1] = [Math.min(...vy), Math.max(...vy)]
  const px = (x1 - x0) * 0.05 || 1, py = (y1 - y0) * 0.06 || 1
  x0 -= px; x1 += px; y0 -= py; y1 += py
  const sx = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R)
  const sy = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)
  const cats = useMemo(() => (c ? Array.from(new Set(c.map(String))).sort() : []), [c])
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`}>
        {niceTicks(y0, y1, 4).map((t) => (
          <g key={t}><line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} className="grid" /><text x={L - 5} y={sy(t) + 4} textAnchor="end" className="tick">{short(t)}</text></g>
        ))}
        {niceTicks(x0, x1, 6).map((t) => <text key={t} x={sx(t)} y={H - B + 15} textAnchor="middle" className="tick">{short(t)}</text>)}
        {line && residuals && x.map((xv, i) => <line key={`r${i}`} x1={sx(xv)} x2={sx(xv)} y1={sy(y[i])} y2={sy(line.slope * xv + line.intercept)} stroke="#fb7185" strokeOpacity={0.5} />)}
        {x.map((xv, i) => isFinite(xv) && isFinite(y[i]) && (
          <circle key={i} cx={sx(xv)} cy={sy(y[i])} r={size} fill={c ? PALETTE[cats.indexOf(String(c[i])) % PALETTE.length] : PALETTE[0]} fillOpacity={0.8} />
        ))}
        {line && <line x1={sx(x0)} x2={sx(x1)} y1={sy(line.slope * x0 + line.intercept)} y2={sy(line.slope * x1 + line.intercept)} stroke={line.color || '#fbbf24'} strokeWidth={2.5} />}
        {extra?.(sx, sy)}
        {xLabel && <text x={L + (W - L - R) / 2} y={H - 3} textAnchor="middle" className="axis-label">{xLabel}</text>}
        {yLabel && <text x={11} y={T + (H - T - B) / 2} transform={`rotate(-90 11 ${T + (H - T - B) / 2})`} textAnchor="middle" className="axis-label">{yLabel}</text>}
      </svg>
      {c && cats.length <= 10 && (
        <div className="legend">{cats.map((k, i) => <span key={k}><i style={{ background: PALETTE[i % PALETTE.length] }} />{classes?.[Number(k)] ?? k}</span>)}</div>
      )}
    </div>
  )
}

/** Decision boundary: background = model's real predicted probability (binary) or class (multi) on a grid. */
export function BoundaryPlot({ grid, points, height = 300, classes }: { grid: Any; points?: { x: number[]; y: number[]; label: (number | string)[] }; height?: number; classes?: string[] }) {
  const W = 360, H = height
  const res = grid.res
  const [x0, x1] = grid.x_range, [y0, y1] = grid.y_range
  const sx = (v: number) => ((v - x0) / (x1 - x0)) * W
  const sy = (v: number) => (1 - (v - y0) / (y1 - y0)) * H
  const cw = W / res, ch = H / res
  const z: number[][] = grid.z
  const multi = (grid.classes && grid.classes.length > 2) || z.flat().some((v) => v > 1.0001)
  const labels = points ? Array.from(new Set(points.label.map(String))).sort() : []
  const fill = (v: number) => {
    if (multi) return PALETTE[Math.round(v) % PALETTE.length]
    const a = v - 0.5
    return a >= 0 ? `rgba(244,114,182,${Math.min(0.75, a * 1.5)})` : `rgba(34,211,238,${Math.min(0.75, -a * 1.5)})`
  }
  return (
    <div className="chart boundary">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {z.map((row, i) => row.map((v, j) => (
          <rect key={`${i}-${j}`} x={j * cw} y={H - (i + 1) * ch} width={cw + 0.6} height={ch + 0.6} fill={fill(v)} opacity={multi ? 0.35 : 1} />
        )))}
        {points && points.x.map((xv, i) => (
          <circle key={i} cx={sx(xv)} cy={sy(points.y[i])} r={3} fill={labels.length === 2 ? (labels.indexOf(String(points.label[i])) === 1 ? '#f472b6' : '#22d3ee') : PALETTE[labels.indexOf(String(points.label[i])) % PALETTE.length]} stroke="#0b1020" strokeWidth={0.8} />
        ))}
      </svg>
      {points && (
        <div className="legend">{labels.map((k, i) => <span key={k}><i style={{ background: labels.length === 2 ? (i === 1 ? '#f472b6' : '#22d3ee') : PALETTE[i % PALETTE.length] }} />{classes?.[i] ?? `class ${k}`}</span>)}
          {!multi && <span className="muted">background = model's predicted probability</span>}</div>
      )}
    </div>
  )
}

export function ConfusionMatrix({ labels, matrix, onCell, selected }: { labels: string[]; matrix: number[][]; onCell?: (r: number, c: number) => void; selected?: [number, number] | null }) {
  const binary = labels.length === 2
  const names = binary ? [['TN', 'FP'], ['FN', 'TP']] : null
  const total = matrix.flat().reduce((a, b) => a + b, 0)
  return (
    <div className="cm">
      <div className="cm-axis-top">Predicted →</div>
      <div className="cm-wrap">
        <div className="cm-axis-left">Actual →</div>
        <table>
          <thead><tr><th />{labels.map((l) => <th key={l}>{l}</th>)}</tr></thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <th>{labels[i]}</th>
                {row.map((v, j) => {
                  const diag = i === j
                  const t = v / (Math.max(...row, 1))
                  return (
                    <td key={j} onClick={() => onCell?.(i, j)} className={`${diag ? 'cm-ok' : 'cm-bad'} ${selected && selected[0] === i && selected[1] === j ? 'cm-sel' : ''}`}
                      style={{ background: diag ? `rgba(52,211,153,${0.12 + 0.6 * t})` : `rgba(251,113,133,${0.08 + 0.6 * t})`, cursor: onCell ? 'pointer' : undefined }}
                      title={`${v} of ${total} test rows: actual ${labels[i]}, predicted ${labels[j]}`}>
                      <b>{v}</b>{names && <small>{names[i][j]}</small>}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function Ring({ value, size = 44, color = '#22d3ee', label }: { value: number; size?: number; color?: string; label?: string }) {
  const r = size / 2 - 4, c = 2 * Math.PI * r
  return (
    <svg width={size} height={size} className="ring">
      <circle cx={size / 2} cy={size / 2} r={r} stroke="#1e293b" strokeWidth={4} fill="none" />
      <circle cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth={4} fill="none" strokeDasharray={`${c * Math.max(0, Math.min(1, value))} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} strokeLinecap="round" />
      <text x="50%" y="54%" textAnchor="middle" dominantBaseline="middle" fontSize={size * 0.26} fill="#e2e8f0">{label ?? Math.round(value * 100)}</text>
    </svg>
  )
}
