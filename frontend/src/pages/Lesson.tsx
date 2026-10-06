import { useCallback, useEffect, useState } from 'react'
import { get, post, qs, type Any } from '../api'
import { PredictionCard } from '../labs/Prediction'
import { Bar, Btn, Card, CodeBlock, ErrorBox, Loading, Mentor, Pill, Rich, go, useGame } from '../ui'
import { WIDGETS, Widget } from '../widgets'
import { StatusPill } from './Tree'

const CARD_LABEL: Record<string, string> = {
  teach: 'Learn', mini_lesson: 'A simpler angle', recap: 'Recap', visual: 'Visualise', example: 'Worked example', code: 'In code',
  guided: 'Guided practice', predict: 'Predict', practice: 'Practice', practice_easy: 'Easy practice', challenge: 'Challenge', reflect: 'Reflect',
}
const QUESTION_CARDS = new Set(['guided', 'practice', 'practice_easy', 'challenge'])

function QuestionCard({ cid, purpose, onDone, mode }: { cid: string; purpose: string; onDone: (res: Any) => void; mode: Any }) {
  const { pid, reward } = useGame()
  const [q, setQ] = useState<Any>(null)
  const [choice, setChoice] = useState<number | null>(null)
  const [num, setNum] = useState('')
  const [hint, setHint] = useState<string | null>(null)
  const [hintUsed, setHintUsed] = useState(false)
  const [res, setRes] = useState<Any>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    get(`/api/p/${pid}/lesson/${cid}/question${qs({ purpose })}`).then((d) => { setQ(d); if (d.hint) setHint(d.hint) }).catch((e) => setErr(e.message))
  }, [pid, cid, purpose])
  const submit = useCallback(async () => {
    if (!q || res) return
    const response = q.type === 'mcq' ? choice : num
    if (response === null || response === '') return
    try {
      const r = await post(`/api/p/${pid}/lesson/${cid}/answer`, { key: q.key, response, hint_used: hintUsed, purpose })
      setRes(r)
      if (r.hint) setHint(r.hint)
      reward(r)
    } catch (e: Any) { setErr(e.message) }
  }, [q, res, choice, num, pid, cid, hintUsed, purpose, reward])
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (!q || res || q.type !== 'mcq' || (e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') return
      const n = Number(e.key)
      if (n >= 1 && n <= q.options.length) setChoice(n - 1)
      if (e.key === 'Enter') submit()
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [q, res, submit])
  const askHint = async () => {
    try {
      const h = await get(`/api/p/${pid}/lesson/${cid}/hint${qs({ key: q.key })}`)
      setHint(h.hint); setHintUsed(true)
    } catch (e: Any) { setErr(e.message) }
  }
  if (err) return <ErrorBox error={err} />
  if (!q) return <Loading />
  const diffLabel = ['', 'easy', 'medium', 'hard'][q.difficulty]
  return (
    <div className="col">
      <div className="row"><Pill kind={q.difficulty === 3 ? 'red' : q.difficulty === 2 ? 'amber' : 'green'}>{diffLabel}</Pill>{q.key.startsWith('g:') && <Pill kind="violet">freshly generated</Pill>}{purpose === 'guided' && <Pill kind="cyan">hint included — no penalty</Pill>}</div>
      <Rich text={q.prompt} />
      {q.type === 'mcq' ? q.options.map((o: string, i: number) => {
        const cls = res ? (i === res.answer ? 'right' : i === choice ? 'wrong' : '') : choice === i ? 'sel' : ''
        return <button key={i} className={`option ${cls}`} disabled={!!res} onClick={() => setChoice(i)}><span className="key">{i + 1}</span>{o}</button>
      }) : (
        <input type="text" inputMode="decimal" placeholder="Type a number" value={num} disabled={!!res} onChange={(e) => setNum(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && submit()} style={{ maxWidth: 220 }} />
      )}
      {hint && <div className="info-box">💡 {hint}</div>}
      {!res && (
        <div className="row">
          <Btn onClick={submit} disabled={q.type === 'mcq' ? choice === null : !num}>Check answer</Btn>
          {!hint && q.hint_available && mode.hints === 'on_request' && <Btn kind="ghost" onClick={askHint}>💡 Hint{q.hint_cost ? ` (−${Math.round(q.hint_cost * 100)}% XP)` : ''}</Btn>}
        </div>
      )}
      {res && (
        <div className="col">
          <div className={res.correct ? 'ok-box' : 'error-box'}>
            <b>{res.correct ? '✅ Correct!' : `❌ Not quite. Answer: ${res.answer_text}`}</b>
            <Rich text={res.explanation} />
          </div>
          <div className="row">
            <small className="muted">Mastery estimate</small>
            <div style={{ flex: 1, maxWidth: 260 }}><Bar value={res.p} label={`${Math.round(res.p_before * 100)}% → ${Math.round(res.p * 100)}%`} /></div>
            <StatusPill status={res.status} />
          </div>
          {res.adaptation && <div className="warn-box">🧠 {res.adaptation}</div>}
          <div><Btn onClick={() => onDone(res)}>Continue →</Btn></div>
        </div>
      )}
    </div>
  )
}

function ReflectCard({ cid, prompt, onDone }: { cid: string; prompt: string; onDone: () => void }) {
  const { pid, reward } = useGame()
  const [text, setText] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const save = async () => {
    try { reward(await post(`/api/p/${pid}/lesson/${cid}/reflect`, { text })); onDone() } catch (e: Any) { setErr(e.message) }
  }
  return (
    <div className="col">
      <p><b>{prompt}</b></p>
      <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} placeholder="Explain it in your own words — writing it down is what makes it stick." />
      <ErrorBox error={err} />
      <div className="row"><Btn onClick={save} disabled={text.trim().length < 10}>Save reflection (+10 XP)</Btn><Btn kind="ghost" onClick={onDone}>Skip</Btn></div>
    </div>
  )
}

export default function Lesson({ cid }: { cid: string }) {
  const { pid, ov } = useGame()
  const [lesson, setLesson] = useState<Any>(null)
  const [cards, setCards] = useState<string[]>([])
  const [idx, setIdx] = useState(0)
  const [err, setErr] = useState<string | null>(null)
  const [adapted, setAdapted] = useState(false)
  const [results, setResults] = useState<Any[]>([])
  const [nonce, setNonce] = useState(0)
  useEffect(() => {
    setLesson(null); setErr(null); setIdx(0); setResults([]); setAdapted(false)
    get(`/api/p/${pid}/lesson/${cid}`).then((l) => { setLesson(l); setCards(l.plan.cards) }).catch((e) => setErr(e.message))
  }, [pid, cid, nonce])
  if (err) return <div className="stack"><Btn kind="ghost" small onClick={() => go('/tree')}>← Knowledge tree</Btn><div className="error-box">🔒 {err}</div></div>
  if (!lesson) return <Loading />
  const c = lesson.concept
  const m = lesson.mentor
  const card = cards[idx]
  const next = () => setIdx((i) => i + 1)
  const onAnswer = (res: Any) => {
    setResults((r) => [...r, res])
    // live adaptation: if the learner starts struggling mid-lesson, insert a simpler explanation + easier question right away
    if (res.struggling && !adapted && lesson.plan.kind !== 'remediation') {
      setAdapted(true)
      setCards((cs) => [...cs.slice(0, idx + 1), 'mini_lesson', ...(c.visual ? ['visual'] : []), 'practice_easy', ...cs.slice(idx + 1)])
    }
    next()
  }
  const done = idx >= cards.length
  const last = results[results.length - 1]
  return (
    <div className="stack" style={{ maxWidth: 980 }}>
      <div className="topbar">
        <Btn kind="ghost" small onClick={() => go(`/area/${c.area}`)}>← {m.area}</Btn>
        <div>
          <div className="kicker">{c.branch}</div>
          <h1 style={{ margin: 0 }}>{c.name}</h1>
        </div>
        <div className="spacer" />
        <StatusPill status={last?.status || lesson.state.status} />
        <div style={{ width: 140 }}><Bar value={last?.p ?? lesson.state.p} label={`mastery ${Math.round((last?.p ?? lesson.state.p) * 100)}%`} /></div>
      </div>
      {lesson.plan.message && <div className={lesson.plan.kind === 'remediation' ? 'warn-box' : 'info-box'}>{lesson.plan.kind === 'remediation' ? '🧩 ' : '⚡ '}{lesson.plan.message}</div>}
      <div className="lesson-steps" aria-label="Lesson progress">
        {cards.map((k, i) => <div key={i} className={`lesson-step ${i < idx ? 'done' : i === idx ? 'cur' : ''}`} title={CARD_LABEL[k]} />)}
      </div>
      {!done && (
        <Card className="lesson-card" title={<><span className="kicker" style={{ margin: 0 }}>{idx + 1}/{cards.length}</span>&nbsp; {CARD_LABEL[card]}</>}>
          {(card === 'teach' || card === 'recap') && (
            <div className="col">
              <Mentor name={m.name} icon={m.icon} color={m.color}><Rich text={card === 'recap' ? c.explain.split('. ').slice(0, 2).join('. ') + '.' : c.explain} /></Mentor>
              {c.analogy && card === 'teach' && <div className="info-box">🪄 <b>Analogy:</b> {c.analogy}</div>}
              {c.prereqs.length > 0 && card === 'teach' && <small className="muted">Builds on: {c.prereqs.map((p: string) => <a key={p} href={`#/lesson/${p}`} style={{ marginRight: 8 }}>{p.replace(/_/g, ' ')}</a>)}</small>}
              <div><Btn onClick={next}>Got it →</Btn></div>
            </div>
          )}
          {card === 'mini_lesson' && (
            <div className="col">
              <Mentor name={m.name} icon={m.icon} color={m.color}>
                <p>Let's try a different angle — no rush.</p>
                {c.analogy && <p>🪄 {c.analogy}</p>}
              </Mentor>
              <Rich text={c.explain} />
              {c.example && <div className="info-box"><b>Example:</b> <Rich text={c.example} /></div>}
              <div><Btn onClick={next}>I'm ready for an easier question →</Btn></div>
            </div>
          )}
          {card === 'visual' && c.visual && (
            <div className="col">
              <div className="kicker">{WIDGETS[c.visual]?.title}</div>
              <Widget name={c.visual} />
              <div><Btn onClick={next}>Continue →</Btn></div>
            </div>
          )}
          {card === 'example' && <div className="col"><Rich text={c.example} /><div><Btn onClick={next}>Continue →</Btn></div></div>}
          {card === 'code' && (
            <div className="col">
              {c.code ? <CodeBlock code={c.code} /> : <p className="muted">This concept is practised in the Code Dojo with real, auto-graded Python.</p>}
              {lesson.linked.exercises.length > 0 && <div className="row">{lesson.linked.exercises.map((e: string) => <Btn key={e} kind="ghost" small onClick={() => go(`/dojo/${e}`)}>⌨️ Open exercise {e.replace('ex_', '')}</Btn>)}</div>}
              <div><Btn onClick={next}>Continue →</Btn></div>
            </div>
          )}
          {QUESTION_CARDS.has(card) && <QuestionCard key={`${idx}-${card}`} cid={cid} purpose={card === 'practice_easy' ? 'practice_easy' : card} onDone={onAnswer} mode={lesson.mode} />}
          {card === 'predict' && lesson.prediction && (
            <div className="col"><PredictionCard exp={lesson.prediction} /><div><Btn kind="ghost" onClick={next}>Continue →</Btn></div></div>
          )}
          {card === 'reflect' && <ReflectCard cid={cid} prompt={c.reflect} onDone={next} />}
        </Card>
      )}
      {done && (
        <Card title="Lesson complete" icon="🎓" className="glow">
          <div className="row" style={{ gap: '1.5rem' }}>
            <div><div className="metric-l">questions</div><div className="metric-v">{results.filter((r) => r.correct).length} / {results.length} correct</div></div>
            <div><div className="metric-l">mastery</div><div className="metric-v">{Math.round((last?.p ?? lesson.state.p) * 100)}%</div></div>
            <StatusPill status={last?.status || lesson.state.status} />
          </div>
          {last?.struggling && <div className="warn-box" style={{ marginTop: '0.8rem' }}>This one is still tricky. Come back to it — next time the lesson starts with a simpler explanation and easier questions, and the harder question returns once you've recovered.</div>}
          {last && !last.struggling && last.p >= 0.6 && <div className="ok-box" style={{ marginTop: '0.8rem' }}>You're proficient. It will come back for spaced review in a day or so — revisiting just before forgetting is the most efficient way to make it permanent.</div>}
          <div className="row" style={{ marginTop: '1rem' }}>
            <Btn onClick={() => setNonce(nonce + 1)}>{last?.struggling ? 'Try the simpler version' : 'Practise again'}</Btn>
            {lesson.linked.missions.map((mid: string) => <Btn key={mid} kind="ghost" onClick={() => go(`/mission/${mid}`)}>🎯 Related mission</Btn>)}
            {lesson.linked.bosses.map((b: string) => <Btn key={b} kind="danger" onClick={() => go(`/boss/${b}`)}>⚔️ Related boss</Btn>)}
            <Btn kind="ghost" onClick={() => go('/')}>Campus</Btn>
          </div>
          {ov.recommendations[0] && <p className="muted" style={{ marginTop: '0.8rem' }}>Suggested next: <a href={`#${ov.recommendations[0].kind === 'mission' ? '/mission/' : ov.recommendations[0].kind === 'boss' ? '/boss/' : '/lesson/'}${ov.recommendations[0].id}`}>{ov.recommendations[0].title}</a></p>}
        </Card>
      )}
    </div>
  )
}
