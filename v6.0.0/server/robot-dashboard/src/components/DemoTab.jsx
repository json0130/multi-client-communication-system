import { useState, useEffect, useRef, useCallback } from 'react'
import {
  getDemoStatus,
  startDemo, stopDemo, pauseDemo, resumeDemo, nextDemoStep,
  startQaMode, endQaMode, reviseDemo,
  getRobots, chatRobot,
} from '../api'

// Preset reasons for an operator override. Every control click is recorded as a
// supervisor correction on the server; the reason is what turns "they clicked
// Move On here" into "they clicked Move On *because* it was dragging". Presets
// exist because nobody types during a live demo — one click is the whole cost,
// and leaving it blank still records the correction.
const OVERRIDE_REASONS = [
  'dragging on',
  'wrong robot answered',
  'visitors lost interest',
  'running late',
  'robot got it wrong',
  'visitors had more to ask',
]

/** mm:ss for the run clock. */
function clock(sec) {
  if (sec == null) return '—'
  const s = Math.max(0, Math.round(sec))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

// ── Browser TTS ───────────────────────────────────────────────────────────────

// Per-robot voice identity. `prefer` is matched against voice names in order —
// the first available match that no other robot has taken wins, so each robot
// gets a genuinely different voice rather than one voice at four pitches.
// pitch/rate stay as secondary separation when the browser ships few voices.
const ROBOT_VOICE = {
  pepper_01:  { pitch: 1.12, rate: 0.94, prefer: [/aria|jenny|samantha|zira|female/i, /uk english female/i] },
  chatbox_01: { pitch: 1.22, rate: 1.04, prefer: [/guy|eric|mark|david|us english male/i] },
  navel_01:   { pitch: 1.02, rate: 0.88, prefer: [/libby|sonia|hazel|uk english female/i] },
  silbot_01:  { pitch: 0.82, rate: 0.92, prefer: [/ryan|thomas|george|uk english male/i] },
}

// Same identities under the production client_ids, so the hardcoded demo script
// and the dynamic one sound identical.
ROBOT_VOICE.chatbox_jetson_001 = ROBOT_VOICE.chatbox_01
ROBOT_VOICE.navel_001          = ROBOT_VOICE.navel_01
ROBOT_VOICE.pepper_001         = ROBOT_VOICE.pepper_01

// Voices load asynchronously in Chrome: getVoices() returns [] on first call and
// only fills in after `voiceschanged`. Cache them and rebuild the assignment
// whenever the list grows, otherwise the first line of the demo speaks in the
// browser default voice and everything after it does not.
let _voices     = []
let _assignment = null

function loadVoices() {
  const found = window.speechSynthesis?.getVoices?.() || []
  if (found.length !== _voices.length) {
    _voices     = found
    _assignment = null      // rebuild against the fuller list
  }
  return _voices
}

if (typeof window !== 'undefined' && window.speechSynthesis) {
  loadVoices()
  window.speechSynthesis.addEventListener('voiceschanged', loadVoices)
}

/** English voices, best-sounding first. Robotic espeak fallbacks sink to the bottom. */
function englishVoices() {
  return _voices
    .filter(v => v.lang?.toLowerCase().startsWith('en'))
    .sort((a, b) => rankVoice(a) - rankVoice(b))
}

function rankVoice(v) {
  const n = v.name.toLowerCase()
  if (/espeak|pico|festival/.test(n)) return 3
  if (/google|microsoft|natural|neural/.test(n)) return 0
  return v.localService ? 2 : 1
}

/** robotId -> SpeechSynthesisVoice, every robot distinct while voices last. */
function voiceAssignment() {
  if (_assignment) return _assignment

  const pool  = englishVoices()
  const taken = new Set()
  const map   = {}

  for (const [robotId, cfg] of Object.entries(ROBOT_VOICE)) {
    let pick = null
    for (const rx of cfg.prefer || []) {
      pick = pool.find(v => !taken.has(v.name) && rx.test(v.name))
      if (pick) break
    }
    if (!pick) pick = pool.find(v => !taken.has(v.name))   // any unclaimed voice
    if (!pick) pick = pool[Object.keys(map).length % (pool.length || 1)] || null
    if (pick) {
      taken.add(pick.name)
      map[robotId] = pick
    }
  }

  _assignment = map
  if (pool.length) {
    console.log('[TTS] voices:', Object.entries(map).map(([r, v]) => `${r} → ${v.name}`).join(' · '))
  }
  return map
}

/** Deterministic pitch/rate for a robot the table does not know about. */
function fallbackProfile(robotId) {
  let h = 0
  for (let i = 0; i < robotId.length; i++) h = (h * 31 + robotId.charCodeAt(i)) >>> 0
  return { pitch: 0.8 + (h % 50) / 100, rate: 0.88 + ((h >> 5) % 25) / 100, prefer: [] }
}

function browserSpeak(text, robotId, volume = 1) {
  if (!window.speechSynthesis || !text) return
  window.speechSynthesis.cancel()          // stop any current speech first

  const say = () => {
    const utt  = new SpeechSynthesisUtterance(text)
    const cfg  = ROBOT_VOICE[robotId] || fallbackProfile(robotId || '')
    utt.pitch  = cfg.pitch
    utt.rate   = cfg.rate
    utt.volume = volume

    const voice = voiceAssignment()[robotId]
                || englishVoices()[0]
                || _voices[0]
    if (voice) utt.voice = voice

    window.speechSynthesis.speak(utt)
  }

  // First utterance of a fresh tab: wait briefly for the voice list rather than
  // speaking in whatever the browser defaults to.
  if (loadVoices().length) return say()

  const onReady = () => {
    clearTimeout(timer)
    window.speechSynthesis.removeEventListener('voiceschanged', onReady)
    loadVoices()
    say()
  }
  const timer = setTimeout(() => {
    window.speechSynthesis.removeEventListener('voiceschanged', onReady)
    say()
  }, 800)
  window.speechSynthesis.addEventListener('voiceschanged', onReady)
}

// ── Constants ─────────────────────────────────────────────────────────────────

const STATE_COLORS = {
  idle:        { bg: '#374151', text: '#d1d5db' },
  running:     { bg: '#1d4ed8', text: '#eff6ff' },
  waiting_ack: { bg: '#1d4ed8', text: '#eff6ff' },
  qa_window:   { bg: '#6d28d9', text: '#ede9fe' },
  paused:      { bg: '#92400e', text: '#fef3c7' },
  completed:   { bg: '#065f46', text: '#d1fae5' },
  error:       { bg: '#991b1b', text: '#fee2e2' },
}

function ts() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// ── Sub-components ────────────────────────────────────────────────────────────

function RobotSelectorModal({ onClose, onStart }) {
  const [allRobots,   setAllRobots]   = useState([])
  const [loading,     setLoading]     = useState(true)
  const [selectedIds, setSelectedIds] = useState([])  // ordered list
  // Minutes the whole tour should take. Blank is meaningful, not missing: with
  // no budget the server never trims the script on its own, so leaving this
  // empty is how you opt out of clock-driven plan revision entirely.
  const [budgetMin,   setBudgetMin]   = useState('')

  useEffect(() => {
    getRobots()
      .then(data => {
        const online = (data.robots || []).filter(r => r.ws_connected)
        setAllRobots(data.robots || [])
        setSelectedIds(online.map(r => r.client_id))
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const robotMap = Object.fromEntries(allRobots.map(r => [r.client_id, r]))

  const toggle = (id) => {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }

  const moveUp = (i) => {
    if (i === 0) return
    setSelectedIds(prev => {
      const next = [...prev]
      ;[next[i - 1], next[i]] = [next[i], next[i - 1]]
      return next
    })
  }

  const moveDown = (i) => {
    setSelectedIds(prev => {
      if (i === prev.length - 1) return prev
      const next = [...prev]
      ;[next[i], next[i + 1]] = [next[i + 1], next[i]]
      return next
    })
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 520 }} onClick={e => e.stopPropagation()}>

        <div className="modal-title">
          <span>Configure Demo Robots</span>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {loading ? (
          <div className="muted" style={{ padding: '24px', textAlign: 'center' }}>Loading robots…</div>
        ) : (
          <>
            {/* ── Available robots ── */}
            <div style={{ marginBottom: 20 }}>
              <div className="demo-section-title" style={{ marginBottom: 10 }}>Available Robots</div>
              {allRobots.length === 0 && (
                <div className="muted" style={{ fontSize: '0.82rem' }}>No robots registered.</div>
              )}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {allRobots.map(r => {
                  const isOnline  = r.ws_connected
                  const isChecked = selectedIds.includes(r.client_id)
                  return (
                    <label
                      key={r.client_id}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 10, cursor: isOnline ? 'pointer' : 'default',
                        opacity: isOnline ? 1 : 0.4,
                        padding: '6px 10px', borderRadius: 6,
                        background: isChecked ? 'rgba(59,130,246,0.12)' : 'transparent',
                        border: `1px solid ${isChecked ? 'rgba(59,130,246,0.4)' : 'transparent'}`,
                        transition: 'background 0.15s, border 0.15s',
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        disabled={!isOnline}
                        onChange={() => toggle(r.client_id)}
                        style={{ accentColor: '#3b82f6', width: 15, height: 15 }}
                      />
                      <span style={{ flex: 1, fontSize: '0.87rem', fontWeight: 500 }}>
                        {r.robot_name}
                        <span className="muted" style={{ marginLeft: 6, fontWeight: 400, fontSize: '0.78rem' }}>
                          ({r.client_id})
                        </span>
                      </span>
                      <span style={{
                        fontSize: '0.72rem', padding: '2px 7px', borderRadius: 999,
                        background: isOnline ? 'rgba(16,185,129,0.15)' : 'rgba(107,114,128,0.15)',
                        color: isOnline ? '#34d399' : '#9ca3af',
                      }}>
                        {isOnline ? 'online' : 'offline'}
                      </span>
                    </label>
                  )
                })}
              </div>
            </div>

            {/* ── Demo order ── */}
            <div style={{ marginBottom: 24 }}>
              <div className="demo-section-title" style={{ marginBottom: 6 }}>
                Demo Order
                <span className="muted" style={{ marginLeft: 8, fontWeight: 400, fontSize: '0.75rem' }}>
                  first = guide / host
                </span>
              </div>
              {selectedIds.length === 0 ? (
                <div className="muted" style={{ fontSize: '0.82rem', padding: '8px 0' }}>
                  Select at least one robot above.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {selectedIds.map((id, i) => {
                    const r = robotMap[id]
                    return (
                      <div
                        key={id}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 8,
                          padding: '7px 10px', borderRadius: 6,
                          background: 'var(--surface2, rgba(255,255,255,0.04))',
                          border: '1px solid var(--border, rgba(255,255,255,0.08))',
                        }}
                      >
                        <span style={{
                          fontSize: '0.7rem', fontWeight: 600, padding: '2px 7px',
                          borderRadius: 999, minWidth: 44, textAlign: 'center',
                          background: i === 0 ? 'rgba(245,158,11,0.2)' : 'rgba(99,102,241,0.15)',
                          color: i === 0 ? '#fbbf24' : '#a5b4fc',
                        }}>
                          {i === 0 ? 'HOST' : `#${i}`}
                        </span>
                        <span style={{ flex: 1, fontSize: '0.85rem' }}>
                          {r?.robot_name || id}
                          <span className="muted" style={{ marginLeft: 6, fontSize: '0.76rem' }}>({id})</span>
                        </span>
                        <div style={{ display: 'flex', gap: 3 }}>
                          <button
                            className="btn btn-sm"
                            style={{ padding: '2px 7px', fontSize: '0.75rem' }}
                            disabled={i === 0}
                            onClick={() => moveUp(i)}
                            title="Move up"
                          >↑</button>
                          <button
                            className="btn btn-sm"
                            style={{ padding: '2px 7px', fontSize: '0.75rem' }}
                            disabled={i === selectedIds.length - 1}
                            onClick={() => moveDown(i)}
                            title="Move down"
                          >↓</button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* ── Time budget ── */}
            <div style={{ marginBottom: 24 }}>
              <div className="demo-section-title" style={{ marginBottom: 6 }}>
                Time Budget
                <span className="muted" style={{ marginLeft: 8, fontWeight: 400, fontSize: '0.75rem' }}>
                  optional
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <input
                  className="form-input"
                  type="number"
                  min="1"
                  placeholder="e.g. 20"
                  value={budgetMin}
                  onChange={e => setBudgetMin(e.target.value)}
                  style={{ width: 90 }}
                />
                <span className="muted" style={{ fontSize: '0.78rem' }}>
                  minutes. Leave blank and the tour is never shortened automatically.
                </span>
              </div>
            </div>

            {/* ── Actions ── */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button className="btn btn-sm" onClick={onClose}>Cancel</button>
              <button
                className="btn btn-primary btn-sm"
                disabled={selectedIds.length === 0}
                onClick={() => onStart(
                  selectedIds,
                  Number(budgetMin) > 0 ? Number(budgetMin) * 60 : null,
                )}
              >
                Start Demo →
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

/**
 * Run clock, and the projected finish against the budget.
 *
 * The projection extends the pace so far over the steps that remain — the same
 * estimator the server uses for PLAN_REVISE, so the operator sees the number the
 * system is acting on rather than a second opinion. It is crude early in a run,
 * which is why it only shows once a few steps are done.
 */
function RunClock({ status }) {
  const { elapsed_sec: elapsed, time_budget_sec: budget, step_idx: idx, total } = status || {}
  if (elapsed == null) return null

  let overrun = null
  if (budget && idx > 0) {
    overrun = (elapsed + (elapsed / idx) * Math.max(total - idx, 0)) - budget
  }
  const late = overrun != null && overrun > 30

  return (
    <span className="demo-progress" title={budget ? `Budget ${clock(budget)}` : 'No time budget set'}>
      ⏱ {clock(elapsed)}{budget ? ` / ${clock(budget)}` : ''}
      {overrun != null && (
        <span style={{ marginLeft: 6, color: late ? '#f87171' : '#34d399' }}>
          {late ? `~${clock(overrun)} over` : 'on time'}
        </span>
      )}
    </span>
  )
}

function StatePill({ state }) {
  const c = STATE_COLORS[state] || STATE_COLORS.idle
  return (
    <span className="demo-state-pill" style={{ background: c.bg, color: c.text }}>
      {state?.replace(/_/g, ' ') || 'idle'}
    </span>
  )
}

function ChatMessage({ msg }) {
  const isUser   = msg.role === 'user'
  const isSystem = msg.role === 'system'
  return (
    <div className={`chat-msg ${isUser ? 'chat-msg-user' : ''} ${isSystem ? 'chat-msg-system' : ''}`}>
      {!isUser && (
        <div className="chat-msg-header">
          <span className="chat-msg-name">{isSystem ? '— system —' : msg.robot}</span>
          <span className="chat-msg-time muted">{msg.time}</span>
        </div>
      )}
      <div className="chat-bubble">{msg.text}</div>
      {isUser && (
        <div className="chat-msg-header" style={{ justifyContent: 'flex-end' }}>
          <span className="chat-msg-time muted">{msg.time}</span>
          <span className="chat-msg-name" style={{ color: 'var(--muted)' }}>You</span>
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function DemoTab() {
  // ── Demo state ──────────────────────────────────────────────────────────────
  const [status,       setStatus]       = useState(null)
  const [ctrlLoad,     setCtrlLoad]     = useState(false)
  const [showSelector, setShowSelector] = useState(false)
  // Why the operator is overriding. Optional — see OVERRIDE_REASONS.
  const [reason,       setReason]       = useState('')

  // ── TTS state ───────────────────────────────────────────────────────────────
  const [ttsEnabled, setTtsEnabled] = useState(true)
  const ttsRef = useRef(true)                          // readable inside effects without re-running them
  useEffect(() => { ttsRef.current = ttsEnabled }, [ttsEnabled])

  // ── Chat state ──────────────────────────────────────────────────────────────
  const [robots,    setRobots]    = useState([])
  const [targetId,  setTargetId]  = useState('')
  const [messages,  setMessages]  = useState([])
  const [chatInput, setChatInput] = useState('')
  const [sending,   setSending]   = useState(false)
  const [thinking,  setThinking]  = useState(null)  // { robot, time }

  const feedRef      = useRef(null)
  const inputRef     = useRef(null)
  const currentRef   = useRef(null)
  const lastStepKey  = useRef(null)

  // ── Load connected robots once ──────────────────────────────────────────────
  useEffect(() => {
    getRobots().then(data => {
      const connected = (data.robots || []).filter(r => r.ws_connected)
      setRobots(connected)
      if (connected.length > 0) setTargetId(connected[0].client_id)
    }).catch(() => {})
  }, [])

  // ── Poll demo status ────────────────────────────────────────────────────────
  useEffect(() => {
    const poll = async () => {
      try { setStatus(await getDemoStatus()) } catch {}
    }
    poll()
    const id = setInterval(poll, 2000)
    return () => clearInterval(id)
  }, [])

  // ── Auto-add demo speech to chat feed ──────────────────────────────────────
  const addMsg = useCallback(msg => {
    setMessages(prev => [...prev, { id: Date.now() + Math.random(), ...msg }])
  }, [])

  useEffect(() => {
    if (!status) return

    // Use a step-level key for speech states so that running→waiting_ack
    // transitions on the same step don't add the same message twice.
    const isSpeech = status.state === 'running' || status.state === 'waiting_ack'
    const key = isSpeech
      ? `step:${status.step_idx}`
      : `${status.step_idx}:${status.state}`

    if (key === lastStepKey.current) return

    if (isSpeech) {
      // While the server is still generating, status.text is null — don't mark
      // the key as seen yet so the next poll can retry with the real text.
      if (!status.text) return
      const clean = status.text.replace(/\[.*?\]/g, '').trim()
      if (!clean) return
      lastStepKey.current = key   // mark seen only once we have real speech text
      const demoRobotName = robots.find(r => r.client_id === status.robot_id)?.robot_name || status.robot_id || 'Robot'
      addMsg({ role: 'robot', robot: demoRobotName, text: clean, time: ts(), demo: true })
      if (ttsRef.current) browserSpeak(clean, status.robot_id)
      return
    }

    // Non-speech states — always mark as seen immediately
    lastStepKey.current = key
    if (status.state === 'qa_window') {
      addMsg({ role: 'system', text: 'Q&A window open — visitors can ask questions', time: ts() })
    }
    if (status.state === 'completed') {
      addMsg({ role: 'system', text: 'Demo completed ✓', time: ts() })
    }
  }, [status, addMsg])

  // ── Auto-scroll chat feed ───────────────────────────────────────────────────
  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight
  }, [messages])

  // ── Auto-scroll timeline to current step ───────────────────────────────────
  useEffect(() => {
    if (currentRef.current)
      currentRef.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [status?.step_idx])

  // ── Demo controls ───────────────────────────────────────────────────────────
  // Every override carries the currently-selected reason, then clears it — a
  // reason belongs to one click, and a stale one silently mislabels the next
  // correction, which is worse than no label at all.
  const run = async fn => {
    setCtrlLoad(true)
    try { setStatus(await fn(reason)) } catch {}
    finally {
      setReason('')
      setCtrlLoad(false)
    }
  }

  // ── Mid-demo plan revision ──────────────────────────────────────────────────
  const revise = async (kind, robotId) => {
    setCtrlLoad(true)
    try {
      const res = await reviseDemo([{ kind, robot_id: robotId }], reason)
      setStatus(res)
      const applied = res.revision?.applied?.length || 0
      addMsg({
        role: 'system',
        text: applied
          ? `Script revised: ${kind} ${robotId || ''}`.trim()
          // Ops that no longer apply come back ignored rather than as errors —
          // by the time someone says "skip that", it may already be over.
          : `Revision had no effect: ${res.revision?.ignored?.[0]?.why || kind}`,
        time: ts(),
      })
    } catch (e) {
      addMsg({ role: 'system', text: `Revision failed: ${e.message}`, time: ts() })
    } finally {
      setReason('')
      setCtrlLoad(false)
    }
  }

  // Project blocks that still have steps ahead of the play head. The block
  // currently presenting is included — its farewell has not been said yet — but
  // it cannot be reordered, only skipped or extended.
  const upcomingBlocks = (() => {
    const seen = []
    for (const s of (status?.steps || []).slice((status?.step_idx ?? 0) + 1)) {
      if (s.block_robot_id && !seen.includes(s.block_robot_id)) seen.push(s.block_robot_id)
    }
    return seen
  })()

  // ── Chat ────────────────────────────────────────────────────────────────────
  const handleSend = async () => {
    const text = chatInput.trim()
    if (!text || !targetId || sending) return
    setChatInput('')
    setSending(true)
    addMsg({ role: 'user', text, time: ts() })
    const name = robots.find(r => r.client_id === targetId)?.robot_name || targetId
    setThinking({ robot: name, time: ts() })
    try {
      const res = await chatRobot(targetId, text)
      if (res.delegation_result) {
        // Delegation: hide Pepper's internal handoff message; show only the target's answer.
        const dr = res.delegation_result
        addMsg({ role: 'robot', robot: dr.robot_name, text: dr.clean_text, time: ts() })
        const drId = robots.find(r => r.robot_name === dr.robot_name)?.client_id || ''
        if (ttsRef.current) browserSpeak(dr.clean_text, drId)
      } else {
        const reply = (res.clean_text || res.response || '').trim()
        if (reply) {
          addMsg({ role: 'robot', robot: name, text: reply, time: ts() })
          if (ttsRef.current) browserSpeak(reply, targetId)
        }
      }
    } catch (e) {
      addMsg({ role: 'system', text: `Error: ${e.message}`, time: ts() })
    } finally {
      setThinking(null)
      setSending(false)
      inputRef.current?.focus()
    }
  }

  // ── Derived state ───────────────────────────────────────────────────────────
  const inQa      = status?.state === 'qa_window'
  const isRunning = ['running', 'waiting_ack', 'qa_window', 'paused'].includes(status?.state)
  const isIdle    = !status || ['idle', 'completed', 'error'].includes(status.state)

  return (
    <div className="demox-root">

      {/* ═══════════════════ TOP BAR ═══════════════════════════════════════ */}
      <div className="demox-topbar">
        <div className="demox-state-row">
          <StatePill state={status?.state} />
          {status?.total > 0 && (
            <span className="demo-progress">
              Step {status.step_idx + 1}/{status.total}
              {status.robot_id && <span className="demo-robot"> · {status.robot_id}</span>}
            </span>
          )}
          {isRunning && <RunClock status={status} />}
        </div>

        {/* Compact controls */}
        <div className="demox-controls">
          <button className="btn btn-primary btn-sm" disabled={ctrlLoad || isRunning}  onClick={() => setShowSelector(true)}>Start</button>
          <button className="btn btn-danger  btn-sm" disabled={ctrlLoad || isIdle}     onClick={() => run(stopDemo)}>Stop</button>
          {status?.state === 'paused'
            ? <button className="btn btn-sm" disabled={ctrlLoad}              onClick={() => run(resumeDemo)}>Resume</button>
            : <button className="btn btn-sm" disabled={ctrlLoad || isIdle}    onClick={() => run(pauseDemo)}>Pause</button>
          }
          <button className="btn btn-sm"     disabled={ctrlLoad || isIdle}    onClick={() => run(nextDemoStep)} title="Skip current step">Skip</button>
          {inQa
            ? <button className="btn btn-sm btn-danger" disabled={ctrlLoad}   onClick={() => run(endQaMode)}>End Q&amp;A</button>
            : <button className="btn btn-sm btn-qa"     disabled={ctrlLoad || isIdle} onClick={() => run(r => startQaMode('', r))}>Q&amp;A</button>
          }
          {/* Optional label for the next override. One click, then it clears. */}
          <select
            className="form-select"
            value={reason}
            disabled={ctrlLoad || isIdle}
            onChange={e => setReason(e.target.value)}
            title="Why are you overriding? Optional — recorded as a training label."
            style={{ width: 'auto', maxWidth: 170, fontSize: '0.75rem', padding: '4px 6px' }}
          >
            <option value="">why? (optional)</option>
            {OVERRIDE_REASONS.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
          {/* TTS toggle */}
          <button
            className={`btn btn-sm ${ttsEnabled ? 'btn-tts-on' : ''}`}
            title={ttsEnabled ? 'Mute browser TTS' : 'Unmute browser TTS'}
            onClick={() => {
              if (ttsEnabled) window.speechSynthesis?.cancel()
              setTtsEnabled(v => !v)
            }}
          >
            {ttsEnabled ? '🔊' : '🔇'}
          </button>
        </div>
      </div>

      {/* ═══════════════════ MOVE-ON BANNER (qa_window only) ══════════════ */}
      {inQa && (
        <div className="demox-moveon-banner">
          <div className="demox-moveon-left">
            <span className="demox-moveon-label">⏸ Pepper is waiting for visitors</span>
            {status?.text && (
              <span className="demox-moveon-quote">
                &ldquo;{status.text.replace(/\[.*?\]/g, '').trim()}&rdquo;
              </span>
            )}
          </div>
          <button
            className="btn demox-moveon-btn"
            disabled={ctrlLoad}
            onClick={() => run(nextDemoStep)}
          >
            Move On →
          </button>
        </div>
      )}

      {/* ═══════════════════ BODY: TIMELINE + CHAT ════════════════════════ */}
      <div className="demox-body">

        {/* ── Left: Step timeline ──────────────────────────────────────── */}
        <div className="demox-timeline-pane">
          <div className="demo-section-title">Script</div>

          {/* Mid-demo revision. The pre-start modal can reorder robots; this is
              the equivalent once the tour is under way, for when visitors say
              they are short on time or want more of one project. */}
          {isRunning && upcomingBlocks.length > 0 && (
            <div className="demox-revise">
              <div className="muted" style={{ fontSize: '0.72rem', marginBottom: 5 }}>
                Remaining projects — revise
              </div>
              {upcomingBlocks.map((rid, i) => (
                <div key={rid} className="demox-revise-row">
                  <span className="demox-revise-name">
                    {robots.find(r => r.client_id === rid)?.robot_name || rid}
                  </span>
                  <div style={{ display: 'flex', gap: 3 }}>
                    <button className="btn btn-sm" disabled={ctrlLoad}
                            onClick={() => revise('extend_qa', rid)}
                            title="More Q&A time for this project">+Q&amp;A</button>
                    <button className="btn btn-sm" disabled={ctrlLoad}
                            onClick={() => revise('compress', rid)}
                            title="Keep the research talk, drop the intro and greeting">Trim</button>
                    <button className="btn btn-sm" disabled={ctrlLoad || i === 0}
                            onClick={() => revise('reorder', rid)}
                            title="Move this project next">↑</button>
                    <button className="btn btn-sm btn-danger" disabled={ctrlLoad}
                            onClick={() => revise('skip', rid)}
                            title="Drop this project entirely">Skip</button>
                  </div>
                </div>
              ))}
              <button
                className="btn btn-sm btn-danger"
                disabled={ctrlLoad}
                onClick={() => revise('drop_remaining')}
                title="Cut to the wrap-up — the closing steps still run"
                style={{ marginTop: 5, width: '100%' }}
              >
                Cut to wrap-up
              </button>
            </div>
          )}
          <div className="demox-timeline">
            {(status?.steps || []).map((step, i) => {
              const isCurrent   = i === status.step_idx
              const isCompleted = i < status.step_idx
              return (
                <div
                  key={step.step_id}
                  ref={isCurrent ? currentRef : null}
                  className={[
                    'demox-step',
                    isCurrent   ? 'current'   : '',
                    isCompleted ? 'completed' : '',
                    step.qa_window ? 'qa'    : '',
                  ].filter(Boolean).join(' ')}
                >
                  <span className="demox-step-num">{i + 1}</span>
                  <div className="demox-step-info">
                    <span className="demox-step-id">{step.step_id}</span>
                    <span className="demox-step-robot muted">{step.robot_id}</span>
                  </div>
                  {step.qa_window && <span className="timeline-qa-badge">Q&amp;A</span>}
                  {isCompleted && <span className="timeline-check">✓</span>}
                  {isCurrent   && <span className="timeline-arrow">▶</span>}
                </div>
              )
            })}
            {!status?.steps?.length && (
              <div className="muted" style={{ fontSize: '0.78rem', padding: 8 }}>Start the demo to see the script.</div>
            )}
          </div>
        </div>

        {/* ── Right: Chat ──────────────────────────────────────────────── */}
        <div className="demox-chat-pane">

          {/* Robot selector */}
          <div className="demox-chat-controls">
            <label className="chat-label">Talk to:</label>
            <select
              className="form-select"
              value={targetId}
              onChange={e => setTargetId(e.target.value)}
              style={{ flex: 1 }}
            >
              {robots.length === 0 && <option value="">No robots connected</option>}
              {robots.map(r => (
                <option key={r.client_id} value={r.client_id}>
                  {r.robot_name} ({r.client_id})
                </option>
              ))}
            </select>
          </div>

          {/* Feed */}
          <div className="demox-feed" ref={feedRef}>
            {messages.length === 0 && !thinking ? (
              <div className="chat-empty">
                Start the demo or send a message to see the conversation here.
              </div>
            ) : (
              messages.map(m => <ChatMessage key={m.id} msg={m} />)
            )}
            {thinking && (
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, padding: '8px 0', opacity: 0.6 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', fontFamily: 'var(--mono)', minWidth: 70 }}>
                  {thinking.robot}
                </div>
                <div style={{ fontSize: 12, color: 'var(--muted)', fontStyle: 'italic' }}>
                  thinking…
                </div>
                <div style={{ fontSize: 10, color: 'var(--muted)', marginLeft: 'auto' }}>
                  {thinking.time}
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="demox-input-row">
            <input
              ref={inputRef}
              className="form-input"
              placeholder={targetId
                ? `Message ${robots.find(r => r.client_id === targetId)?.robot_name || targetId}…`
                : 'Connect a robot first…'
              }
              value={chatInput}
              disabled={!targetId || sending}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
            />
            <button
              className="btn btn-primary btn-sm"
              disabled={!chatInput.trim() || !targetId || sending}
              onClick={handleSend}
              style={{ width: 'auto', padding: '8px 16px', whiteSpace: 'nowrap' }}
            >
              {sending ? '…' : 'Send'}
            </button>
          </div>

        </div>
      </div>

      {/* ═══════════════════ ROBOT SELECTOR MODAL ═════════════════════════ */}
      {showSelector && (
        <RobotSelectorModal
          onClose={() => setShowSelector(false)}
          onStart={(robotIds, timeBudgetSec) => {
            setShowSelector(false)
            run(() => startDemo(robotIds, timeBudgetSec))
          }}
        />
      )}
    </div>
  )
}
