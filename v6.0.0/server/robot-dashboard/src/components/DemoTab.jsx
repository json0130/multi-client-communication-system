import { useState, useEffect, useRef, useCallback } from 'react'
import {
  getDemoStatus,
  startDemo, stopDemo, pauseDemo, resumeDemo, nextDemoStep,
  startQaMode, endQaMode,
  getRobots, chatRobot,
} from '../api'

// ── Browser TTS ───────────────────────────────────────────────────────────────

// Per-robot voice characteristics (pitch / rate). Browser picks best voice.
const ROBOT_VOICE = {
  pepper_01:  { pitch: 1.15, rate: 0.92 },   // warm guide voice
  chatbox_01: { pitch: 1.25, rate: 1.05 },   // energetic, slightly faster
  navel_01:   { pitch: 1.05, rate: 0.88 },   // slower, empathetic
  silbot_01:  { pitch: 0.80, rate: 0.93 },   // deeper, measured
}

function browserSpeak(text, robotId, volume = 1) {
  if (!window.speechSynthesis || !text) return
  window.speechSynthesis.cancel()          // stop any current speech first

  const utt  = new SpeechSynthesisUtterance(text)
  const cfg  = ROBOT_VOICE[robotId] || { pitch: 1, rate: 1 }
  utt.pitch  = cfg.pitch
  utt.rate   = cfg.rate
  utt.volume = volume

  // Try to grab a good English voice; fall back to default
  const voices = window.speechSynthesis.getVoices()
  if (voices.length) {
    const en = voices.find(v => v.lang.startsWith('en') && v.localService)
            || voices.find(v => v.lang.startsWith('en'))
            || voices[0]
    if (en) utt.voice = en
  }

  window.speechSynthesis.speak(utt)
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

            {/* ── Actions ── */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button className="btn btn-sm" onClick={onClose}>Cancel</button>
              <button
                className="btn btn-primary btn-sm"
                disabled={selectedIds.length === 0}
                onClick={() => onStart(selectedIds)}
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
      addMsg({ role: 'robot', robot: status.robot_id || 'Robot', text: clean, time: ts(), demo: true })
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
  const run = async fn => {
    setCtrlLoad(true)
    try { setStatus(await fn()) } catch {}
    finally { setCtrlLoad(false) }
  }

  // ── Chat ────────────────────────────────────────────────────────────────────
  const handleSend = async () => {
    const text = chatInput.trim()
    if (!text || !targetId || sending) return
    setChatInput('')
    setSending(true)
    addMsg({ role: 'user', text, time: ts() })
    try {
      const res = await chatRobot(targetId, text)
      const name = robots.find(r => r.client_id === targetId)?.robot_name || targetId
      const reply = (res.clean_text || res.response || '').trim()
      if (reply) {
        addMsg({ role: 'robot', robot: name, text: reply, time: ts() })
        if (ttsRef.current) browserSpeak(reply, targetId)
      }
    } catch (e) {
      addMsg({ role: 'system', text: `Error: ${e.message}`, time: ts() })
    } finally {
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
            : <button className="btn btn-sm btn-qa"     disabled={ctrlLoad || isIdle} onClick={() => run(() => startQaMode(''))}>Q&amp;A</button>
          }
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
            {messages.length === 0 ? (
              <div className="chat-empty">
                Start the demo or send a message to see the conversation here.
              </div>
            ) : (
              messages.map(m => <ChatMessage key={m.id} msg={m} />)
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
          onStart={robotIds => {
            setShowSelector(false)
            run(() => startDemo(robotIds))
          }}
        />
      )}
    </div>
  )
}
