import { useState, useEffect, useRef, useCallback } from 'react'
import {
  getDemoStatus,
  startDemo, stopDemo, pauseDemo, resumeDemo, nextDemoStep,
  startQaMode, endQaMode,
  getRobots, chatRobot,
} from '../api'

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
  const [status,   setStatus]   = useState(null)
  const [ctrlLoad, setCtrlLoad] = useState(false)

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
    const key = `${status.step_idx}:${status.state}`
    if (key === lastStepKey.current) return
    lastStepKey.current = key

    if ((status.state === 'running' || status.state === 'waiting_ack') && status.text) {
      const clean = status.text.replace(/\[.*?\]/g, '').trim()
      if (clean) addMsg({ role: 'robot', robot: status.robot_id || 'Robot', text: clean, time: ts(), demo: true })
    }
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
      if (reply) addMsg({ role: 'robot', robot: name, text: reply, time: ts() })
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
          <button className="btn btn-primary btn-sm" disabled={ctrlLoad || isRunning}  onClick={() => run(startDemo)}>Start</button>
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
    </div>
  )
}
