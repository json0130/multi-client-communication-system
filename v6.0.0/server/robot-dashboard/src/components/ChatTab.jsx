import { useState, useEffect, useRef } from 'react'
import { getRobots, chatRobot, getDemoStatus, startQaMode, endQaMode } from '../api'

// ── helpers ───────────────────────────────────────────────────────────────────

function ts() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// ── sub-components ────────────────────────────────────────────────────────────

function DemoBanner({ demoStatus }) {
  if (!demoStatus || demoStatus.state === 'idle') return null
  const stateLabel = demoStatus.state?.replace('_', ' ').toUpperCase()
  const colors = {
    running:     '#1d4ed8', waiting_ack: '#1d4ed8',
    qa_window:   '#6d28d9', paused: '#92400e',
    completed:   '#065f46', error: '#991b1b',
  }
  const bg = colors[demoStatus.state] || '#374151'
  return (
    <div className="chat-demo-banner" style={{ background: bg + '33', borderColor: bg }}>
      <span className="chat-demo-label" style={{ color: bg === '#374151' ? '#9ca3af' : undefined }}>
        Demo {stateLabel}
      </span>
      {demoStatus.robot_id && (
        <span className="chat-demo-robot">{demoStatus.robot_id}</span>
      )}
      {demoStatus.text && (
        <span className="chat-demo-text">&ldquo;{demoStatus.text.slice(0, 80)}{demoStatus.text.length > 80 ? '…' : ''}&rdquo;</span>
      )}
      {demoStatus.total > 0 && (
        <span className="chat-demo-progress muted">
          {demoStatus.step_idx + 1}/{demoStatus.total}
        </span>
      )}
    </div>
  )
}

function ChatMessage({ msg }) {
  const isUser   = msg.role === 'user'
  const isSystem = msg.role === 'system'
  return (
    <div className={`chat-msg ${isUser ? 'chat-msg-user' : ''} ${isSystem ? 'chat-msg-system' : ''}`}>
      {!isUser && (
        <div className="chat-msg-header">
          <span className="chat-msg-name">{isSystem ? '— System —' : msg.robot}</span>
          <span className="chat-msg-time muted">{msg.time}</span>
        </div>
      )}
      <div className="chat-bubble">
        {msg.text}
      </div>
      {isUser && (
        <div className="chat-msg-header" style={{ justifyContent: 'flex-end' }}>
          <span className="chat-msg-time muted">{msg.time}</span>
          <span className="chat-msg-name">You</span>
        </div>
      )}
    </div>
  )
}

// ── main component ────────────────────────────────────────────────────────────

export default function ChatTab() {
  const [robots,      setRobots]      = useState([])
  const [targetId,    setTargetId]    = useState('')
  const [messages,    setMessages]    = useState([])
  const [input,       setInput]       = useState('')
  const [sending,     setSending]     = useState(false)
  const [demoStatus,  setDemoStatus]  = useState(null)
  const [qaMsg,       setQaMsg]       = useState('')
  const [showQaInput, setShowQaInput] = useState(false)
  const feedRef   = useRef(null)
  const inputRef  = useRef(null)

  // Load connected robots
  useEffect(() => {
    getRobots().then(data => {
      const connected = (data.robots || []).filter(r => r.ws_connected)
      setRobots(connected)
      if (connected.length > 0 && !targetId) {
        setTargetId(connected[0].client_id)
      }
    }).catch(() => {})
  }, [])

  // Poll demo status every 2 s
  useEffect(() => {
    const poll = () => getDemoStatus().then(setDemoStatus).catch(() => {})
    poll()
    const id = setInterval(poll, 2000)
    return () => clearInterval(id)
  }, [])

  // Track demo steps as system messages in the feed
  const lastStepRef = useRef(null)
  useEffect(() => {
    if (!demoStatus) return
    const key = `${demoStatus.step_idx}:${demoStatus.state}`
    if (key === lastStepRef.current) return
    lastStepRef.current = key

    if (demoStatus.state === 'running' || demoStatus.state === 'waiting_ack') {
      if (demoStatus.text) {
        addMsg({
          role:  'robot',
          robot: demoStatus.robot_id || 'Robot',
          text:  demoStatus.text.replace(/\[.*?\]/g, '').trim(),
          time:  ts(),
          demo:  true,
        })
      }
    }
    if (demoStatus.state === 'qa_window') {
      addMsg({ role: 'system', text: 'Q&A window open — speak to any robot', time: ts() })
    }
    if (demoStatus.state === 'completed') {
      addMsg({ role: 'system', text: 'Demo completed', time: ts() })
    }
  }, [demoStatus])

  // Auto-scroll feed
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight
    }
  }, [messages])

  function addMsg(msg) {
    setMessages(prev => [...prev, { id: Date.now() + Math.random(), ...msg }])
  }

  async function handleSend() {
    const text = input.trim()
    if (!text || !targetId || sending) return
    setInput('')
    setSending(true)

    addMsg({ role: 'user', text, time: ts() })

    try {
      const res = await chatRobot(targetId, text)
      const robotName = robots.find(r => r.client_id === targetId)?.robot_name || targetId
      addMsg({
        role:  'robot',
        robot: robotName,
        text:  (res.clean_text || res.response || '').trim(),
        time:  ts(),
      })
    } catch (e) {
      addMsg({ role: 'system', text: `Error: ${e.message}`, time: ts() })
    } finally {
      setSending(false)
      inputRef.current?.focus()
    }
  }

  async function handleQaInterrupt() {
    try {
      if (demoStatus?.state === 'qa_window') {
        await endQaMode()
        addMsg({ role: 'system', text: 'Q&A window closed — demo resuming', time: ts() })
      } else {
        const msg = qaMsg.trim()
        await startQaMode(msg)
        addMsg({ role: 'system', text: msg ? `Q&A opened: "${msg}"` : 'Q&A window opened', time: ts() })
        setQaMsg('')
        setShowQaInput(false)
      }
    } catch (e) {
      addMsg({ role: 'system', text: `Q&A error: ${e.message}`, time: ts() })
    }
  }

  const inQa     = demoStatus?.state === 'qa_window'
  const demoLive = demoStatus && !['idle', 'completed'].includes(demoStatus.state)

  return (
    <div className="chat-tab">
      {/* ── Demo banner ─────────────────────────────────────── */}
      <DemoBanner demoStatus={demoStatus} />

      {/* ── Controls bar ────────────────────────────────────── */}
      <div className="chat-controls">
        {/* Robot selector */}
        <div className="chat-target-row">
          <label className="chat-label">Talking to:</label>
          <select
            className="form-select"
            value={targetId}
            onChange={e => setTargetId(e.target.value)}
            style={{ flex: 1 }}
          >
            {robots.length === 0 && (
              <option value="">No robots connected</option>
            )}
            {robots.map(r => (
              <option key={r.client_id} value={r.client_id}>
                {r.robot_name} ({r.client_id})
              </option>
            ))}
          </select>
        </div>

        {/* Q&A interrupt */}
        {demoLive && (
          <div className="chat-qa-row">
            {!inQa && showQaInput && (
              <input
                className="form-input"
                placeholder="Optional: message for Pepper to say..."
                value={qaMsg}
                onChange={e => setQaMsg(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleQaInterrupt()}
                style={{ flex: 1 }}
              />
            )}
            <button
              className={`btn btn-sm ${inQa ? 'btn-danger' : 'btn-qa'}`}
              onClick={() => {
                if (inQa) { handleQaInterrupt() }
                else if (showQaInput) { handleQaInterrupt() }
                else { setShowQaInput(true) }
              }}
            >
              {inQa ? '⏹ End Q&A' : '💬 Interrupt → Q&A'}
            </button>
            {!inQa && showQaInput && (
              <button className="btn btn-sm" onClick={() => setShowQaInput(false)}>
                Cancel
              </button>
            )}
          </div>
        )}
      </div>

      {/* ── Feed ────────────────────────────────────────────── */}
      <div className="chat-feed" ref={feedRef}>
        {messages.length === 0 ? (
          <div className="chat-empty">
            <p>Select a robot above and start chatting.</p>
            {demoLive && (
              <p style={{ marginTop: 6 }}>Demo is running — robot speech will appear here automatically.</p>
            )}
          </div>
        ) : (
          messages.map(m => <ChatMessage key={m.id} msg={m} />)
        )}
      </div>

      {/* ── Input ───────────────────────────────────────────── */}
      <div className="chat-input-row">
        <input
          ref={inputRef}
          className="form-input"
          placeholder={targetId ? `Message ${robots.find(r => r.client_id === targetId)?.robot_name || targetId}…` : 'Connect a robot first…'}
          value={input}
          disabled={!targetId || sending}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
        />
        <button
          className="btn btn-primary btn-sm"
          disabled={!input.trim() || !targetId || sending}
          onClick={handleSend}
          style={{ width: 'auto', padding: '8px 18px' }}
        >
          {sending ? '…' : 'Send'}
        </button>
      </div>
    </div>
  )
}
