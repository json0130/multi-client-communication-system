import { useState, useEffect, useRef } from 'react'
import {
  getDemoStatus,
  startDemo, stopDemo, pauseDemo, resumeDemo, nextDemoStep,
  startQaMode, endQaMode,
} from '../api'

// State pill colour mapping
const STATE_COLORS = {
  idle:        { bg: '#374151', text: '#d1d5db' },
  running:     { bg: '#1d4ed8', text: '#eff6ff' },
  waiting_ack: { bg: '#1d4ed8', text: '#eff6ff' },
  qa_window:   { bg: '#6d28d9', text: '#ede9fe' },
  paused:      { bg: '#92400e', text: '#fef3c7' },
  completed:   { bg: '#065f46', text: '#d1fae5' },
  error:       { bg: '#991b1b', text: '#fee2e2' },
}

function StatePill({ state }) {
  const c = STATE_COLORS[state] || STATE_COLORS.idle
  return (
    <span
      className="demo-state-pill"
      style={{ background: c.bg, color: c.text }}
    >
      {state?.replace('_', ' ') || 'idle'}
    </span>
  )
}

export default function DemoTab() {
  const [status, setStatus]     = useState(null)
  const [qaMsg, setQaMsg]       = useState('')
  const [loading, setLoading]   = useState(false)
  const currentRef              = useRef(null)
  const intervalRef             = useRef(null)

  // Poll demo status every 2 seconds
  useEffect(() => {
    const poll = async () => {
      try {
        const data = await getDemoStatus()
        setStatus(data)
      } catch {
        // server unreachable — keep last status
      }
    }
    poll()
    intervalRef.current = setInterval(poll, 2000)
    return () => clearInterval(intervalRef.current)
  }, [])

  // Auto-scroll current step into view
  useEffect(() => {
    if (currentRef.current) {
      currentRef.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [status?.step_idx])

  const run = async (fn) => {
    setLoading(true)
    try {
      const data = await fn()
      setStatus(data)
    } catch {
      // ignore — next poll will catch up
    } finally {
      setLoading(false)
    }
  }

  const handleQa = () => {
    if (status?.state === 'qa_window') {
      run(endQaMode)
    } else {
      run(() => startQaMode(qaMsg))
      setQaMsg('')
    }
  }

  const inQa      = status?.state === 'qa_window'
  const isRunning = ['running', 'waiting_ack', 'qa_window', 'paused'].includes(status?.state)
  const isIdle    = !status || status.state === 'idle' || status.state === 'completed' || status.state === 'error'

  return (
    <div className="demo-tab">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="demo-header">
        <div className="demo-status-row">
          <StatePill state={status?.state} />
          {status && status.total > 0 && (
            <span className="demo-progress">
              Step {status.step_idx + 1} / {status.total}
              {status.robot_id && (
                <span className="demo-robot"> — {status.robot_id}</span>
              )}
            </span>
          )}
        </div>

        {status?.text && (
          <p className="demo-step-text">&ldquo;{status.text}&rdquo;</p>
        )}
      </div>

      {/* ── Controls ───────────────────────────────────────────────────── */}
      <div className="demo-controls">
        <button
          className="btn btn-primary"
          disabled={loading || isRunning}
          onClick={() => run(startDemo)}
        >
          Start
        </button>
        <button
          className="btn btn-danger"
          disabled={loading || isIdle}
          onClick={() => run(stopDemo)}
        >
          Stop
        </button>
        {status?.state === 'paused' ? (
          <button
            className="btn"
            disabled={loading}
            onClick={() => run(resumeDemo)}
          >
            Resume
          </button>
        ) : (
          <button
            className="btn"
            disabled={loading || isIdle}
            onClick={() => run(pauseDemo)}
          >
            Pause
          </button>
        )}
        <button
          className="btn"
          disabled={loading || isIdle}
          onClick={() => run(nextDemoStep)}
          title="Force-advance past current step (use to recover from timeout)"
        >
          Next Step
        </button>
        <button
          className={`btn ${inQa ? 'btn-danger' : 'btn-qa'}`}
          disabled={loading || isIdle}
          onClick={handleQa}
        >
          {inQa ? 'End Q&A' : 'Q&A Mode'}
        </button>
      </div>

      {/* Q&A message input (shown when not already in Q&A) */}
      {!inQa && isRunning && (
        <div className="qa-row">
          <input
            className="form-input"
            placeholder="Optional: Pepper says... (leave blank for silent Q&A)"
            value={qaMsg}
            onChange={e => setQaMsg(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleQa()}
          />
        </div>
      )}

      {/* ── Step timeline ──────────────────────────────────────────────── */}
      {status?.steps?.length > 0 && (
        <div className="demo-timeline-wrapper">
          <h3 className="demo-section-title">Script Timeline</h3>
          <div className="demo-timeline">
            {status.steps.map((step, i) => {
              const isCurrent   = i === status.step_idx
              const isCompleted = i < status.step_idx
              const isQaStep    = step.qa_window
              return (
                <div
                  key={step.step_id}
                  ref={isCurrent ? currentRef : null}
                  className={[
                    'timeline-step',
                    isCurrent   ? 'current'   : '',
                    isCompleted ? 'completed' : '',
                    isQaStep    ? 'qa'        : '',
                  ].filter(Boolean).join(' ')}
                >
                  <span className="timeline-idx">{i + 1}</span>
                  <span className="timeline-id">{step.step_id}</span>
                  <span className="timeline-robot muted">{step.robot_id}</span>
                  <span className="timeline-text muted">{step.text}</span>
                  {isQaStep && <span className="timeline-qa-badge">Q&amp;A</span>}
                  {isCompleted && <span className="timeline-check">✓</span>}
                  {isCurrent   && <span className="timeline-arrow">▶</span>}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
