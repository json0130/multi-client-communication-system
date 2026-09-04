import { useState, useEffect, useCallback } from 'react'
import { getFlowStatus, previewFlowPlan } from '../api'
import FlowChart from './FlowChart'

/**
 * The flow graph: the tour's structure (blocks, roles, measured duration) and
 * the planner that compresses it under a stated budget.
 *
 * Read-only preview. Nothing here calls revise_script() — this answers "what
 * WOULD the planner do?", the same question a visitor's stated time pressure
 * asks live, without needing a running demo or a spoken phrase to test it.
 */

function fmt(sec) {
  if (sec == null) return '—'
  const s = Math.round(sec)
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`
}

const OP_LABEL = {
  set_qa_budget: 'set Q&A budget',
  compress: 'compress',
  skip: 'skip',
  drop_remaining: 'drop everything remaining',
  reorder: 'reorder',
  extend_qa: 'extend Q&A',
}

export default function FlowTab() {
  const [flow, setFlow] = useState(null)
  const [error, setError] = useState('')
  const [budgetMin, setBudgetMin] = useState('15')
  const [interest, setInterest] = useState('')
  const [busy, setBusy] = useState(false)
  const [plan, setPlan] = useState(null)

  const load = useCallback(async () => {
    try {
      const data = await getFlowStatus()
      setFlow(data)
      setError('')
    } catch (e) { setError(e.message) }
  }, [])

  // Poll rather than fetch once — the whole point of the "NOW" highlight is
  // that it tracks a demo that may be running, not a snapshot from whenever
  // this tab happened to load.
  useEffect(() => {
    load()
    const id = setInterval(load, 2000)
    return () => clearInterval(id)
  }, [load])

  const runPreview = async () => {
    const sec = Number(budgetMin) * 60
    if (!(sec > 0)) { setError('Enter a budget in minutes.'); return }
    setBusy(true); setError(''); setPlan(null)
    try {
      setPlan(await previewFlowPlan(sec, interest))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="kg-root">
      <div className="kg-header">
        <div>
          <div className="demo-section-title">Flow Graph &amp; Planner Preview</div>
          <div className="muted" style={{ fontSize: '0.76rem', marginTop: 2 }}>
            The remaining tour's structure, and what the planner would cut to
            fit a stated budget. Preview only — nothing here changes a running demo.
          </div>
        </div>
        <button className="btn btn-sm" disabled={busy} onClick={load}>Refresh</button>
      </div>

      {error && <div className="kg-error">{error}</div>}

      {flow && (
        <div className="kg-stats">
          <Stat label="blocks" value={flow.blocks.length} />
          <Stat label="fixed (open+close)" value={fmt(flow.fixed_sec)} />
          <Stat label="estimate, no cuts" value={fmt(flow.estimate_no_cuts?.total_sec)}
                hint="Q&A costed at the default 90s/window — nothing has been cut" />
          <Stat label="measured coverage"
                value={`${Math.round((flow.measured_coverage || 0) * 100)}%`}
                hint="Fraction of steps with a real timing behind them, versus the default guess" />
          {flow.current_state && flow.current_state !== 'idle' && (
            <Stat label="demo state" value={flow.current_state.replace(/_/g, ' ')} />
          )}
        </div>
      )}

      {/* ── Flowchart: the whole tour, "NOW" tracking whatever is running ── */}
      {flow && flow.full_steps?.length > 0 && (
        <div className="kg-matrix-wrap" style={{ marginBottom: 4 }}>
          <FlowChart
            fullSteps={flow.full_steps}
            robotLabel={rid => flow.blocks.find(b => b.robot_id === rid)?.robot_id || rid}
          />
        </div>
      )}

      {flow && flow.blocks.length === 0 && (
        <div className="kg-empty">
          <strong>No project blocks in the remaining script.</strong> Load or
          start a demo first — see the Demo tab.
        </div>
      )}

      {flow && flow.blocks.length > 0 && (
        <div className="kg-body">
          {/* ── Per-block numbers behind the chart above ─────────────────── */}
          <div className="kg-matrix-wrap" style={{ flex: 1 }}>
            <div className="muted" style={{ fontSize: '0.7rem', marginBottom: 4 }}>
              Remaining blocks only — what a cut could still touch
            </div>
            <table className="kg-matrix" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th className="kg-corner" style={{ textAlign: 'left' }}>Block</th>
                  <th className="kg-robot-head">scripted</th>
                  <th className="kg-robot-head">compress saves</th>
                  <th className="kg-robot-head">Q&amp;A windows</th>
                  <th className="kg-robot-head">steps</th>
                </tr>
              </thead>
              <tbody>
                {flow.blocks.map(b => (
                  <tr key={b.robot_id}>
                    <td className="kg-topic-head" style={{ textAlign: 'left' }}>{b.robot_id}</td>
                    <td className="kg-cell" style={{ cursor: 'default' }}>{fmt(b.scripted_sec)}</td>
                    <td className="kg-cell" style={{ cursor: 'default' }}>{fmt(b.compression_saving_sec)}</td>
                    <td className="kg-cell" style={{ cursor: 'default' }}>{b.qa_windows}</td>
                    <td className="kg-cell" style={{ cursor: 'default', fontSize: '0.68rem' }}>
                      {b.steps.map(s => (
                        <span key={s.step_id}
                              title={`${s.role}${s.compressible ? ' (compressible)' : ''}`}
                              style={{
                                display: 'inline-block', width: 8, height: 8, margin: '0 1px',
                                borderRadius: 2,
                                background: s.role === 'project' || s.role === 'qa'
                                  ? '#34d399' : (s.compressible ? 'var(--muted)' : '#3b82f6'),
                              }} />
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ── Planner preview ──────────────────────────────────────────── */}
          <div className="kg-detail" style={{ width: 340 }}>
            <div className="demo-section-title" style={{ marginBottom: 6 }}>
              Preview: fit a budget
            </div>
            <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
              <input className="form-input" type="number" min="1" placeholder="minutes"
                     value={budgetMin} onChange={e => setBudgetMin(e.target.value)}
                     style={{ width: 80 }} />
              <span className="muted" style={{ alignSelf: 'center', fontSize: '0.76rem' }}>minutes</span>
            </div>
            <input className="form-input" placeholder="visitor interest (optional), e.g. emotion recognition"
                   value={interest} onChange={e => setInterest(e.target.value)}
                   style={{ marginBottom: 8, fontSize: '0.78rem' }} />
            <button className="btn btn-sm btn-primary" disabled={busy} onClick={runPreview}
                    style={{ width: '100%', marginBottom: 10 }}>
              {busy ? '…' : 'Preview plan'}
            </button>

            {plan && (
              <>
                <table className="kg-kv" style={{ marginBottom: 10 }}>
                  <tbody>
                    <Row k="feasible" v={plan.feasible ? 'yes' : 'NO'} />
                    <Row k="already fit" v={plan.fits_already ? 'yes' : 'no'} />
                    <Row k="estimate" v={fmt(plan.estimate?.total_sec)} />
                    <Row k="measured coverage" v={`${Math.round((plan.measured_coverage || 0) * 100)}%`} />
                    {plan.interest_resolved && (
                      <Row k="interest resolved to" v={plan.resolved_topic} />
                    )}
                    {interest && !plan.interest_resolved && (
                      <Row k="interest" v="not resolved — ignored" />
                    )}
                  </tbody>
                </table>

                <div className="demo-section-title" style={{ margin: '8px 0 5px' }}>
                  What it would do
                </div>
                {plan.ops.length === 0 ? (
                  <div className="muted" style={{ fontSize: '0.78rem' }}>Nothing — it already fits.</div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginBottom: 8 }}>
                    {plan.ops.map((op, i) => (
                      <div key={i} className="kg-neighbour">
                        <span style={{ flex: 1 }}>
                          {OP_LABEL[op.kind] || op.kind}
                          {op.robot_id ? ` — ${op.robot_id}` : ''}
                          {op.seconds != null ? ` (${fmt(op.seconds)})` : ''}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                <div className="demo-section-title" style={{ margin: '8px 0 5px' }}>
                  Reasoning
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--muted)', lineHeight: 1.6 }}>
                  {plan.trace.map((t, i) => <div key={i}>· {t}</div>)}
                </div>

                {Object.keys(plan.importance || {}).length > 0 && (
                  <>
                    <div className="demo-section-title" style={{ margin: '8px 0 5px' }}>
                      Block importance used
                    </div>
                    {Object.entries(plan.importance).map(([rid, v]) => (
                      <div key={rid} className="kg-neighbour">
                        <span style={{ flex: 1 }}>{rid}</span>
                        <span className="muted">{v.toFixed(2)}</span>
                      </div>
                    ))}
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, hint }) {
  return (
    <div className="kg-stat" title={hint || ''}>
      <div className="kg-stat-value">{value}</div>
      <div className="kg-stat-label">{label}</div>
    </div>
  )
}

function Row({ k, v }) {
  return (
    <tr>
      <td className="muted">{k}</td>
      <td style={{ textAlign: 'right', fontFamily: 'var(--mono)' }}>{v}</td>
    </tr>
  )
}
