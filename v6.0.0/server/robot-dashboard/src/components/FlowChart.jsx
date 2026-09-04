/**
 * The tour as an actual flowchart: boxes in sequence, arrows between them,
 * grouped into lanes (opening / each project block / closing), with the step
 * that is CURRENTLY RUNNING highlighted.
 *
 * Steps flow left-to-right within a lane; lanes stack top-to-bottom, connected
 * by a vertical arrow from the last box of one lane to the first box of the
 * next. This mirrors build_script()'s own structure — a lane IS a block — so
 * the chart is a direct picture of what the tour actually is, not a redrawing
 * of it.
 */

const ROLE_LABEL = {
  opening: 'open', intro: 'intro', handoff: 'handoff', greeting: 'greet',
  prompt: 'prompt', project: 'TALK', qa: 'Q&A', transition: 'sign-off',
  closing: 'close',
}

const ROLE_ACCENT = {
  project: '#34d399', qa: '#a78bfa',
}

function StepBox({ step, isCurrent, isCompleted }) {
  const accent = ROLE_ACCENT[step.role]
  return (
    <div
      title={step.step_id}
      style={{
        position: 'relative',
        minWidth: 64, padding: '6px 9px', borderRadius: 7,
        fontSize: '0.68rem', textAlign: 'center', flexShrink: 0,
        background: isCurrent ? 'rgba(59,130,246,0.22)'
                  : isCompleted ? 'var(--surface2)' : 'var(--surface)',
        border: isCurrent ? '1.5px solid var(--accent)'
              : `1px solid ${accent && !isCompleted ? accent + '55' : 'var(--border)'}`,
        opacity: isCompleted && !isCurrent ? 0.55 : 1,
        boxShadow: isCurrent ? '0 0 0 3px rgba(59,130,246,0.18)' : 'none',
        animation: isCurrent ? 'flow-pulse 1.6s ease-in-out infinite' : 'none',
      }}
    >
      {isCurrent && (
        <div style={{
          position: 'absolute', top: -9, left: '50%', transform: 'translateX(-50%)',
          fontSize: '0.58rem', fontWeight: 700, color: 'var(--accent)',
          background: 'var(--bg)', padding: '0 4px', letterSpacing: '0.04em',
        }}>NOW</div>
      )}
      <div style={{
        fontSize: '0.6rem', fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: '0.03em',
        color: accent && !isCompleted ? accent : 'var(--muted)',
      }}>
        {ROLE_LABEL[step.role] || step.role || '?'}
      </div>
      {isCompleted && !isCurrent && (
        <div style={{ fontSize: '0.6rem', color: 'var(--muted)' }}>✓</div>
      )}
    </div>
  )
}

function Arrow({ dir = 'right' }) {
  return (
    <div style={{
      flexShrink: 0, color: 'var(--muted)', fontSize: '0.85rem',
      display: 'flex', alignItems: 'center', padding: dir === 'right' ? '0 2px' : '2px 0',
    }}>
      {dir === 'right' ? '→' : '↓'}
    </div>
  )
}

function Lane({ label, steps, currentId, completedIds }) {
  if (!steps.length) return null
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 3, padding: '6px 8px',
      background: 'var(--bg)', borderRadius: 8, overflowX: 'auto',
    }}>
      {label && (
        <div style={{
          flexShrink: 0, fontSize: '0.66rem', fontWeight: 700, color: 'var(--accent)',
          minWidth: 64, marginRight: 2,
        }}>
          {label}
        </div>
      )}
      {steps.map((s, i) => (
        <div key={s.step_id} style={{ display: 'flex', alignItems: 'center' }}>
          {i > 0 && <Arrow />}
          <StepBox step={s} isCurrent={s.step_id === currentId}
                   isCompleted={completedIds.has(s.step_id)} />
        </div>
      ))}
    </div>
  )
}

/**
 * `fullSteps`: [{step_id, robot_id, role, block_robot_id, qa_window,
 *                is_current, is_completed}] — the WHOLE tour in order, from
 * /flow/status's full_steps. Grouped into lanes here by consecutive
 * block_robot_id (opening/closing have none and form their own lanes).
 */
export default function FlowChart({ fullSteps, robotLabel }) {
  if (!fullSteps || fullSteps.length === 0) {
    return <div className="muted" style={{ fontSize: '0.8rem' }}>No script loaded.</div>
  }

  const lanes = []
  let cur = null
  for (const s of fullSteps) {
    const key = s.block_robot_id || (s.role === 'closing' ? '__closing' : '__opening')
    if (!cur || cur.key !== key) {
      cur = { key, block_robot_id: s.block_robot_id, steps: [] }
      lanes.push(cur)
    }
    cur.steps.push(s)
  }

  const currentId = fullSteps.find(s => s.is_current)?.step_id
  const completedIds = new Set(fullSteps.filter(s => s.is_completed).map(s => s.step_id))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <style>{`
        @keyframes flow-pulse {
          0%, 100% { box-shadow: 0 0 0 3px rgba(59,130,246,0.18); }
          50%      { box-shadow: 0 0 0 6px rgba(59,130,246,0.08); }
        }
      `}</style>
      {lanes.map((lane, i) => (
        <div key={lane.key}>
          <Lane
            label={lane.block_robot_id ? robotLabel(lane.block_robot_id)
                 : (lane.key === '__opening' ? 'Opening' : 'Closing')}
            steps={lane.steps}
            currentId={currentId}
            completedIds={completedIds}
          />
          {i < lanes.length - 1 && (
            <div style={{ display: 'flex', justifyContent: 'center', margin: '1px 0' }}>
              <Arrow dir="down" />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
