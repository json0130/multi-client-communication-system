import { deletePersona } from '../api'
import { useToast } from './Toast'

const OCEAN_LABELS = { O: 'Open', C: 'Cons', E: 'Extra', A: 'Agree', N: 'Neuro' }

export default function PersonaCard({ persona, onDeleted }) {
  const toast = useToast()

  async function handleDelete() {
    if (!confirm(`Delete persona "${persona.name}"?`)) return
    try {
      await deletePersona(persona.id)
      toast.success(`'${persona.name}' deleted`)
      onDeleted(persona.id)
    } catch (e) {
      toast.error(e.message)
    }
  }

  const p = persona.personality || {}
  const caps = persona.capabilities || {}
  const activeCaps = Object.entries(caps).filter(([, v]) => v).map(([k]) => k)

  return (
    <div className={`persona-card ${persona.is_default ? 'default-card' : ''}`}>
      {persona.is_default && <div className="default-tag">Default</div>}
      <div className="persona-name">{persona.name}</div>
      <div className="persona-desc">{persona.description || 'No description'}</div>

      {/* Modules */}
      <div className="card-section">
        <div className="card-label">Modules</div>
        <div className="module-list">
          {(persona.modules || []).map(m => (
            <span key={m} className="module-badge active">{m}</span>
          ))}
        </div>
      </div>

      {/* Capabilities */}
      {activeCaps.length > 0 && (
        <div className="card-section">
          <div className="card-label">Capabilities</div>
          <div className="module-list">
            {activeCaps.map(c => (
              <span key={c} className="module-badge active">
                {c.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Voice */}
      <div className="card-section">
        <div className="card-label">Voice</div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--muted)' }}>
          {persona.voice_config?.gender || 'female'} ·{' '}
          {persona.voice_config?.language || 'en'} ·{' '}
          {persona.voice_config?.rate || '+0%'}
        </div>
      </div>

      {/* OCEAN mini bars */}
      {Object.keys(p).length > 0 && (
        <div className="card-section">
          <div className="card-label">Personality (OCEAN)</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {Object.entries(OCEAN_LABELS).map(([key, label]) => (
              <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--muted)', width: 32 }}>
                  {label}
                </span>
                <div style={{ flex: 1, height: 4, background: 'var(--surface2)', borderRadius: 2 }}>
                  <div style={{
                    width: `${(p[key] || 0) * 100}%`,
                    height: '100%',
                    background: 'var(--accent)',
                    borderRadius: 2,
                    opacity: 0.7,
                  }} />
                </div>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--muted)', width: 24 }}>
                  {(p[key] || 0).toFixed(1)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
        {!persona.is_default && (
          <button className="btn btn-danger" onClick={handleDelete} style={{ flex: 1 }}>
            Delete
          </button>
        )}
      </div>
    </div>
  )
}