import { deletePersona } from '../api'
import { useToast } from './Toast'
import { OceanChart } from './OceanRadar'

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

  const p        = persona.personality || {}
  const caps     = persona.capabilities || {}
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
              <span key={c} className="module-badge active">{c.replace(/_/g, ' ')}</span>
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

      {/* OCEAN radar chart */}
      {Object.keys(p).length > 0 && (
        <div className="card-section">
          <div className="card-label">Personality (OCEAN)</div>
          <div style={{ display: 'flex', justifyContent: 'center', marginTop: 8 }}>
            <OceanChart values={p} size={130} />
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
