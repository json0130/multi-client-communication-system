import { useState, useRef, useEffect } from 'react'
import { connectRobot, disconnectRobot, assignPersona } from '../api'
import { useToast } from './Toast'

const ALL_MODULES = ['gpt', 'speech', 'emotion', 'rag', 'navigation']

function RobotPlaceholder() {
  return (
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"
      style={{ width: '100%', height: '100%' }}>
      <rect width="64" height="64" rx="8" fill="var(--surface2)" />
      <rect x="18" y="28" width="28" height="22" rx="4" stroke="var(--border)" strokeWidth="2" fill="none" />
      <rect x="20" y="14" width="24" height="16" rx="3" stroke="var(--border)" strokeWidth="2" fill="none" />
      <line x1="32" y1="14" x2="32" y2="10" stroke="var(--border)" strokeWidth="2" strokeLinecap="round" />
      <circle cx="32" cy="9" r="1.5" fill="var(--border)" />
      <circle cx="27" cy="21" r="2" fill="var(--muted)" opacity="0.5" />
      <circle cx="37" cy="21" r="2" fill="var(--muted)" opacity="0.5" />
      <path d="M27 27h10" stroke="var(--border)" strokeWidth="1.5" strokeLinecap="round" />
      <rect x="10" y="31" width="8" height="5" rx="2.5" stroke="var(--border)" strokeWidth="1.5" fill="none" />
      <rect x="46" y="31" width="8" height="5" rx="2.5" stroke="var(--border)" strokeWidth="1.5" fill="none" />
      <rect x="21" y="50" width="8" height="7" rx="2" stroke="var(--border)" strokeWidth="1.5" fill="none" />
      <rect x="35" y="50" width="8" height="7" rx="2" stroke="var(--border)" strokeWidth="1.5" fill="none" />
    </svg>
  )
}

export default function RobotCard({ robot, personas, onRefresh }) {
  const [busy, setBusy]                       = useState(false)
  const [selectedPersona, setSelectedPersona] = useState('')
  const [expanded, setExpanded]               = useState(false)
  const [photo, setPhoto]                     = useState(null)
  const fileRef                               = useRef(null)
  const toast = useToast()

  const isOnline       = robot.ws_connected
  const currentPersona = personas.find(p => p.id === robot.persona_id)
  const modules        = robot.modules || []
  const photoKey       = `robot_photo_${robot.client_id}`

  useEffect(() => {
    const saved = localStorage.getItem(photoKey)
    if (saved) setPhoto(saved)
  }, [photoKey])

  function handlePhotoClick() { fileRef.current?.click() }

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => {
      const dataUrl = ev.target.result
      setPhoto(dataUrl)
      localStorage.setItem(photoKey, dataUrl)
    }
    reader.readAsDataURL(file)
    e.target.value = ''
  }

  async function handleConnect() {
    setBusy(true)
    try {
      await connectRobot(robot.client_id)
      toast.success(`Connecting to ${robot.robot_name}...`)
      setTimeout(onRefresh, 1500)
    } catch (e) { toast.error(e.message) }
    finally { setBusy(false) }
  }

  async function handleDisconnect() {
    setBusy(true)
    try {
      await disconnectRobot(robot.client_id)
      toast.info(`${robot.robot_name} disconnected`)
      setTimeout(onRefresh, 800)
    } catch (e) { toast.error(e.message) }
    finally { setBusy(false) }
  }

  async function handleAssign() {
    if (!selectedPersona) return
    setBusy(true)
    try {
      const res = await assignPersona(robot.client_id, selectedPersona)
      toast.success(res.message)
      setSelectedPersona('')
      onRefresh()
    } catch (e) { toast.error(e.message) }
    finally { setBusy(false) }
  }

  return (
    <div className={`robot-card ${isOnline ? 'online' : ''}`}>
      {/* Header row: info + photo */}
      <div className="card-header">
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="card-name">{robot.robot_name}</div>
          <div className="card-id">{robot.client_id}</div>
          {robot.ip_address && (
            <div className="card-id" style={{ marginTop: 2 }}>
              {robot.ip_address}{robot.ws_port ? `:${robot.ws_port}` : ''}
            </div>
          )}
          <div className={`status-badge ${isOnline ? 'online' : 'offline'}`} style={{ marginTop: 6, display: 'inline-flex' }}>
            <span className="status-dot" />
            {isOnline ? 'Online' : 'Offline'}
          </div>
        </div>

        {/* Photo thumbnail */}
        <div className="robot-photo" onClick={handlePhotoClick} title="Click to upload photo">
          {photo
            ? <img src={photo} alt={robot.robot_name}
                style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 6 }} />
            : <RobotPlaceholder />
          }
          <div className="robot-photo-overlay"><span>📷</span></div>
        </div>
        <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
      </div>

      {/* Persona */}
      <div className="card-section">
        <div className="card-label">Persona</div>
        <div className={`persona-pill ${!currentPersona ? 'none' : ''}`}>
          {currentPersona
            ? <><span style={{ opacity: 0.6 }}>◈</span>{currentPersona.name}</>
            : <><span style={{ opacity: 0.4 }}>◻</span>No persona assigned</>
          }
        </div>
        {currentPersona?.robot_role && (
          <>
            <div style={{
              marginTop: 6, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--muted)',
              lineHeight: 1.5, overflow: 'hidden', display: '-webkit-box',
              WebkitLineClamp: expanded ? 'unset' : 2, WebkitBoxOrient: 'vertical', cursor: 'pointer',
            }} onClick={() => setExpanded(e => !e)}>
              {currentPersona.robot_role}
            </div>
            <button onClick={() => setExpanded(e => !e)} style={{
              background: 'none', border: 'none', color: 'var(--muted)',
              fontSize: 10, cursor: 'pointer', padding: '2px 0', fontFamily: 'var(--mono)',
            }}>
              {expanded ? '▲ less' : '▼ more'}
            </button>
          </>
        )}
      </div>

      {/* Modules */}
      <div className="card-section">
        <div className="card-label">Modules</div>
        <div className="module-list">
          {ALL_MODULES.map(m => (
            <span key={m} className={`module-badge ${modules.includes(m) ? 'active' : ''}`}>{m}</span>
          ))}
        </div>
      </div>

      {/* Assign persona */}
      <div className="card-section">
        <div className="card-label">Assign persona</div>
        <div className="assign-row">
          <select
            className="form-select"
            value={selectedPersona}
            onChange={e => setSelectedPersona(e.target.value)}
            disabled={busy}
          >
            <option value="">Select...</option>
            {personas.map(p => (
              <option key={p.id} value={p.id}>
                {p.name}{p.is_default ? ' ★' : ''}
              </option>
            ))}
          </select>
          <button
            className="btn btn-primary assign-btn"
            onClick={handleAssign}
            disabled={busy || !selectedPersona}
          >
            Assign
          </button>
        </div>
        {!isOnline && selectedPersona && (
          <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4, fontFamily: 'var(--mono)' }}>
            Offline — will apply on next connect
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="card-actions">
        {isOnline ? (
          <button className="btn btn-disconnect" onClick={handleDisconnect} disabled={busy}>
            {busy ? 'Working...' : 'Disconnect'}
          </button>
        ) : (
          <button className="btn btn-connect" onClick={handleConnect} disabled={busy}>
            {busy ? 'Connecting...' : 'Connect'}
          </button>
        )}
      </div>
    </div>
  )
}
