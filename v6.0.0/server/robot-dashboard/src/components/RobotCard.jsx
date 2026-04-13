import { useState } from 'react'
import { connectRobot, disconnectRobot, assignPersona } from '../api'
import { useToast } from './Toast'

const ALL_MODULES = ['gpt', 'speech', 'emotion', 'rag', 'navigation']

export default function RobotCard({ robot, personas, onRefresh }) {
  const [busy, setBusy] = useState(false)
  const [selectedPersona, setSelectedPersona] = useState('')
  const toast = useToast()

  const isOnline = robot.ws_connected

  async function handleConnect() {
    setBusy(true)
    try {
      await connectRobot(robot.client_id)
      toast.success(`Connecting to ${robot.robot_name}...`)
      setTimeout(onRefresh, 1500)
    } catch (e) {
      toast.error(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleDisconnect() {
    setBusy(true)
    try {
      await disconnectRobot(robot.client_id)
      toast.info(`${robot.robot_name} disconnected`)
      setTimeout(onRefresh, 800)
    } catch (e) {
      toast.error(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleAssign() {
    if (!selectedPersona) return
    setBusy(true)
    try {
      const res = await assignPersona(robot.client_id, selectedPersona)
      toast.success(res.message)
      setSelectedPersona('')
      onRefresh()
    } catch (e) {
      toast.error(e.message)
    } finally {
      setBusy(false)
    }
  }

  const currentPersona = personas.find(p => p.id === robot.persona_id)
  const modules = robot.modules || []

  return (
    <div className={`robot-card ${isOnline ? 'online' : ''}`}>
      <div className="card-header">
        <div>
          <div className="card-name">{robot.robot_name}</div>
          <div className="card-id">{robot.client_id}</div>
        </div>
        <div className={`status-badge ${isOnline ? 'online' : 'offline'}`}>
          <span className="status-dot" />
          {isOnline ? 'Online' : 'Offline'}
        </div>
      </div>

      {/* Current Persona */}
      <div className="card-section">
        <div className="card-label">Persona</div>
        <div className={`persona-pill ${!currentPersona ? 'none' : ''}`}>
          {currentPersona ? (
            <>
              <span style={{ opacity: 0.6 }}>◈</span>
              {currentPersona.name}
            </>
          ) : (
            <>
              <span style={{ opacity: 0.4 }}>◻</span>
              No persona assigned
            </>
          )}
        </div>
      </div>

      {/* Modules */}
      <div className="card-section">
        <div className="card-label">Modules</div>
        <div className="module-list">
          {ALL_MODULES.map(m => (
            <span
              key={m}
              className={`module-badge ${modules.includes(m) ? 'active' : ''}`}
            >
              {m}
            </span>
          ))}
        </div>
      </div>

      {/* Assign Persona */}
      <div className="card-section">
        <div className="card-label">Assign persona</div>
        <div className="assign-row">
          <select
            className="form-select"
            value={selectedPersona}
            onChange={e => setSelectedPersona(e.target.value)}
            disabled={busy}
          >
            <option value="">Select persona...</option>
            {personas.map(p => (
              <option key={p.id} value={p.id}>
                {p.name}{p.is_default ? ' (default)' : ''}
              </option>
            ))}
          </select>
          <button
            className="btn btn-primary"
            style={{ flex: '0 0 auto', padding: '8px 14px' }}
            onClick={handleAssign}
            disabled={busy || !selectedPersona}
          >
            Assign
          </button>
        </div>
      </div>

      {/* Actions */}
      <div className="card-actions">
        {isOnline ? (
          <button
            className="btn btn-disconnect"
            onClick={handleDisconnect}
            disabled={busy}
          >
            {busy ? 'Working...' : 'Disconnect'}
          </button>
        ) : (
          <button
            className="btn btn-connect"
            onClick={handleConnect}
            disabled={busy}
          >
            {busy ? 'Connecting...' : 'Connect'}
          </button>
        )}
      </div>
    </div>
  )
}