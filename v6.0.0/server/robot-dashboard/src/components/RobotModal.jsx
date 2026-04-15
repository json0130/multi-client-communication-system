import { useState } from 'react'
import { registerRobot } from '../api'
import { useToast } from './Toast'

const ALL_MODULES = ['gpt', 'speech', 'emotion', 'rag', 'navigation']

function genClientId(name) {
  const slug = name.trim().toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '') || 'robot'
  const rand = Math.random().toString(36).slice(2, 7)
  return `${slug}_${rand}`
}

const defaults = {
  robot_name: '',
  client_id:  '',
  ip_address: '',
  ws_port:    '8765',
  modules:    ['gpt', 'speech'],
}

export default function RobotModal({ onClose, onCreated }) {
  const [form, setForm] = useState(defaults)
  const [autoId, setAutoId] = useState(true)
  const [saving, setSaving] = useState(false)
  const toast = useToast()

  function setField(key, value) {
    setForm(prev => ({ ...prev, [key]: value }))
  }

  function handleNameChange(value) {
    setForm(prev => ({
      ...prev,
      robot_name: value,
      client_id:  autoId ? genClientId(value) : prev.client_id,
    }))
  }

  function handleClientIdChange(value) {
    setAutoId(false)
    setField('client_id', value)
  }

  function regenerateId() {
    setAutoId(true)
    setField('client_id', genClientId(form.robot_name))
  }

  function toggleModule(mod) {
    setForm(prev => ({
      ...prev,
      modules: prev.modules.includes(mod)
        ? prev.modules.filter(m => m !== mod)
        : [...prev.modules, mod],
    }))
  }

  async function handleSave() {
    if (!form.robot_name.trim()) { toast.error('Robot name is required'); return }
    if (!form.client_id.trim())  { toast.error('Client ID is required');  return }

    const payload = {
      robot_name: form.robot_name.trim(),
      client_id:  form.client_id.trim(),
      modules:    form.modules,
    }
    if (form.ip_address.trim()) payload.ip_address = form.ip_address.trim()
    if (form.ws_port.trim())    payload.ws_port    = parseInt(form.ws_port, 10) || 8765

    setSaving(true)
    try {
      const res = await registerRobot(payload)
      toast.success(`Robot '${form.robot_name}' registered!`)
      onCreated(res.robot || payload)
    } catch (e) {
      toast.error(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ maxWidth: 460 }}>
        <div className="modal-title">
          Add robot
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {/* Robot Name */}
        <div className="form-group">
          <label className="form-label">Robot name <span style={{ color: 'var(--danger)' }}>*</span></label>
          <input
            className="form-input"
            placeholder="e.g. Patrol Bot Alpha"
            value={form.robot_name}
            onChange={e => handleNameChange(e.target.value)}
            autoFocus
          />
        </div>

        {/* Client ID */}
        <div className="form-group">
          <label className="form-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Client ID <span style={{ color: 'var(--danger)' }}>*</span></span>
            <button
              onClick={regenerateId}
              style={{
                background: 'none', border: 'none', color: 'var(--accent)',
                fontSize: 10, cursor: 'pointer', fontFamily: 'var(--mono)', padding: 0,
              }}
            >
              ↻ regenerate
            </button>
          </label>
          <input
            className="form-input"
            style={{ fontFamily: 'var(--mono)', fontSize: 12 }}
            placeholder="unique_robot_id"
            value={form.client_id}
            onChange={e => handleClientIdChange(e.target.value)}
          />
          <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
            Unique identifier used for WebSocket connection
          </div>
        </div>

        {/* IP Address */}
        <div className="form-group">
          <label className="form-label">IP address</label>
          <input
            className="form-input"
            style={{ fontFamily: 'var(--mono)' }}
            placeholder="e.g. 192.168.1.42"
            value={form.ip_address}
            onChange={e => setField('ip_address', e.target.value)}
          />
        </div>

        {/* Port */}
        <div className="form-group">
          <label className="form-label">WebSocket port</label>
          <input
            className="form-input"
            style={{ fontFamily: 'var(--mono)' }}
            placeholder="8765"
            value={form.ws_port}
            onChange={e => setField('ws_port', e.target.value)}
          />
        </div>

        {/* Modules */}
        <div className="form-group">
          <label className="form-label">Modules</label>
          <div className="checkbox-group">
            {ALL_MODULES.map(m => (
              <label
                key={m}
                className={`checkbox-pill ${form.modules.includes(m) ? 'checked' : ''}`}
              >
                <input type="checkbox" onChange={() => toggleModule(m)} />
                {m}
              </label>
            ))}
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
            {saving ? 'Registering...' : 'Add robot'}
          </button>
        </div>
      </div>
    </div>
  )
}
