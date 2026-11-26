import { useState, useRef, useEffect } from 'react'
import { connectRobot, disconnectRobot, updateRobot, deleteRobot } from '../api'
import { useToast } from './Toast'

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

// `role`: this robot's entry on the Roles page (its GET /registry): what was applied there, or undefined
export default function RobotCard({ robot, role, onRefresh }) {
  const [busy, setBusy]                       = useState(false)
  const [photo, setPhoto]                     = useState(null)
  const fileRef                               = useRef(null)
  const toast = useToast()

  // ── Edit state ──────────────────────────────────────────────────────────────
  const [editing, setEditing]       = useState(false)
  const [editName, setEditName]     = useState(robot.robot_name)
  const [editIp,   setEditIp]       = useState(robot.ip_address || '')
  const [editPort, setEditPort]     = useState(robot.ws_port ? String(robot.ws_port) : '')
  const [confirmDel, setConfirmDel] = useState(false)

  const isOnline       = robot.ws_connected
  const photoKey       = `robot_photo_${robot.client_id}`

  useEffect(() => {
    const saved = localStorage.getItem(photoKey)
    if (saved) setPhoto(saved)
  }, [photoKey])

  // Reset edit form whenever the card data refreshes
  useEffect(() => {
    setEditName(robot.robot_name)
    setEditIp(robot.ip_address || '')
    setEditPort(robot.ws_port ? String(robot.ws_port) : '')
  }, [robot.robot_name, robot.ip_address, robot.ws_port])

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

  // ── Handlers ────────────────────────────────────────────────────────────────

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

  async function handleSaveEdit() {
    if (!editName.trim()) { toast.error('Name cannot be empty'); return }
    setBusy(true)
    try {
      await updateRobot(robot.client_id, {
        robot_name: editName.trim(),
        ip_address: editIp.trim() || null,
        ws_port:    editPort ? (parseInt(editPort, 10) || null) : null,
      })
      toast.success('Robot updated')
      setEditing(false)
      onRefresh()
    } catch (e) { toast.error(e.message) }
    finally { setBusy(false) }
  }

  async function handleDelete() {
    if (!confirmDel) { setConfirmDel(true); return }
    setBusy(true)
    try {
      await deleteRobot(robot.client_id)
      toast.info(`${robot.robot_name} deleted`)
      onRefresh()
    } catch (e) {
      toast.error(e.message)
      setBusy(false)
      setConfirmDel(false)
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className={`robot-card ${isOnline ? 'online' : ''}`}>

      {/* ── Inline edit form ─────────────────────────────────────────────── */}
      {editing ? (
        <div className="card-edit-form">
          <div className="card-label" style={{ marginBottom: 8 }}>Edit Robot</div>

          <label className="edit-field-label">Name</label>
          <input
            className="form-input"
            value={editName}
            onChange={e => setEditName(e.target.value)}
            placeholder="Robot name"
          />

          <label className="edit-field-label" style={{ marginTop: 8 }}>IP Address &amp; Port</label>
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              className="form-input"
              style={{ flex: 3 }}
              value={editIp}
              onChange={e => setEditIp(e.target.value)}
              placeholder="192.168.1.x"
            />
            <input
              className="form-input"
              style={{ flex: 1 }}
              value={editPort}
              onChange={e => setEditPort(e.target.value)}
              placeholder="port"
              type="number"
            />
          </div>

          <div style={{ display: 'flex', gap: 6, marginTop: 12 }}>
            <button className="btn btn-primary btn-sm" onClick={handleSaveEdit} disabled={busy}>
              {busy ? 'Saving…' : 'Save'}
            </button>
            <button className="btn btn-sm" onClick={() => { setEditing(false); setConfirmDel(false) }} disabled={busy}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* ── Header row: info + edit icon + photo ───────────────────── */}
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

            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
              {/* Edit icon button */}
              <button
                className="btn-icon"
                title="Edit robot details"
                onClick={() => { setEditing(true); setConfirmDel(false) }}
                disabled={busy}
              >
                ✎
              </button>

              {/* Photo thumbnail */}
              <div className="robot-photo" onClick={handlePhotoClick} title="Click to upload photo">
                {photo
                  ? <img src={photo} alt={robot.robot_name}
                      style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 6 }} />
                  : <RobotPlaceholder />
                }
                <div className="robot-photo-overlay"><span>📷</span></div>
              </div>
            </div>

            <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
          </div>

          {/* Persona: what the Roles page applied to this robot */}
          <div className="card-section">
            <div className="card-label">Persona</div>
            <div className={`persona-pill ${!role?.role_label ? 'none' : ''}`}>
              {role?.role_label
                ? <><span style={{ opacity: 0.6 }}>◈</span>{role.role_label}</>
                : <><span style={{ opacity: 0.4 }}>◻</span>None applied on the Roles page</>
              }
            </div>
            {role?.persona && (
              <div style={{ marginTop: 6, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--muted)', lineHeight: 1.5 }}>
                {role.persona}
              </div>
            )}
          </div>
        </>
      )}

      {/* ── Actions (always visible) ────────────────────────────────────── */}
      <div className="card-actions">
        {isOnline ? (
          <button className="btn btn-disconnect" onClick={handleDisconnect} disabled={busy}>
            {busy ? 'Working...' : 'Disconnect'}
          </button>
        ) : (
          <button className="btn btn-connect" onClick={handleConnect} disabled={busy || editing}>
            {busy ? 'Connecting...' : 'Connect'}
          </button>
        )}

        <button
          className={`btn btn-sm ${confirmDel ? 'btn-danger' : 'btn-delete'}`}
          onClick={handleDelete}
          disabled={busy}
          onBlur={() => setTimeout(() => setConfirmDel(false), 200)}
          title="Delete robot"
        >
          {confirmDel ? 'Confirm?' : 'Delete'}
        </button>
      </div>
    </div>
  )
}
