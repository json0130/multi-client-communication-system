import { useState, useEffect, useCallback } from 'react'
import { getRobots, getPersonas } from './api'
import RobotCard from './components/RobotCard'
import PersonaCard from './components/PersonaCard'
import PersonaModal from './components/PersonaModal'
import { ToastContainer } from './components/Toast'

const REFRESH_INTERVAL = 5000

function RobotIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <rect x="3" y="5" width="10" height="8" rx="2"/>
      <path d="M6 5V3.5M10 5V3.5M6 9h.01M10 9h.01M6 11h4"/>
      <path d="M1 9h2M13 9h2"/>
    </svg>
  )
}

export default function App() {
  const [tab, setTab]           = useState('robots')
  const [robots, setRobots]     = useState([])
  const [personas, setPersonas] = useState([])
  const [loading, setLoading]   = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [lastSync, setLastSync] = useState(null)

  const fetchAll = useCallback(async () => {
    try {
      const [rbResp, psResp] = await Promise.all([getRobots(), getPersonas()])
      setRobots(rbResp.robots || [])
      setPersonas(psResp.personas || [])
      setLastSync(new Date())
    } catch (e) {
      console.error('Fetch error:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  // Initial load
  useEffect(() => { fetchAll() }, [fetchAll])

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const id = setInterval(fetchAll, REFRESH_INTERVAL)
    return () => clearInterval(id)
  }, [fetchAll])

  const onlineCount  = robots.filter(r => r.ws_connected).length
  const offlineCount = robots.length - onlineCount

  function handlePersonaCreated(persona) {
    setPersonas(prev => [...prev, persona])
    setShowModal(false)
  }

  function handlePersonaDeleted(id) {
    setPersonas(prev => prev.filter(p => p.id !== id))
  }

  return (
    <div className="layout">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="header-logo"><RobotIcon /></div>
          <span className="header-title">Robot Management</span>
        </div>
        <nav className="nav-tabs">
          <button
            className={`nav-tab ${tab === 'robots' ? 'active' : ''}`}
            onClick={() => setTab('robots')}
          >
            Robots {robots.length > 0 && `(${robots.length})`}
          </button>
          <button
            className={`nav-tab ${tab === 'personas' ? 'active' : ''}`}
            onClick={() => setTab('personas')}
          >
            Personas {personas.length > 0 && `(${personas.length})`}
          </button>
        </nav>
        <div className="header-right">
          <div className="live-dot">
            <span />
            Auto-refresh 5s
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="main">
        {/* Robots tab */}
        {tab === 'robots' && (
          <>
            <div className="page-header">
              <div>
                <div className="page-title">Connected robots</div>
                <div className="page-meta">
                  {onlineCount} online · {offlineCount} offline
                  {lastSync && ` · synced ${lastSync.toLocaleTimeString()}`}
                </div>
              </div>
            </div>

            <div className="robot-grid">
              {loading ? (
                <div className="loading-state">
                  <div className="loading-spinner" />
                  <p>Loading robots...</p>
                </div>
              ) : robots.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">◻</div>
                  <p>No robots registered yet.</p>
                  <p style={{ marginTop: 8, fontSize: 11 }}>
                    Register a robot via POST /robots/register
                  </p>
                </div>
              ) : (
                robots.map(r => (
                  <RobotCard
                    key={r.client_id}
                    robot={r}
                    personas={personas}
                    onRefresh={fetchAll}
                  />
                ))
              )}
            </div>

            <div className="refresh-bar">
              <div className="refresh-bar-inner" key={lastSync?.getTime()} />
            </div>
          </>
        )}

        {/* Personas tab */}
        {tab === 'personas' && (
          <>
            <div className="page-header">
              <div>
                <div className="page-title">Persona library</div>
                <div className="page-meta">{personas.length} persona{personas.length !== 1 ? 's' : ''}</div>
              </div>
              <button className="btn btn-primary" style={{ flex: 'none', width: 'auto', padding: '9px 18px' }} onClick={() => setShowModal(true)}>
                + New persona
              </button>
            </div>

            <div className="persona-grid">
              {loading ? (
                <div className="loading-state">
                  <div className="loading-spinner" />
                  <p>Loading personas...</p>
                </div>
              ) : personas.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">◻</div>
                  <p>No personas yet. Create one to get started.</p>
                </div>
              ) : (
                personas.map(p => (
                  <PersonaCard
                    key={p.id}
                    persona={p}
                    onDeleted={handlePersonaDeleted}
                  />
                ))
              )}
            </div>
          </>
        )}
      </main>

      {showModal && (
        <PersonaModal
          onClose={() => setShowModal(false)}
          onCreated={handlePersonaCreated}
        />
      )}

      <ToastContainer />
    </div>
  )
}