import { useState, useEffect, useCallback, useMemo } from 'react'
import { getRobots, getPersonas } from './api'
import RobotCard from './components/RobotCard'
import RobotModal from './components/RobotModal'
import PersonaCard from './components/PersonaCard'
import PersonaModal from './components/PersonaModal'
import DemoTab from './components/DemoTab'
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

function SearchIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <circle cx="6.5" cy="6.5" r="4.5"/>
      <path d="M10.5 10.5L14 14"/>
    </svg>
  )
}

export default function App() {
  const [tab, setTab]               = useState('robots')
  const [robots, setRobots]         = useState([])
  const [personas, setPersonas]     = useState([])
  const [loading, setLoading]       = useState(true)
  const [showPersonaModal, setShowPersonaModal] = useState(false)
  const [showRobotModal,   setShowRobotModal]   = useState(false)
  const [lastSync, setLastSync]     = useState(null)
  const [search, setSearch]         = useState('')
  const [filter, setFilter]         = useState('all')

  const fetchAll = useCallback(async () => {
    try {
      const [rbResp, psResp] = await Promise.all([getRobots(), getPersonas()])
      setRobots(rbResp.robots || [])
      setPersonas(psResp.personas || [])
      setLastSync(new Date())
    } catch {
      // silent background refresh — errors suppressed
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  // Silent background auto-refresh
  useEffect(() => {
    const id = setInterval(fetchAll, REFRESH_INTERVAL)
    return () => clearInterval(id)
  }, [fetchAll])

  const filteredRobots = useMemo(() => {
    return robots.filter(r => {
      const matchesSearch = !search ||
        r.robot_name?.toLowerCase().includes(search.toLowerCase()) ||
        r.client_id?.toLowerCase().includes(search.toLowerCase())
      const matchesFilter =
        filter === 'all' ||
        (filter === 'online'  &&  r.ws_connected) ||
        (filter === 'offline' && !r.ws_connected)
      return matchesSearch && matchesFilter
    })
  }, [robots, search, filter])

  const onlineRobots  = filteredRobots.filter(r =>  r.ws_connected)
  const offlineRobots = filteredRobots.filter(r => !r.ws_connected)

  const onlineCount  = robots.filter(r => r.ws_connected).length
  const offlineCount = robots.length - onlineCount

  function handlePersonaCreated(persona) {
    setPersonas(prev => [...prev, persona])
    setShowPersonaModal(false)
  }

  function handlePersonaDeleted(id) {
    setPersonas(prev => prev.filter(p => p.id !== id))
  }

  function handleRobotCreated() {
    setShowRobotModal(false)
    fetchAll()
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
          <button className={`nav-tab ${tab === 'robots'   ? 'active' : ''}`} onClick={() => setTab('robots')}>
            Robots {robots.length > 0 && `(${robots.length})`}
          </button>
          <button className={`nav-tab ${tab === 'personas' ? 'active' : ''}`} onClick={() => setTab('personas')}>
            Personas {personas.length > 0 && `(${personas.length})`}
          </button>
          <button className={`nav-tab ${tab === 'demo' ? 'active' : ''}`} onClick={() => setTab('demo')}>
            Demo
          </button>
        </nav>
        <div className="header-right">
          <button className="btn btn-secondary" style={{ padding: '5px 12px', fontSize: 12 }} onClick={fetchAll}>
            Refresh
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="main">

        {/* ── Robots tab ── */}
        {tab === 'robots' && (
          <>
            <div className="page-header">
              <div>
                <div className="page-title">Connected robots</div>
                <div className="page-meta">
                  <span style={{ color: 'var(--online)' }}>{onlineCount} online</span>
                  {' · '}
                  {offlineCount} offline
                  {lastSync && ` · synced ${lastSync.toLocaleTimeString()}`}
                </div>
              </div>
              <button
                className="btn btn-primary"
                style={{ flex: 'none', width: 'auto', padding: '9px 18px' }}
                onClick={() => setShowRobotModal(true)}
              >
                + Add robot
              </button>
            </div>

            {/* Search + filter bar */}
            <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
              <div style={{ position: 'relative', flex: 1 }}>
                <span style={{
                  position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)',
                  color: 'var(--muted)', display: 'flex',
                }}>
                  <SearchIcon />
                </span>
                <input
                  className="form-input"
                  style={{ paddingLeft: 32 }}
                  placeholder="Search by name or ID..."
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                />
              </div>
              <div style={{ display: 'flex', gap: 4 }}>
                {['all', 'online', 'offline'].map(f => (
                  <button
                    key={f}
                    className={`nav-tab ${filter === f ? 'active' : ''}`}
                    onClick={() => setFilter(f)}
                    style={{ padding: '6px 12px' }}
                  >
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {loading ? (
              <div className="loading-state">
                <div className="loading-spinner" />
                <p>Loading robots...</p>
              </div>
            ) : filteredRobots.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">◻</div>
                <p>{search || filter !== 'all' ? 'No robots match your filter.' : 'No robots registered yet.'}</p>
                {!search && filter === 'all' && (
                  <p style={{ marginTop: 8, fontSize: 11 }}>Use "+ Add robot" to register one.</p>
                )}
              </div>
            ) : (
              <>
                {/* ── Online section ── */}
                {onlineRobots.length > 0 && (
                  <div style={{ marginBottom: 28 }}>
                    <div style={{
                      display: 'flex', alignItems: 'center', gap: 8,
                      marginBottom: 12,
                    }}>
                      <span style={{
                        width: 8, height: 8, borderRadius: '50%',
                        background: 'var(--online)', display: 'inline-block',
                        boxShadow: '0 0 6px var(--online)',
                      }} />
                      <span style={{
                        fontSize: 11, fontWeight: 600, letterSpacing: '0.08em',
                        color: 'var(--online)', textTransform: 'uppercase',
                        fontFamily: 'var(--mono)',
                      }}>
                        Online — {onlineRobots.length}
                      </span>
                      <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
                    </div>
                    <div className="robot-grid">
                      {onlineRobots.map(r => (
                        <RobotCard key={r.client_id} robot={r} personas={personas} onRefresh={fetchAll} />
                      ))}
                    </div>
                  </div>
                )}

                {/* ── Offline section ── */}
                {offlineRobots.length > 0 && (
                  <div>
                    <div style={{
                      display: 'flex', alignItems: 'center', gap: 8,
                      marginBottom: 12,
                    }}>
                      <span style={{
                        width: 8, height: 8, borderRadius: '50%',
                        background: 'var(--muted)', display: 'inline-block',
                      }} />
                      <span style={{
                        fontSize: 11, fontWeight: 600, letterSpacing: '0.08em',
                        color: 'var(--muted)', textTransform: 'uppercase',
                        fontFamily: 'var(--mono)',
                      }}>
                        Offline — {offlineRobots.length}
                      </span>
                      <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
                    </div>
                    <div className="robot-grid">
                      {offlineRobots.map(r => (
                        <RobotCard key={r.client_id} robot={r} personas={personas} onRefresh={fetchAll} />
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}

        {/* ── Personas tab ── */}
        {tab === 'personas' && (
          <>
            <div className="page-header">
              <div>
                <div className="page-title">Persona library</div>
                <div className="page-meta">
                  {personas.length} persona{personas.length !== 1 ? 's' : ''}
                  {' · '}OCEAN personality model
                </div>
              </div>
              <button
                className="btn btn-primary"
                style={{ flex: 'none', width: 'auto', padding: '9px 18px' }}
                onClick={() => setShowPersonaModal(true)}
              >
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
                  <PersonaCard key={p.id} persona={p} onDeleted={handlePersonaDeleted} />
                ))
              )}
            </div>
          </>
        )}
        {/* ── Demo tab ── */}
        {tab === 'demo' && <DemoTab />}

      </main>

      {showRobotModal && (
        <RobotModal onClose={() => setShowRobotModal(false)} onCreated={handleRobotCreated} />
      )}
      {showPersonaModal && (
        <PersonaModal onClose={() => setShowPersonaModal(false)} onCreated={handlePersonaCreated} />
      )}
      <ToastContainer />
    </div>
  )
}
