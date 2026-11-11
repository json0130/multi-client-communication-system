import { useState, useEffect } from 'react'

/**
 * Roles: assign a role + OCEAN personality to a robot and watch the role/persona graph.
 *
 * The page is served by the server's Roles service (gateway/roles_gateway.py, ROLES_PORT, default 8780), which
 * sends to the same robot connections as the rest of this dashboard. It is embedded as it is rather than
 * re-built here, and fills the whole window under the header (it is a full-screen graph, so the dashboard's
 * 1200px content column would only squeeze it).
 *
 * The iframe is only created the first time the tab is opened (the page polls its server), then stays
 * mounted like the other tabs so the graph keeps its layout. App.jsx hides this tab with display:none,
 * which hides the fixed iframe with it.
 */

const URL_ = import.meta.env.VITE_ROLES_URL || `http://${window.location.hostname}:8780/`
const HEADER_PX = 76     // the floating .header in index.css: 12 top + 52 high + 12 gap

export default function RolesTab({ active }) {
  const [opened, setOpened] = useState(false)
  useEffect(() => { if (active) setOpened(true) }, [active])

  return opened ? (
    <iframe title="Roles" src={URL_}
            style={{ position: 'fixed', top: HEADER_PX, left: 0, width: '100vw', height: `calc(100vh - ${HEADER_PX}px)`,
                     border: 0, background: '#0b0d12', zIndex: 10 }} />
  ) : null
}
