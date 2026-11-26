import { useState, useEffect, useRef } from 'react'
import { ROLES_URL as URL_ } from '../api'

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
 *
 * Light/dark: the page gets the theme in its URL when the frame is created, then each change as a
 * postMessage {theme}, so switching doesn't reload the graph.
 */

const HEADER_PX = 76     // the floating .header in index.css: 12 top + 52 high + 12 gap

export default function RolesTab({ active, theme }) {
  const [src, setSrc] = useState(null)
  const frame = useRef(null)
  useEffect(() => { if (active && !src) setSrc(`${URL_}?theme=${theme}`) }, [active, src, theme])
  useEffect(() => { frame.current?.contentWindow?.postMessage({ theme }, '*') }, [theme])

  return src ? (
    <iframe ref={frame} title="Roles" src={src}
            style={{ position: 'fixed', top: HEADER_PX, left: 0, width: '100vw', height: `calc(100vh - ${HEADER_PX}px)`,
                     border: 0, background: 'var(--bg)', zIndex: 10 }} />
  ) : null
}
