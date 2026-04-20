/**
 * api.js — all calls to the Flask server
 * Uses /api prefix which Vite proxies to http://127.0.0.1:5000
 */

const BASE = '/api'

async function req(method, path, body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body) opts.body = JSON.stringify(body)
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || res.statusText)
  }
  return res.json()
}

// ── Robots ────────────────────────────────────────────────────────────────────
export const getRobots       = ()           => req('GET',    '/robots')
export const registerRobot   = (data)       => req('POST',   '/robots/register', data)
export const updateRobot     = (id, data)   => req('PUT',    `/robots/${id}`, data)
export const deleteRobot     = (id)         => req('DELETE', `/robots/${id}`)
export const connectRobot    = (id)         => req('POST',   `/robots/${id}/connect`)
export const disconnectRobot = (id)         => req('POST',   `/robots/${id}/disconnect`)
export const getRobotHealth  = (id)         => req('GET',    `/robots/${id}/health`)
export const assignPersona   = (id, persona_id) =>
  req('POST', `/robots/${id}/persona`, { persona_id })
export const chatRobot       = (id, message)    =>
  req('POST', `/robots/${id}/chat`, { message })

// ── Personas ──────────────────────────────────────────────────────────────────
export const getPersonas    = ()         => req('GET',  '/personas')
export const getPersona     = (id)       => req('GET',  `/personas/${id}`)
export const createPersona  = (data)     => req('POST', '/personas', data)
export const updatePersona  = (id, data) => req('PUT',  `/personas/${id}`, data)
export const deletePersona  = (id)       => req('DELETE', `/personas/${id}`)

// ── Demo ──────────────────────────────────────────────────────────────────────
export const getDemoStatus = ()              => req('GET',  '/demo/status')
export const startDemo     = ()              => req('POST', '/demo/start')
export const stopDemo      = ()              => req('POST', '/demo/stop')
export const pauseDemo     = ()              => req('POST', '/demo/pause')
export const resumeDemo    = ()              => req('POST', '/demo/resume')
export const nextDemoStep  = ()              => req('POST', '/demo/next')
export const startQaMode   = (message = '')  => req('POST', '/demo/qa',     { message })
export const endQaMode     = ()              => req('POST', '/demo/qa_end')

// ── Projects (RDAC) ───────────────────────────────────────────────────────────
export const getProjects          = ()              => req('GET',    '/projects')
export const createProject        = (data)          => req('POST',   '/projects', data)
export const updateProject        = (id, data)      => req('PUT',    `/projects/${id}`, data)
export const deleteProject        = (id)            => req('DELETE', `/projects/${id}`)
export const getProjectsForRobot  = (robot_id)      => req('GET',    `/projects/for/${robot_id}`)
export const grantProjectAccess   = (id, robot_id)  => req('POST',   `/projects/${id}/access`, { robot_id })
export const revokeProjectAccess  = (id, robot_id)  => req('DELETE', `/projects/${id}/access/${robot_id}`)