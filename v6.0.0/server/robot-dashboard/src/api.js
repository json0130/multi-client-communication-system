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
export const getRobots       = ()           => req('GET',  '/robots')
export const registerRobot   = (data)       => req('POST', '/robots/register', data)
export const connectRobot    = (id)         => req('POST', `/robots/${id}/connect`)
export const disconnectRobot = (id)         => req('POST', `/robots/${id}/disconnect`)
export const getRobotHealth  = (id)         => req('GET',  `/robots/${id}/health`)
export const assignPersona   = (id, persona_id) =>
  req('POST', `/robots/${id}/persona`, { persona_id })

// ── Personas ──────────────────────────────────────────────────────────────────
export const getPersonas    = ()         => req('GET',  '/personas')
export const getPersona     = (id)       => req('GET',  `/personas/${id}`)
export const createPersona  = (data)     => req('POST', '/personas', data)
export const updatePersona  = (id, data) => req('PUT',  `/personas/${id}`, data)
export const deletePersona  = (id)       => req('DELETE', `/personas/${id}`)