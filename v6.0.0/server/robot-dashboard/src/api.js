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
// Control calls take an optional `reason`. It is the supervisor's note on why
// they overrode the system, and it is stored as a training label — see
// decision/ on the server. Leaving it blank is fine: the timestamp and step
// already say what was wrong, and an operator mid-demo has no time to type.
export const getDemoStatus = ()              => req('GET',  '/demo/status')
export const startDemo     = (robotIds = [], timeBudgetSec = null) => req(
  'POST', '/demo/start',
  (robotIds.length || timeBudgetSec)
    ? {
        ...(robotIds.length  ? { robot_ids: robotIds }          : {}),
        // Without a budget the tour has nothing to run late against, so the
        // server never trims the script on its own.
        ...(timeBudgetSec    ? { time_budget_sec: timeBudgetSec } : {}),
      }
    : null,
)
export const stopDemo      = ()              => req('POST', '/demo/stop')
export const pauseDemo     = (reason = '')   => req('POST', '/demo/pause',  { reason })
export const resumeDemo    = (reason = '')   => req('POST', '/demo/resume', { reason })
export const nextDemoStep  = (reason = '')   => req('POST', '/demo/next',   { reason })
export const startQaMode   = (message = '', reason = '') => req('POST', '/demo/qa', { message, reason })
export const endQaMode     = (reason = '')   => req('POST', '/demo/qa_end', { reason })
// ops: [{ kind: 'skip'|'reorder'|'compress'|'extend_qa'|'drop_remaining',
//         robot_id?, position? }]
export const reviseDemo    = (ops, reason = '') => req('POST', '/demo/revise', { ops, reason })

// ── Projects (RDAC) ───────────────────────────────────────────────────────────
export const getProjects          = ()              => req('GET',    '/projects')
export const createProject        = (data)          => req('POST',   '/projects', data)
export const updateProject        = (id, data)      => req('PUT',    `/projects/${id}`, data)
export const deleteProject        = (id)            => req('DELETE', `/projects/${id}`)
export const getProjectsForRobot  = (robot_id)      => req('GET',    `/projects/for/${robot_id}`)
export const grantProjectAccess   = (id, robot_id)  => req('POST',   `/projects/${id}/access`, { robot_id })
export const revokeProjectAccess  = (id, robot_id)  => req('DELETE', `/projects/${id}/access/${robot_id}`)
// ── Knowledge graph ───────────────────────────────────────────────────────────
// The robot→topic competence graph. `weight` is what the system has learned;
// `clamped` is that weight pulled toward 0.5 by how few observations back it.
// Show clamped — the raw weight of a once-seen edge looks far more confident
// than it is.
export const getKgGraph   = (robotId = '') => req('GET', robotId ? `/kg/graph/${robotId}` : '/kg/graph')
export const getKgTopics  = ()             => req('GET',  '/kg/topics')
export const getKgSummary = ()             => req('GET',  '/kg/summary')
export const seedKg       = (dryRun = false) => req('POST', '/kg/seed', { dry_run: dryRun })
export const observeKg    = (robotId, topicId, target, kind = 'supervisor') =>
  req('POST', '/kg/observe', { robot_id: robotId, topic_id: topicId, target, kind })
