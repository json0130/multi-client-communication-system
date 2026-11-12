/**
 * One colour per robot, shared by the Demo chat and the KG graph: the Roles graph's orange (ChatBox) and
 * blue (Navel) a shade lighter, a fixed colour for the other lab robots, and a small palette for anything
 * else, picked from the name so it never changes between renders.
 */
export const ROBOT_COLORS = {
  chatbox: '#ffa46b', navel: '#7dd0f9', pepper: '#c792ea', silbot: '#7fd98b', irobi: '#ffd24a',
}
const FALLBACK = ['#f28e8e', '#8ecae6', '#b5e48c', '#f9c74f', '#cdb4db']

export function robotColor(name) {
  const key = String(name || '').toLowerCase()
  const hit = Object.keys(ROBOT_COLORS).find(k => key.includes(k))
  if (hit) return ROBOT_COLORS[hit]
  let h = 0
  for (const c of key) h = (h * 31 + c.charCodeAt(0)) >>> 0
  return FALLBACK[h % FALLBACK.length]
}
