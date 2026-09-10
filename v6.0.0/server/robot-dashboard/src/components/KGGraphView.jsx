import { useEffect, useMemo, useRef, useState } from 'react'

/**
 * Obsidian-style force-directed rendering of the competence graph.
 *
 * No graph library is installed (react/react-dom only), so this is a small
 * hand-written force simulation rather than a dependency — repulsion between
 * every node pair, springs along edges, a weak pull to center, damping. At this
 * graph's scale (well under a hundred nodes) an O(n²) simulation settles in a
 * few dozen frames with no perceptible cost.
 *
 * Matches Obsidian's graph view in the ways that make it read as "a graph"
 * rather than a diagram: nodes repel until edges pull them into a natural
 * layout, hovering a node highlights its neighbourhood and fades the rest,
 * and nodes are draggable.
 *
 * Colour still keys off `clamped`, not the raw weight — see KGTab's docstring
 * for why: an edge seen once must not look as certain as one seen fifty times.
 */

const NEUTRAL = 0.5
const WIDTH = 640
const HEIGHT = 560
const CENTER = { x: WIDTH / 2, y: HEIGHT / 2 }

// Physics constants. Tuned by feel, not measurement — this is a layout aid,
// not a simulation anything downstream depends on being "correct".
const REPULSION = 2600
const SPRING_TOPIC_LINK = 0.02
const SPRING_EDGE = 0.05
const CENTER_PULL = 0.012
const DAMPING = 0.82
const SETTLE_FRAMES = 240

function edgeColor(clamped) {
  const d = clamped - NEUTRAL
  if (Math.abs(d) < 0.02) return 'var(--muted)'
  return d > 0 ? `rgba(52,211,153,${0.45 + d * 1.1})` : `rgba(248,113,113,${0.45 - d * 1.1})`
}

/** Stable pseudo-random seed from a string id, so layout doesn't jump between
    renders of the same data — only the physics moves nodes, not React. */
function seededPos(id) {
  let h = 0
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0
  const a = (h % 1000) / 1000 * Math.PI * 2
  const r = 60 + (h % 137)
  return { x: CENTER.x + r * Math.cos(a), y: CENTER.y + r * Math.sin(a) }
}

export default function KGGraphView({ topics, links, edges, robots, robotIds, selected, onSelect }) {
  // Every DECLARED edge is drawn, observed or not. Only observed ones used to
  // be, and with a fresh graph that left four robots as unconnected dots and
  // one lone link — a picture that reads as "nothing is wired up" when in
  // fact each robot's whole subject scope was sitting in the payload,
  // n_obs = 0. Declared scope is configuration, not evidence, so hiding it
  // behind an evidence threshold answers a question nobody asked.
  //
  // The distinction still has to be visible, so it is carried in the STROKE
  // (see below) rather than in whether the line exists: solid and coloured
  // for something the system has actually observed, dashed and faint for a
  // subject it has merely been told this robot owns.
  const competenceEdges = edges

  // ── Node/edge model, rebuilt only when the DATA (not physics) changes ──────
  const { nodeList, linkList } = useMemo(() => {
    const nodes = [
      ...robotIds.map(id => ({ id, kind: 'robot',
        label: robots.find(r => r.client_id === id)?.robot_name || id })),
      ...topics.map(t => ({ id: t.id, kind: 'topic', label: t.label })),
    ]
    const edgeLinks = [
      ...links.map(l => ({ a: l.topic_a, b: l.topic_b, kind: 'topic_link', weight: l.weight })),
      ...competenceEdges.map(e => ({
        a: e.robot_id, b: e.topic_id, kind: 'competence',
        clamped: e.clamped, confidence: e.confidence,
        observed: e.n_obs > 0,
      })),
    ]
    return { nodeList: nodes, linkList: edgeLinks }
  }, [topics, links, robotIds, robots, competenceEdges])

  // Physics state lives in a ref (mutated every frame) so re-renders don't
  // fight the simulation; a small tick counter in React state repaints the SVG.
  const posRef = useRef(new Map())
  const velRef = useRef(new Map())
  const dragRef = useRef(null)     // node id currently being dragged, or null
  const [, forceTick] = useState(0)
  const [hovered, setHovered] = useState(null)

  // (Re)seed any node that doesn't have a position yet — new nodes join the
  // existing layout instead of restarting it, so seeding a fresh topic doesn't
  // reshuffle every node already settled.
  useEffect(() => {
    for (const n of nodeList) {
      if (!posRef.current.has(n.id)) posRef.current.set(n.id, seededPos(n.id))
      if (!velRef.current.has(n.id)) velRef.current.set(n.id, { x: 0, y: 0 })
    }
    const ids = new Set(nodeList.map(n => n.id))
    for (const id of [...posRef.current.keys()]) {
      if (!ids.has(id)) { posRef.current.delete(id); velRef.current.delete(id) }
    }
  }, [nodeList])

  // The settle animation: run for SETTLE_FRAMES after the node/link set
  // changes, then stop — Obsidian's graph does the same "drop and settle"
  // rather than animating forever, which would just burn a CPU core for no
  // visible benefit once it has converged.
  useEffect(() => {
    let frame = 0
    let raf = null
    const step = () => {
      const pos = posRef.current, vel = velRef.current
      // Repulsion, every pair.
      for (const a of nodeList) {
        for (const b of nodeList) {
          if (a.id >= b.id) continue
          const pa = pos.get(a.id), pb = pos.get(b.id)
          let dx = pa.x - pb.x, dy = pa.y - pb.y
          let d2 = dx * dx + dy * dy || 0.01
          const f = REPULSION / d2
          const d = Math.sqrt(d2)
          dx /= d; dy /= d
          if (dragRef.current !== a.id) { vel.get(a.id).x += dx * f; vel.get(a.id).y += dy * f }
          if (dragRef.current !== b.id) { vel.get(b.id).x -= dx * f; vel.get(b.id).y -= dy * f }
        }
      }
      // Springs along edges.
      for (const l of linkList) {
        const pa = pos.get(l.a), pb = pos.get(l.b)
        if (!pa || !pb) continue
        const k = l.kind === 'topic_link' ? SPRING_TOPIC_LINK : SPRING_EDGE
        const dx = pb.x - pa.x, dy = pb.y - pa.y
        if (dragRef.current !== l.a) { vel.get(l.a).x += dx * k; vel.get(l.a).y += dy * k }
        if (dragRef.current !== l.b) { vel.get(l.b).x -= dx * k; vel.get(l.b).y -= dy * k }
      }
      // Weak center pull so the whole thing doesn't drift off-canvas.
      for (const n of nodeList) {
        if (dragRef.current === n.id) continue
        const p = pos.get(n.id), v = vel.get(n.id)
        v.x += (CENTER.x - p.x) * CENTER_PULL
        v.y += (CENTER.y - p.y) * CENTER_PULL
        v.x *= DAMPING; v.y *= DAMPING
        p.x += v.x; p.y += v.y
        p.x = Math.max(20, Math.min(WIDTH - 20, p.x))
        p.y = Math.max(20, Math.min(HEIGHT - 20, p.y))
      }
      frame += 1
      forceTick(t => t + 1)
      if (frame < SETTLE_FRAMES) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [nodeList, linkList])

  // ── Dragging ────────────────────────────────────────────────────────────────
  const svgRef = useRef(null)
  const toSvgPoint = (evt) => {
    const rect = svgRef.current.getBoundingClientRect()
    return {
      x: (evt.clientX - rect.left) * (WIDTH / rect.width),
      y: (evt.clientY - rect.top) * (HEIGHT / rect.height),
    }
  }
  const startDrag = (id) => (evt) => {
    evt.preventDefault()
    dragRef.current = id
    const move = (e) => {
      const p = toSvgPoint(e)
      posRef.current.set(id, p)
      velRef.current.set(id, { x: 0, y: 0 })
      forceTick(t => t + 1)
    }
    const up = () => {
      dragRef.current = null
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  // ── Highlighting ──────────────────────────────────────────────────────────
  const focus = hovered || (selected ? selected.robot_id || selected.topic_id : null)
  const neighboursOf = (id) => {
    const set = new Set([id])
    for (const l of linkList) {
      if (l.a === id) set.add(l.b)
      if (l.b === id) set.add(l.a)
    }
    return set
  }
  const dimmed = focus ? neighboursOf(focus) : null

  return (
    <svg ref={svgRef} viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
         style={{ width: '100%', height: '100%', maxHeight: 580, touchAction: 'none' }}>
      {linkList.map((l, i) => {
        const pa = posRef.current.get(l.a), pb = posRef.current.get(l.b)
        if (!pa || !pb) return null
        const faded = dimmed && !(dimmed.has(l.a) && dimmed.has(l.b))
        if (l.kind === 'topic_link') {
          return (
            <line key={`tl${i}`} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                  stroke="var(--border)" strokeWidth={1 + l.weight * 1.5}
                  opacity={faded ? 0.08 : 0.55} />
          )
        }
        const isSel = selected?.robot_id === l.a && selected?.topic_id === l.b
        // Declared but never observed: the robot owns the subject, the system
        // has no evidence about how well it handles it. Dashed and faint, so
        // scope reads as scope and is never mistaken for a learned opinion.
        if (!l.observed) {
          return (
            <line key={`ce${i}`} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                  stroke="var(--border)" strokeWidth={isSel ? 3 : 1.2}
                  strokeDasharray="4 4"
                  opacity={faded ? 0.08 : 0.5}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onSelect({ robot_id: l.a, topic_id: l.b })} />
          )
        }
        return (
          <line key={`ce${i}`} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                stroke={edgeColor(l.clamped)}
                strokeWidth={isSel ? 4 : 1.5 + l.confidence * 2.5}
                opacity={faded ? 0.1 : 1}
                style={{ cursor: 'pointer' }}
                onClick={() => onSelect({ robot_id: l.a, topic_id: l.b })} />
        )
      })}

      {nodeList.map(n => {
        const p = posRef.current.get(n.id)
        if (!p) return null
        const isRobot = n.kind === 'robot'
        const faded = dimmed && !dimmed.has(n.id)
        return (
          <g key={n.id}
             opacity={faded ? 0.25 : 1}
             style={{ cursor: 'grab' }}
             onPointerDown={startDrag(n.id)}
             onMouseEnter={() => setHovered(n.id)}
             onMouseLeave={() => setHovered(null)}>
            <circle cx={p.x} cy={p.y} r={isRobot ? 11 : 5.5}
                    fill={isRobot ? 'var(--accent-dim)' : 'var(--surface2)'}
                    stroke={isRobot ? 'var(--accent)' : 'var(--border)'}
                    strokeWidth={isRobot ? 1.8 : 1} />
            <text x={p.x} y={p.y + (isRobot ? 24 : -9)}
                  fontSize={isRobot ? 10.5 : 9}
                  fontWeight={isRobot ? 600 : 400}
                  fill={isRobot ? 'var(--text)' : 'var(--muted)'}
                  textAnchor="middle" style={{ userSelect: 'none' }}>
              {n.label}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
