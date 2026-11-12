import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { robotColor } from '../robotColors'

/**
 * The competence graph, drawn the way the Roles page draws its graph: d3-force with the same forces and
 * constants as modules/graph_relationship/viz/index.html, so the two pages move alike. Dragging a node
 * re-heats the whole simulation, the canvas pans and zooms, hovering a node fades everything not attached
 * to it, and clicking a node pins that highlight.
 *
 * Nodes: robots (large, in the robot's colour) and topics. Lines:
 *   robot → topic   competence; colour = the clamped weight, dashed when declared but never observed
 *   topic — topic   the vocabulary's own links, thin grey
 *   robot ↔ robot   derived: the two robots share a subject, or a topic link joins one's subject to the
 *                   other's. Gold, labelled with the counts; click it for the list.
 *
 * Everything else lives INSIDE the frame: the graph's headline numbers as chips (top left), and a card
 * for the line that was clicked (top right), where an edge can also be taught. The frame is a fixed
 * height: the SVG is positioned absolutely so it can never feed its own size back into the layout.
 *
 * Colour still keys off `clamped`, not the raw weight — see KGTab's docstring for why.
 */

const NEUTRAL = 0.5
const R = 9                          // base radius, as on the Roles page
const PAIR = '#ffd24a'

function edgeColor(clamped) {
  const d = clamped - NEUTRAL
  if (Math.abs(d) < 0.02) return 'var(--muted)'
  return d > 0 ? `rgba(52,211,153,${0.45 + d * 1.1})` : `rgba(248,113,113,${0.45 - d * 1.1})`
}

/** Plain-language reading of an edge, for people who did not build this. */
function describe(edge) {
  if (!edge || !edge.n_obs) return 'Never observed — the system has no opinion yet.'
  const { n_obs, weight } = edge
  const strength = weight >= 0.7 ? 'a good fit for' : weight <= 0.3 ? 'a poor fit for' : 'roughly neutral on'
  const certainty = n_obs >= 10 ? 'and has seen enough to be fairly sure'
    : n_obs >= 4 ? 'but is still only moderately sure'
      : 'but has barely any evidence, so this is close to a guess'
  return `The system thinks this robot is ${strength} this topic, ${certainty} (${n_obs} observation${n_obs === 1 ? '' : 's'}).`
}

/** robot ↔ robot: the subjects two robots share, and the topic links that bridge one's subject to the
    other's. Two rows that are the same robot under different ids (same name) are not a pair. */
function robotPairs(robotIds, nameOf, edges, links, labelOf) {
  const topicsOf = new Map(robotIds.map(id => [id, new Set()]))
  edges.forEach(e => topicsOf.get(e.robot_id)?.add(e.topic_id))
  const pairs = []
  for (let i = 0; i < robotIds.length; i++) {
    for (let j = i + 1; j < robotIds.length; j++) {
      const a = robotIds[i], b = robotIds[j]
      if (nameOf(a) === nameOf(b)) continue
      const ta = topicsOf.get(a), tb = topicsOf.get(b)
      const shared = [...ta].filter(t => tb.has(t))
      const bridged = links
        .filter(l => (ta.has(l.topic_a) && tb.has(l.topic_b)) || (tb.has(l.topic_a) && ta.has(l.topic_b)))
        .filter(l => !(shared.includes(l.topic_a) && shared.includes(l.topic_b)))
      const strength = shared.length + bridged.reduce((s, l) => s + l.weight, 0)
      if (strength > 0) {
        pairs.push({
          robot_a: a, robot_b: b, strength,
          shared: shared.map(labelOf),
          bridged: bridged.map(l => ({ a: labelOf(l.topic_a), b: labelOf(l.topic_b), weight: l.weight })),
        })
      }
    }
  }
  return pairs
}

function highlight(st, id) {
  const { gNodes, gLinks, links } = st
  if (!id) {
    gNodes.selectAll('g.kgg-node').classed('dim', false)
    gLinks.selectAll('line').classed('dim', false)
    return
  }
  const connected = new Set([id])
  links.forEach(l => {
    if (l.source.id === id) connected.add(l.target.id)
    if (l.target.id === id) connected.add(l.source.id)
  })
  gNodes.selectAll('g.kgg-node').classed('dim', d => !connected.has(d.id))
  gLinks.selectAll('line').classed('dim', d => !(d.source.id === id || d.target.id === id))
}

/** First layout, as on the Roles page: seed on a sunflower spiral, run the forces to rest off-screen,
    then show the settled result zoomed to fit. Afterwards the simulation idles, alive for dragging. */
function settle(st) {
  const nodes = st.sim.nodes().slice().sort((a, b) => a.kind.localeCompare(b.kind))
  const golden = Math.PI * (3 - Math.sqrt(5))
  nodes.forEach((n, i) => {
    const r = 62 * Math.sqrt(i + 0.5), a = i * golden
    n.x = r * Math.cos(a); n.y = r * Math.sin(a)
    n.vx = n.vy = 0; n.fx = n.fy = null
  })
  st.sim.alpha(1)
  for (let i = 0; i < 320; i++) st.sim.tick()
  st.sim.on('tick')()
  fitView(st)
  st.sim.alpha(0.02).restart()
}

function fitView(st, pad = 90) {
  const ns = st.sim.nodes()
  if (!ns.length) return
  const x0 = d3.min(ns, d => d.x), x1 = d3.max(ns, d => d.x)
  const y0 = d3.min(ns, d => d.y), y1 = d3.max(ns, d => d.y)
  const k = Math.min(1.2, st.w / (x1 - x0 + 2 * pad), st.h / (y1 - y0 + 2 * pad))
  st.svg.call(st.zoomer.transform, d3.zoomIdentity
    .translate(st.w / 2 - k * (x0 + x1) / 2, st.h / 2 - k * (y0 + y1) / 2).scale(k))
}

const radius = d => (d.kind === 'robot' ? R * 1.6 : R * 1.05)
// a real robot is a circle, a simulated / test one a square, a topic a small circle
const symbol = d3.symbol()
  .type(d => (d.kind === 'robot' && d.sim ? d3.symbolSquare : d3.symbolCircle))
  .size(d => Math.PI * radius(d) ** 2)

function render(st, onSelectRef) {
  const { gLinks, gLabels, gNodes, sim, links } = st

  const link = gLinks.selectAll('line').data(links, d => d.key)
  link.exit().remove()
  link.enter().append('line').attr('class', 'kgg-link')
    .style('cursor', d => (d.kind === 'topic_link' ? null : 'pointer'))
    .on('click', (e, d) => {
      e.stopPropagation()
      if (d.kind === 'competence') onSelectRef.current({ robot_id: d.robot_id, topic_id: d.topic_id })
      if (d.kind === 'pair') onSelectRef.current({ robot_a: d.robot_a, robot_b: d.robot_b, shared: d.shared, bridged: d.bridged })
    })
    .append('title')
  gLinks.selectAll('line')
    .classed('pair', d => d.kind === 'pair')
    .attr('stroke', d => d.kind === 'pair' ? PAIR
      : d.kind === 'topic_link' ? 'var(--border)'
        : d.observed ? edgeColor(d.clamped) : 'var(--muted)')
    .attr('stroke-width', d => d.kind === 'pair' ? 1.5 + Math.min(6, d.strength)
      : d.kind === 'topic_link' ? 1 + d.weight * 1.5
        : d.observed ? 1.5 + d.confidence * 2.5 : 1.2)
    .attr('stroke-dasharray', d => (d.kind === 'competence' && !d.observed ? '4 4' : null))
    .select('title').text(d => {
      if (d.kind === 'pair') {
        const what = [...d.shared, ...d.bridged.map(b => `${b.a} — ${b.b}`)].join(', ')
        return `${d.source.label} ↔ ${d.target.label}: ${what}`
      }
      if (d.kind === 'topic_link') return `related · ${d.weight.toFixed(2)}`
      return `${d.source.label} → ${d.target.label} · ${d.observed ? d.clamped.toFixed(2) : 'declared, nothing observed yet'}`
    })

  const lab = gLabels.selectAll('text').data(links.filter(l => l.kind === 'pair'), d => d.key)
  lab.exit().remove()
  lab.enter().append('text').attr('class', 'kgg-pair-label').attr('text-anchor', 'middle')
  gLabels.selectAll('text')
    .text(d => `${d.shared.length} shared${d.bridged.length ? ` · ${d.bridged.length} linked` : ''}`)

  const node = gNodes.selectAll('g.kgg-node').data(sim.nodes(), d => d.id)
  node.exit().remove()
  const enter = node.enter().append('g').attr('class', 'kgg-node')
    .call(d3.drag().clickDistance(5)
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
      .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null }))
    .on('mouseenter', (e, d) => highlight(st, d.id))
    .on('mouseleave', () => highlight(st, st.pinned))
    .on('click', (e, d) => {
      e.stopPropagation()
      st.pinned = st.pinned === d.id ? null : d.id
      highlight(st, st.pinned)
    })
  enter.append('path')
  enter.append('text').attr('class', 'kgg-node-label').attr('y', 4)
  enter.append('title')

  const all = enter.merge(node)
  all.select('path')
    .attr('d', symbol)
    .attr('fill', d => (d.kind === 'robot' ? d3.color(d.color).copy({ opacity: d.sim ? 0.18 : 0.35 }).formatRgb()
      : 'var(--topic-fill)'))
    .attr('stroke', d => (d.kind === 'robot' ? d.color : 'var(--muted)'))
    .attr('stroke-width', d => (d.kind === 'robot' ? 2 : 1.5))
    .attr('stroke-dasharray', d => (d.kind === 'robot' && d.sim ? '4 3' : null))
  all.select('text')
    .attr('x', d => radius(d) + 5)
    .classed('topic', d => d.kind === 'topic')
    .text(d => d.label + (d.sim ? ' (sim)' : ''))
  all.select('title').text(d => (d.kind === 'robot' ? `${d.id}${d.sim ? ' · simulated / test client' : ' · real robot'}` : d.label))
}

export default function KGGraphView({ topics, links, edges, robots, robotIds, summary,
                                      selected, onSelect, busy, onObserve, active = true }) {
  const wrapRef = useRef(null)
  const svgRef = useRef(null)
  const stateRef = useRef(null)          // the d3 objects, which outlive React renders
  const onSelectRef = useRef(onSelect)
  onSelectRef.current = onSelect

  const nameOf = rid => robots.find(r => r.client_id === rid)?.robot_name || rid
  const labelOf = tid => topics.find(t => t.id === tid)?.label || tid
  // The record carries no sim flag. The sim / test clients (client/configs/sim, client/configs/lab) run
  // on loopback with ids like chatbox_01; the real robots are chatbox_jetson_001 / navel_001 on the LAN.
  const isSim = rid => {
    const ip = robots.find(r => r.client_id === rid)?.ip_address || ''
    return /^(127\.|localhost)/.test(ip) || /_\d{2}$/.test(rid)
  }

  // Build the simulation once.
  useEffect(() => {
    const svg = d3.select(svgRef.current)
    const root = svg.append('g')
    const st = {
      svg, root, w: 800, h: 560, nodeById: new Map(), links: [], pinned: null,
      gLinks: root.append('g'), gLabels: root.append('g'), gNodes: root.append('g'),
    }
    st.sim = d3.forceSimulation([])
      .force('link', d3.forceLink([]).id(d => d.id)
        .distance(l => (l.kind === 'pair' ? 240 : l.kind === 'topic_link' ? 90 : 130)).strength(0.35))
      .force('charge', d3.forceManyBody().strength(-700).distanceMax(600))
      .force('collide', d3.forceCollide().radius(d => (d.kind === 'robot' ? 54 : 38)).iterations(2))
      .force('x', d3.forceX(0).strength(0.04))
      .force('y', d3.forceY(0).strength(0.04))
      .alphaDecay(0.045)
      .on('tick', () => {
        st.gLinks.selectAll('line')
          .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
        st.gLabels.selectAll('text')
          .attr('x', d => (d.source.x + d.target.x) / 2)
          .attr('y', d => (d.source.y + d.target.y) / 2 - 3)
        st.gNodes.selectAll('g.kgg-node').attr('transform', d => `translate(${d.x},${d.y})`)
      })
    st.zoomer = d3.zoom().scaleExtent([0.15, 4]).on('zoom', e => root.attr('transform', e.transform))
    svg.call(st.zoomer).on('click', () => { st.pinned = null; highlight(st, null); onSelectRef.current(null) })
    // The frame's size is only read, never written back: the SVG is absolutely positioned.
    const measure = () => {
      const r = wrapRef.current.getBoundingClientRect()
      st.w = r.width || 800; st.h = r.height || 560
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(wrapRef.current)
    stateRef.current = st
    return () => { ro.disconnect(); st.sim.stop(); svg.on('.zoom', null).on('click', null); svg.selectAll('*').remove() }
  }, [])

  // Feed it the data. Existing nodes keep their positions, so a refresh or a correction nudges the
  // layout instead of reshuffling it; only the very first load runs the settle-and-fit.
  useEffect(() => {
    const st = stateRef.current
    if (!st) return
    const incoming = [
      ...robotIds.map(id => ({ id, kind: 'robot', label: nameOf(id), color: robotColor(nameOf(id)), sim: isSim(id) })),
      ...topics.map(t => ({ id: t.id, kind: 'topic', label: t.label })),
    ]
    const firstLoad = st.nodeById.size === 0
    const ids = new Set(incoming.map(n => n.id))
    for (const n of incoming) {
      const cur = st.nodeById.get(n.id)
      if (cur) Object.assign(cur, n)
      else st.nodeById.set(n.id, { ...n, x: (Math.random() - 0.5) * 80, y: (Math.random() - 0.5) * 80 })
    }
    for (const id of [...st.nodeById.keys()]) if (!ids.has(id)) st.nodeById.delete(id)
    const N = id => st.nodeById.get(id)
    st.links = [
      ...links.filter(l => N(l.topic_a) && N(l.topic_b)).map(l => ({
        source: N(l.topic_a), target: N(l.topic_b), kind: 'topic_link', weight: l.weight,
        key: `t:${l.topic_a}:${l.topic_b}` })),
      ...edges.filter(e => N(e.robot_id) && N(e.topic_id)).map(e => ({
        source: N(e.robot_id), target: N(e.topic_id), kind: 'competence', robot_id: e.robot_id, topic_id: e.topic_id,
        clamped: e.clamped, confidence: e.confidence, observed: e.n_obs > 0,
        key: `c:${e.robot_id}:${e.topic_id}` })),
      ...robotPairs(robotIds, nameOf, edges, links, labelOf).map(p => ({
        source: N(p.robot_a), target: N(p.robot_b), kind: 'pair', ...p,
        key: `p:${p.robot_a}:${p.robot_b}` })),
    ]
    st.sim.nodes([...st.nodeById.values()])
    st.sim.force('link').links(st.links)
    render(st, onSelectRef)
    if (firstLoad && st.nodeById.size) settle(st)
    else st.sim.alpha(0.35).restart()
  }, [topics, links, edges, robots, robotIds])

  // Entering the tab: measure the frame (it was display:none, so 0×0, until now) and fit the whole graph.
  useEffect(() => {
    const st = stateRef.current
    if (!st || !active) return
    const id = requestAnimationFrame(() => {
      const r = wrapRef.current?.getBoundingClientRect()
      if (r?.width) { st.w = r.width; st.h = r.height }
      fitView(st)
    })
    return () => cancelAnimationFrame(id)
  }, [active])

  // The line whose card is open, picked out in gold.
  useEffect(() => {
    const st = stateRef.current
    if (!st) return
    st.gLinks.selectAll('line').classed('sel', d =>
      (selected?.topic_id && d.kind === 'competence' && d.robot_id === selected.robot_id && d.topic_id === selected.topic_id)
      || (selected?.robot_a && d.kind === 'pair' && d.robot_a === selected.robot_a && d.robot_b === selected.robot_b))
  }, [selected, topics, links, edges])

  const sel = selected?.topic_id
    ? edges.find(e => e.robot_id === selected.robot_id && e.topic_id === selected.topic_id)
    : null

  return (
    <div ref={wrapRef} className="kg-graph-wrap">
      <svg ref={svgRef} />

      {summary && (
        <div className="kg-chips">
          <span><b>{summary.topics}</b> topics</span>
          <span><b>{summary.topic_links}</b> topic links</span>
          <span><b>{summary.observed_edges}</b> / {summary.edges} edges observed</span>
          <span title="observations made by a person"><b>{summary.n_supervisor}</b> from supervisors</span>
          <span title="observations from how a Q&A window ended"><b>{summary.n_outcome}</b> from outcomes</span>
          <span title="How much of this graph came from a person rather than an automatic signal">
            <b>{Math.round((summary.human_share || 0) * 100)}%</b> human
          </span>
        </div>
      )}

      {selected?.topic_id && (
        <div className="kg-card">
          <button className="kg-card-x" onClick={() => onSelect(null)} aria-label="close">×</button>
          <div className="kg-card-title">{nameOf(selected.robot_id)} → {labelOf(selected.topic_id)}</div>
          <p className="kg-plain">{describe(sel)}</p>
          <table className="kg-kv"><tbody>
            <tr title="What the corrections add up to, ignoring how many there were">
              <td className="muted">learned weight</td><td>{(sel?.weight ?? NEUTRAL).toFixed(3)}</td></tr>
            <tr title="The weight pulled toward 0.5 because evidence is thin">
              <td className="muted">shown as</td><td>{(sel?.clamped ?? NEUTRAL).toFixed(3)}</td></tr>
            <tr><td className="muted">confidence</td><td>{((sel?.confidence ?? 0) * 100).toFixed(0)}%</td></tr>
            <tr><td className="muted">observations</td><td>{sel?.n_supervisor ?? 0} supervisor · {sel?.n_outcome ?? 0} outcome</td></tr>
          </tbody></table>
          <div className="muted" style={{ fontSize: '0.7rem', margin: '8px 0 4px' }}>
            Teach it: each click moves the weight partway toward your answer, never all the way.
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <button className="btn btn-sm" disabled={busy} style={{ flex: 1 }}
                    onClick={() => onObserve(selected.robot_id, selected.topic_id, 1.0)}>Good fit</button>
            <button className="btn btn-sm" disabled={busy} style={{ flex: 1 }}
                    onClick={() => onObserve(selected.robot_id, selected.topic_id, 0.0)}>Poor fit</button>
          </div>
        </div>
      )}

      {selected?.robot_a && (
        <div className="kg-card">
          <button className="kg-card-x" onClick={() => onSelect(null)} aria-label="close">×</button>
          <div className="kg-card-title">{nameOf(selected.robot_a)} ↔ {nameOf(selected.robot_b)}</div>
          <p className="kg-plain">
            {selected.shared.length
              ? `Both robots are linked to ${selected.shared.length} of the same subject${selected.shared.length === 1 ? '' : 's'}.`
              : 'The two robots have no subject in common.'}
            {selected.bridged.length
              ? ` ${selected.bridged.length} topic link${selected.bridged.length === 1 ? ' joins' : 's join'} one robot's subject to the other's, so a question for one can lead to the other.`
              : ''}
          </p>
          {selected.shared.length > 0 && (
            <>
              <div className="kg-card-sub">Shared</div>
              {selected.shared.map(s => <div key={s} className="kg-neighbour"><span style={{ flex: 1 }}>{s}</span></div>)}
            </>
          )}
          {selected.bridged.length > 0 && (
            <>
              <div className="kg-card-sub">Linked</div>
              {selected.bridged.map((b, i) => (
                <div key={i} className="kg-neighbour">
                  <span style={{ flex: 1 }}>{b.a} — {b.b}</span><span className="muted">{b.weight.toFixed(2)}</span>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      <div className="kg-legend">
        <span><i className="sw" style={{ background: robotColor('chatbox') }} /> real robot</span>
        <span><i className="sw sq" style={{ borderColor: robotColor('chatbox') }} /> sim / test robot</span>
        <span><i className="sw" style={{ background: 'var(--topic-fill)', borderColor: 'var(--muted)' }} /> topic</span>
        <span><i className="ln" style={{ background: 'rgba(52,211,153,.9)' }} /> good fit</span>
        <span><i className="ln" style={{ background: 'rgba(248,113,113,.9)' }} /> poor fit</span>
        <span><i className="ln dashed" /> declared, unobserved</span>
        <span><i className="ln" style={{ background: PAIR }} /> robot ↔ robot</span>
        <span className="hint">drag · scroll to zoom · click a node to pin · click a line for its card</span>
      </div>
    </div>
  )
}
