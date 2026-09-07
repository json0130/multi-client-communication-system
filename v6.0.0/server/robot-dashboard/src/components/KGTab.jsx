import { useState, useEffect, useCallback, useMemo } from 'react'
import { getKgGraph, getKgTopics, getKgSummary, seedKg, observeKg, getRobots,
         getStyleFits, rateStyle, getDemoStatus } from '../api'
import KGGraphView from './KGGraphView'

/**
 * The robot→topic competence graph.
 *
 * Two numbers per edge and they are NOT interchangeable:
 *   weight   what the system has learned, 0..1, 0.5 = no opinion
 *   clamped  that weight pulled toward 0.5 by how little evidence backs it
 *
 * Everything visual keys off `clamped`, because an edge seen once has a weight
 * that looks every bit as confident as one seen fifty times, and showing the raw
 * weight would make one supervisor's click look like an established fact. The
 * raw weight is still available in the detail panel, where the observation count
 * sits next to it.
 */

const NEUTRAL = 0.5

/** Colour for a clamped weight: red = poor fit, grey = unknown, green = good. */
function edgeColor(clamped) {
  const d = clamped - NEUTRAL
  if (Math.abs(d) < 0.02) return 'var(--muted)'
  return d > 0 ? `rgba(52,211,153,${0.35 + d * 1.3})` : `rgba(248,113,113,${0.35 - d * 1.3})`
}

/** Plain-language reading of an edge, for people who did not build this. */
function describe(edge) {
  if (!edge) return ''
  const { n_obs, weight } = edge
  if (!n_obs) return 'Never observed — the system has no opinion yet.'
  const strength = weight >= 0.7 ? 'a good fit for'
    : weight <= 0.3 ? 'a poor fit for'
      : 'roughly neutral on'
  const certainty = n_obs >= 10 ? 'and has seen enough to be fairly sure'
    : n_obs >= 4 ? 'but is still only moderately sure'
      : 'but has barely any evidence, so this is close to a guess'
  return `The system thinks this robot is ${strength} this topic, ${certainty} (${n_obs} observation${n_obs === 1 ? '' : 's'}).`
}

export default function KGTab() {
  const [edges,    setEdges]    = useState([])
  const [topics,   setTopics]   = useState([])
  const [links,    setLinks]    = useState([])
  const [summary,  setSummary]  = useState(null)
  const [robots,   setRobots]   = useState([])
  const [selected, setSelected] = useState(null)   // {robot_id, topic_id}
  const [busy,     setBusy]     = useState(false)
  const [error,    setError]    = useState('')
  const [showAll,  setShowAll]  = useState(false)  // include never-observed edges
  const [view,     setView]     = useState('matrix')  // 'matrix' | 'graph'
  const [fits,     setFits]     = useState([])     // (robot, style) audience fit
  const [runStyle, setRunStyle] = useState(null)   // style of the live run, if any

  const load = useCallback(async () => {
    setError('')
    try {
      const [g, t, s, r, f, d] = await Promise.all([
        getKgGraph(), getKgTopics(), getKgSummary(), getRobots(),
        getStyleFits(), getDemoStatus().catch(() => ({})),
      ])
      setEdges(g.edges || [])
      setTopics(t.topics || [])
      setLinks(t.links || [])
      setSummary(s)
      setRobots((r.robots || []))
      setFits(f.fits || [])
      setRunStyle(d && d.visitor_style ? d.visitor_style : null)
    } catch (e) {
      setError(e.message)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const run = async fn => {
    setBusy(true); setError('')
    try { await fn(); await load() } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  // Robots that actually have edges, plus any CURRENTLY CONNECTED robot, so a
  // robot with no evidence yet still appears rather than silently vanishing.
  //
  // This used to add every row in `robots` unconditionally — every registered
  // robot, connected or not. This project has accumulated several dead
  // duplicate rows for the same physical robot under different client_ids
  // (chatbox_01 / chatbox_jetson_001, navel_001 / navel_01 / Navel_001,
  // pepper_01 / pepper_001) from re-registration over time, and the matrix/
  // graph showed all of them as separate nodes regardless of whether anything
  // was ever connected under that id. Filtering to ws_connected (plus anything
  // with real edges, even if it has since disconnected) shows the robots that
  // are actually part of THIS deployment.
  const robotIds = useMemo(() => {
    const ids = new Set(edges.map(e => e.robot_id))
    robots.filter(r => r.ws_connected).forEach(r => ids.add(r.client_id))
    return [...ids].sort()
  }, [edges, robots])

  const edgeAt = useCallback(
    (rid, tid) => edges.find(e => e.robot_id === rid && e.topic_id === tid),
    [edges])

  const visibleTopics = useMemo(() => {
    if (showAll) return topics
    const seen = new Set(edges.filter(e => e.n_obs > 0).map(e => e.topic_id))
    return topics.filter(t => seen.has(t.id))
  }, [topics, edges, showAll])

  const sel = selected ? edgeAt(selected.robot_id, selected.topic_id) : null
  const selTopic = selected ? topics.find(t => t.id === selected.topic_id) : null

  const linksFor = tid => links
    .filter(l => l.topic_a === tid || l.topic_b === tid)
    .map(l => ({ other: l.topic_a === tid ? l.topic_b : l.topic_a, weight: l.weight }))
    .sort((a, b) => b.weight - a.weight)

  const labelOf = tid => topics.find(t => t.id === tid)?.label || tid

  return (
    <div className="kg-root">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="kg-header">
        <div>
          <div className="demo-section-title">Robot → Topic Competence</div>
          <div className="muted" style={{ fontSize: '0.76rem', marginTop: 2 }}>
            What the system has learned about which robot handles which subject.
            Green = good fit, red = poor fit, grey = not enough evidence.
          </div>
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: 2, background: 'var(--bg)', borderRadius: 6, padding: 2 }}>
            <button className={`nav-tab ${view === 'matrix' ? 'active' : ''}`}
                    style={{ padding: '4px 10px', fontSize: '0.74rem' }}
                    onClick={() => setView('matrix')}>Matrix</button>
            <button className={`nav-tab ${view === 'graph' ? 'active' : ''}`}
                    style={{ padding: '4px 10px', fontSize: '0.74rem' }}
                    onClick={() => setView('graph')}>Graph</button>
          </div>
          <label className="muted" style={{ fontSize: '0.74rem', display: 'flex', gap: 5, alignItems: 'center' }}>
            <input type="checkbox" checked={showAll} onChange={e => setShowAll(e.target.checked)} />
            show unobserved
          </label>
          <button className="btn btn-sm" disabled={busy} onClick={() => run(load)}>Refresh</button>
          <button className="btn btn-sm btn-primary" disabled={busy}
                  onClick={() => run(() => seedKg(false))}
                  title="Create the topic vocabulary and its semantic links">
            Seed vocabulary
          </button>
        </div>
      </div>

      {error && <div className="kg-error">{error}</div>}

      {/* ── Summary ────────────────────────────────────────────────────── */}
      {summary && (
        <div className="kg-stats">
          <Stat label="topics"          value={summary.topics} />
          <Stat label="topic links"     value={summary.topic_links} />
          <Stat label="edges observed"  value={`${summary.observed_edges} / ${summary.edges}`} />
          <Stat label="from supervisors" value={summary.n_supervisor} />
          <Stat label="from outcomes"    value={summary.n_outcome} />
          <Stat label="human share"
                value={`${Math.round((summary.human_share || 0) * 100)}%`}
                hint="How much of this graph came from a person rather than an automatic signal" />
        </div>
      )}

      {summary && summary.topics === 0 && (
        <div className="kg-empty">
          <strong>No vocabulary yet.</strong> The demo system has no topics of its
          own — the CHATBOX knowledge graph's topics are a child's interests and
          transfer nothing here. Press <em>Seed vocabulary</em> to create one.
        </div>
      )}

      {/* ── Matrix ─────────────────────────────────────────────────────── */}
      {visibleTopics.length > 0 && (
        <div className="kg-body">
          {view === 'graph' ? (
            <div className="kg-matrix-wrap" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <KGGraphView
                topics={visibleTopics}
                links={links}
                edges={edges}
                robots={robots}
                robotIds={robotIds}
                selected={selected}
                onSelect={setSelected}
              />
            </div>
          ) : (
          <div className="kg-matrix-wrap">
            <table className="kg-matrix">
              <thead>
                <tr>
                  <th className="kg-corner" />
                  {robotIds.map(rid => (
                    <th key={rid} className="kg-robot-head" title={rid}>
                      {robots.find(r => r.client_id === rid)?.robot_name || rid}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {visibleTopics.map(t => (
                  <tr key={t.id}>
                    <td className="kg-topic-head" title={`${t.label} (${t.category})`}>
                      {t.label}
                    </td>
                    {robotIds.map(rid => {
                      const e = edgeAt(rid, t.id)
                      const clamped = e ? e.clamped : NEUTRAL
                      const isSel = selected?.robot_id === rid && selected?.topic_id === t.id
                      return (
                        <td
                          key={rid}
                          className={`kg-cell ${isSel ? 'sel' : ''}`}
                          style={{ background: edgeColor(clamped) }}
                          title={`${clamped.toFixed(2)} · ${e?.n_obs || 0} obs`}
                          onClick={() => setSelected({ robot_id: rid, topic_id: t.id })}
                        >
                          {e?.n_obs ? clamped.toFixed(2) : ''}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          )}

          {/* ── Detail ───────────────────────────────────────────────── */}
          <div className="kg-detail">
            {!selected ? (
              <div className="muted" style={{ fontSize: '0.8rem' }}>
                Select a cell to see what the system believes and why.
              </div>
            ) : (
              <>
                <div className="demo-section-title" style={{ marginBottom: 6 }}>
                  {robots.find(r => r.client_id === selected.robot_id)?.robot_name
                    || selected.robot_id}
                  {' → '}
                  {selTopic?.label || selected.topic_id}
                </div>

                <p className="kg-plain">{describe(sel)}</p>

                <table className="kg-kv">
                  <tbody>
                    <Row k="learned weight" v={(sel?.weight ?? NEUTRAL).toFixed(3)}
                         hint="What the corrections add up to, ignoring how many there were" />
                    <Row k="shown as" v={(sel?.clamped ?? NEUTRAL).toFixed(3)}
                         hint="The weight pulled toward 0.5 because evidence is thin" />
                    <Row k="confidence" v={((sel?.confidence ?? 0) * 100).toFixed(0) + '%'} />
                    <Row k="supervisor obs" v={sel?.n_supervisor ?? 0} />
                    <Row k="outcome obs" v={sel?.n_outcome ?? 0} />
                    <Row k="last updated"
                         v={sel?.last_updated ? new Date(sel.last_updated).toLocaleString() : '—'} />
                  </tbody>
                </table>

                {/* Manual observation — the same call a correction makes. */}
                <div className="demo-section-title" style={{ margin: '12px 0 5px' }}>
                  Teach it
                </div>
                <div className="muted" style={{ fontSize: '0.73rem', marginBottom: 6 }}>
                  Each click moves the weight partway toward your answer, never all
                  the way. Later clicks move it less.
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="btn btn-sm" disabled={busy}
                          onClick={() => run(() => observeKg(selected.robot_id, selected.topic_id, 1.0))}
                          style={{ flex: 1 }}>
                    Good fit
                  </button>
                  <button className="btn btn-sm" disabled={busy}
                          onClick={() => run(() => observeKg(selected.robot_id, selected.topic_id, 0.0))}
                          style={{ flex: 1 }}>
                    Poor fit
                  </button>
                </div>

                {/* Neighbours — where a correction here would spread to. */}
                {selTopic && (
                  <>
                    <div className="demo-section-title" style={{ margin: '12px 0 5px' }}>
                      Related topics
                    </div>
                    {linksFor(selTopic.id).length === 0 ? (
                      <div className="muted" style={{ fontSize: '0.74rem' }}>
                        None. A correction here teaches this topic only.
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                        {linksFor(selTopic.id).map(n => (
                          <div key={n.other} className="kg-neighbour">
                            <span style={{ flex: 1 }}>{labelOf(n.other)}</span>
                            <span className="muted">{n.weight.toFixed(2)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}

      <StyleFitPanel
        fits={fits}
        robots={robotIds}
        runStyle={runStyle}
        busy={busy}
        onRate={(robotId, style, target) => run(() => rateStyle(robotId, style, target))}
      />
    </div>
  )
}

const STYLES = ['technical', 'business', 'interactive']

/**
 * How well each robot pitches to each kind of audience.
 *
 * Rating was reachable only by curl, which meant it was never going to
 * happen during a live tour — so the table stayed empty and the framing
 * never adapted. Two clicks per cell, and the style of the running demo is
 * highlighted so an operator does not have to remember which audience is in
 * front of them.
 *
 * Supervisor judgements only. There is deliberately no automatic path into
 * this table: a Q&A window closing cleanly says nobody objected to the
 * ROUTING, and says nothing at all about how the answer was pitched.
 */
function StyleFitPanel({ fits, robots, runStyle, busy, onRate }) {
  const byKey = {}
  fits.forEach(f => { byKey[`${f.robot_id}|${f.style}`] = f })

  return (
    <div className="kg-panel" style={{ marginTop: 18 }}>
      <div className="kg-panel-head">
        <h3 style={{ margin: 0, fontSize: 14 }}>Audience fit</h3>
        <span className="muted" style={{ fontSize: 12 }}>
          How well each robot pitches to each audience. Rate after you hear an
          answer land — or not. Never affects who answers, only how the answer
          is framed.
        </span>
      </div>

      <table className="kg-table" style={{ width: '100%' }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left' }}>Robot</th>
            {STYLES.map(st => (
              <th key={st} style={{ textAlign: 'center' }}>
                {st}
                {runStyle === st && (
                  <span className="muted" style={{ fontSize: 10, display: 'block' }}>
                    this run
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {robots.map(rid => (
            <tr key={rid}>
              <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>{rid}</td>
              {STYLES.map(st => {
                const f = byKey[`${rid}|${st}`]
                const n = f ? f.n_supervisor : 0
                const clamped = f ? Number(f.clamped) : 0.5
                // The threshold decision/style_fit.py acts on, shown rather
                // than left implicit — this is the only thing a rating
                // actually changes.
                const reinforced = clamped < 0.40
                return (
                  <td key={st} style={{ textAlign: 'center', padding: '6px 4px' }}>
                    <div style={{
                      fontFamily: 'var(--mono)', fontSize: 12,
                      color: n === 0 ? 'var(--muted)'
                           : reinforced ? 'var(--danger, #c0392b)' : 'inherit',
                    }}>
                      {n === 0 ? '—' : clamped.toFixed(2)}
                      <span className="muted" style={{ fontSize: 10 }}>
                        {n > 0 ? ` n=${n}` : ''}
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: 4, justifyContent: 'center', marginTop: 3 }}>
                      <button className="btn btn-secondary" disabled={busy}
                              style={{ padding: '1px 7px', fontSize: 11 }}
                              title={`${rid} pitched well to a ${st} audience`}
                              onClick={() => onRate(rid, st, 1.0)}>+</button>
                      <button className="btn btn-secondary" disabled={busy}
                              style={{ padding: '1px 7px', fontSize: 11 }}
                              title={`${rid} pitched poorly to a ${st} audience`}
                              onClick={() => onRate(rid, st, 0.0)}>−</button>
                    </div>
                    {reinforced && (
                      <div className="muted" style={{ fontSize: 9, marginTop: 2 }}>
                        corrective note on
                      </div>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Stat({ label, value, hint }) {
  return (
    <div className="kg-stat" title={hint || ''}>
      <div className="kg-stat-value">{value}</div>
      <div className="kg-stat-label">{label}</div>
    </div>
  )
}

function Row({ k, v, hint }) {
  return (
    <tr title={hint || ''}>
      <td className="muted">{k}</td>
      <td style={{ textAlign: 'right', fontFamily: 'var(--mono)' }}>{v}</td>
    </tr>
  )
}
