import { useState, useEffect, useCallback, useMemo } from 'react'
import { getKgGraph, getKgTopics, getKgSummary, seedKg, observeKg, getRobots } from '../api'

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

  const load = useCallback(async () => {
    setError('')
    try {
      const [g, t, s, r] = await Promise.all([
        getKgGraph(), getKgTopics(), getKgSummary(), getRobots(),
      ])
      setEdges(g.edges || [])
      setTopics(t.topics || [])
      setLinks(t.links || [])
      setSummary(s)
      setRobots((r.robots || []))
    } catch (e) {
      setError(e.message)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const run = async fn => {
    setBusy(true); setError('')
    try { await fn(); await load() } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  // Robots that actually have edges, plus any connected robot, so a robot with
  // no evidence yet still appears rather than silently vanishing.
  const robotIds = useMemo(() => {
    const ids = new Set(edges.map(e => e.robot_id))
    robots.forEach(r => ids.add(r.client_id))
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
