import { useState, useEffect, useCallback, useMemo } from 'react'
import { getKgGraph, getKgTopics, getKgSummary, seedKg, observeKg, getRobots,
         getStyleFits, rateStyle, getDemoStatus } from '../api'
import KGGraphView from './KGGraphView'

/**
 * The robot→topic competence graph, drawn like the Roles page (see KGGraphView).
 *
 * Two numbers per edge and they are NOT interchangeable:
 *   weight   what the system has learned, 0..1, 0.5 = no opinion
 *   clamped  that weight pulled toward 0.5 by how little evidence backs it
 *
 * Everything visual keys off `clamped`, because an edge seen once has a weight
 * that looks every bit as confident as one seen fifty times, and showing the raw
 * weight would make one supervisor's click look like an established fact. The
 * raw weight is still shown in the edge's card, next to the observation count.
 */

const POLL_MS = 20000

export default function KGTab({ active = true }) {
  const [edges,    setEdges]    = useState([])
  const [topics,   setTopics]   = useState([])
  const [links,    setLinks]    = useState([])
  const [summary,  setSummary]  = useState(null)
  const [robots,   setRobots]   = useState([])
  const [selected, setSelected] = useState(null)   // {robot_id, topic_id} | {robot_a, robot_b, ...}
  const [busy,     setBusy]     = useState(false)
  const [error,    setError]    = useState('')
  const [fits,     setFits]     = useState([])     // (robot, style) audience fit
  const [runStyle, setRunStyle] = useState(null)   // style of the live run, if any
  const [fitOpen,  setFitOpen]  = useState(false)  // the Audience fit bubble

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

  // No Refresh button: the graph keeps itself current.
  useEffect(() => {
    load()
    const t = setInterval(load, POLL_MS)
    return () => clearInterval(t)
  }, [load])

  const run = async fn => {
    setBusy(true); setError('')
    try { await fn(); await load() } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  // Robots that actually have edges, plus any CURRENTLY CONNECTED robot, so a
  // robot with no evidence yet still appears rather than silently vanishing.
  // Not every registered row: this project has accumulated dead duplicate
  // rows for the same physical robot under different client_ids.
  const robotIds = useMemo(() => {
    const ids = new Set(edges.map(e => e.robot_id))
    robots.filter(r => r.ws_connected).forEach(r => ids.add(r.client_id))
    return [...ids].sort()
  }, [edges, robots])

  // A topic is shown as soon as SOME robot is linked to it, observed or not:
  // declared scope is configuration, and hiding it leaves robots floating
  // unconnected as though nothing had been configured.
  const graphTopics = useMemo(() => {
    const linked = new Set(edges.map(e => e.topic_id))
    return topics.filter(t => linked.has(t.id))
  }, [topics, edges])

  return (
    <div className="kg-root">
      {error && <div className="kg-error">{error}</div>}

      {summary && summary.topics === 0 ? (
        <div className="kg-empty">
          <strong>No vocabulary yet.</strong> The demo system has no topics of its
          own — the CHATBOX knowledge graph's topics are a child's interests and
          transfer nothing here.{' '}
          <button className="btn btn-sm btn-primary" disabled={busy}
                  onClick={() => run(() => seedKg(false))}
                  title="Create the topic vocabulary and its semantic links">
            Seed vocabulary
          </button>
        </div>
      ) : (
        <KGGraphView
          topics={graphTopics}
          links={links}
          edges={edges}
          robots={robots}
          robotIds={robotIds}
          summary={summary}
          selected={selected}
          onSelect={setSelected}
          busy={busy}
          active={active}
          onObserve={(robotId, topicId, target) => run(() => observeKg(robotId, topicId, target))}
        />
      )}

      {fitOpen && (
        <StyleFitPanel
          fits={fits}
          robots={robotIds}
          runStyle={runStyle}
          busy={busy}
          onRate={(robotId, style, target) => run(() => rateStyle(robotId, style, target))}
        />
      )}
      <button className={`kg-fit-bubble ${fitOpen ? 'open' : ''}`} onClick={() => setFitOpen(o => !o)}
              title="Rate how well each robot pitched to the audience">
        Audience fit {fitOpen ? '×' : ''}
      </button>
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
    <div className="kg-fit-panel">
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
