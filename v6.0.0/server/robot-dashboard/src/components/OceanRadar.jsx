/**
 * OceanRadar — SVG pentagon radar chart for the OCEAN personality model.
 * Read-only (PersonaCard) or editable with sliders (PersonaModal).
 */

const KEYS   = ['O', 'C', 'E', 'A', 'N']
const LABELS = { O: 'Openness', C: 'Conscientiousness', E: 'Extraversion', A: 'Agreeableness', N: 'Neuroticism' }
const SHORT  = { O: 'Open', C: 'Cons', E: 'Extra', A: 'Agree', N: 'Neuro' }

// Angle for axis i (start at top, go clockwise)
function axisAngle(i) {
  return (i * 2 * Math.PI) / KEYS.length - Math.PI / 2
}

function polarToCart(cx, cy, r, i) {
  const a = axisAngle(i)
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) }
}

function polygonPoints(cx, cy, r, values) {
  return KEYS.map((key, i) => {
    const v = values[key] ?? 0.5
    const { x, y } = polarToCart(cx, cy, r * v, i)
    return `${x},${y}`
  }).join(' ')
}

function gridPoints(cx, cy, r, level) {
  return KEYS.map((_, i) => {
    const { x, y } = polarToCart(cx, cy, r * level, i)
    return `${x},${y}`
  }).join(' ')
}

// ── Read-only chart ────────────────────────────────────────────────────────────
export function OceanChart({ values = {}, size = 130 }) {
  const cx = size / 2
  const cy = size / 2
  const r  = size * 0.36
  const labelR = r + 17

  return (
    <svg width={size} height={size} style={{ overflow: 'visible', display: 'block' }}>
      {/* Grid rings */}
      {[0.25, 0.5, 0.75, 1].map(lv => (
        <polygon key={lv} points={gridPoints(cx, cy, r, lv)}
          fill="none" stroke="var(--border)" strokeWidth={lv === 1 ? 1 : 0.6} />
      ))}
      {/* Axis lines */}
      {KEYS.map((_, i) => {
        const tip = polarToCart(cx, cy, r, i)
        return <line key={i} x1={cx} y1={cy} x2={tip.x} y2={tip.y}
          stroke="var(--border)" strokeWidth="0.6" />
      })}
      {/* Data polygon */}
      <polygon points={polygonPoints(cx, cy, r, values)}
        fill="rgba(59,130,246,0.15)" stroke="var(--accent)" strokeWidth="1.5" strokeLinejoin="round" />
      {/* Data dots */}
      {KEYS.map((key, i) => {
        const { x, y } = polarToCart(cx, cy, r * (values[key] ?? 0.5), i)
        return <circle key={key} cx={x} cy={y} r="3"
          fill="var(--accent)" stroke="var(--bg)" strokeWidth="1.5" />
      })}
      {/* Labels */}
      {KEYS.map((key, i) => {
        const { x, y } = polarToCart(cx, cy, labelR, i)
        return (
          <text key={key} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
            fill="var(--muted)" fontSize="8" fontFamily="var(--mono)">
            {SHORT[key]}
          </text>
        )
      })}
    </svg>
  )
}

// ── Editable section (chart + sliders) ────────────────────────────────────────
export function OceanEditor({ values = {}, onChange }) {
  const size = 160
  const cx = size / 2
  const cy = size / 2
  const r  = size * 0.34
  const labelR = r + 20

  return (
    <div style={{ display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap' }}>
      {/* Chart preview */}
      <div style={{ flexShrink: 0, display: 'flex', justifyContent: 'center' }}>
        <svg width={size} height={size} style={{ overflow: 'visible' }}>
          {[0.25, 0.5, 0.75, 1].map(lv => (
            <polygon key={lv} points={gridPoints(cx, cy, r, lv)}
              fill="none" stroke="var(--border)" strokeWidth={lv === 1 ? 1 : 0.6} />
          ))}
          {KEYS.map((_, i) => {
            const tip = polarToCart(cx, cy, r, i)
            return <line key={i} x1={cx} y1={cy} x2={tip.x} y2={tip.y}
              stroke="var(--border)" strokeWidth="0.6" />
          })}
          <polygon points={polygonPoints(cx, cy, r, values)}
            fill="rgba(59,130,246,0.18)" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
          {KEYS.map((key, i) => {
            const { x, y } = polarToCart(cx, cy, r * (values[key] ?? 0.5), i)
            return <circle key={key} cx={x} cy={y} r="4"
              fill="var(--accent)" stroke="var(--bg)" strokeWidth="2" />
          })}
          {KEYS.map((key, i) => {
            const { x, y } = polarToCart(cx, cy, labelR, i)
            return (
              <text key={key} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
                fill="var(--muted)" fontSize="9" fontFamily="var(--mono)">
                {SHORT[key]}
              </text>
            )
          })}
        </svg>
      </div>

      {/* Sliders */}
      <div style={{ flex: 1, minWidth: 180, display: 'flex', flexDirection: 'column', gap: 10 }}>
        {KEYS.map(key => (
          <div key={key} className="ocean-item">
            <label>
              {LABELS[key]}
              <span>{(values[key] ?? 0.5).toFixed(1)}</span>
            </label>
            <input
              type="range" min="0" max="1" step="0.05"
              className="ocean-slider"
              value={values[key] ?? 0.5}
              onChange={e => onChange(key, parseFloat(e.target.value))}
            />
          </div>
        ))}
      </div>
    </div>
  )
}
