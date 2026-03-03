"use client"

import { useCallback, useRef, useEffect, useState } from "react"

export interface OceanTraits {
    openness: number
    conscientiousness: number
    extraversion: number
    agreeableness: number
    neuroticism: number
}

const TRAIT_LABELS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
const TRAIT_KEYS: (keyof OceanTraits)[] = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

interface OceanRadarChartProps {
    traits: OceanTraits
    onChange?: (traits: OceanTraits) => void
    disabled?: boolean
    size?: number
    color?: string
}

export function OceanRadarChart({
    traits,
    onChange,
    disabled = false,
    size = 280,
    color = "59, 130, 246",
}: OceanRadarChartProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const [dragging, setDragging] = useState<keyof OceanTraits | null>(null)

    const padding = 50
    const canvasSize = size + padding * 2
    const center = canvasSize / 2
    const radius = size * 0.38
    const levels = 5

    const getPoint = useCallback(
        (index: number, value: number) => {
            const angle = (Math.PI * 2 * index) / 5 - Math.PI / 2
            return {
                x: center + Math.cos(angle) * radius * value,
                y: center + Math.sin(angle) * radius * value,
            }
        },
        [center, radius],
    )

    const draw = useCallback(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        const dpr = window.devicePixelRatio || 1
        canvas.width = canvasSize * dpr
        canvas.height = canvasSize * dpr
        ctx.scale(dpr, dpr)
        ctx.clearRect(0, 0, canvasSize, canvasSize)

        // Draw pentagon grid levels
        for (let level = 1; level <= levels; level++) {
            ctx.beginPath()
            for (let i = 0; i <= 5; i++) {
                const pt = getPoint(i % 5, level / levels)
                if (i === 0) ctx.moveTo(pt.x, pt.y)
                else ctx.lineTo(pt.x, pt.y)
            }
            ctx.closePath()
            ctx.strokeStyle = level === levels ? "rgba(148, 163, 184, 0.5)" : "rgba(148, 163, 184, 0.2)"
            ctx.lineWidth = level === levels ? 1.5 : 1
            ctx.stroke()
        }

        // Draw axis lines from center
        for (let i = 0; i < 5; i++) {
            const pt = getPoint(i, 1)
            ctx.beginPath()
            ctx.moveTo(center, center)
            ctx.lineTo(pt.x, pt.y)
            ctx.strokeStyle = "rgba(148, 163, 184, 0.25)"
            ctx.lineWidth = 1
            ctx.stroke()
        }

        // Draw filled data area
        ctx.beginPath()
        for (let i = 0; i < 5; i++) {
            const value = traits[TRAIT_KEYS[i]]
            const pt = getPoint(i, value)
            if (i === 0) ctx.moveTo(pt.x, pt.y)
            else ctx.lineTo(pt.x, pt.y)
        }
        ctx.closePath()
        ctx.fillStyle = `rgba(${color}, 0.2)`
        ctx.fill()
        ctx.strokeStyle = `rgba(${color}, 0.8)`
        ctx.lineWidth = 2
        ctx.stroke()

        // Draw data points
        for (let i = 0; i < 5; i++) {
            const value = traits[TRAIT_KEYS[i]]
            const pt = getPoint(i, value)

            ctx.beginPath()
            ctx.arc(pt.x, pt.y, disabled ? 4 : 6, 0, Math.PI * 2)
            ctx.fillStyle = `rgba(${color}, 1)`
            ctx.fill()
            ctx.strokeStyle = "#ffffff"
            ctx.lineWidth = 2
            ctx.stroke()
        }

        // Draw labels
        ctx.font = "12px system-ui, -apple-system, sans-serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        for (let i = 0; i < 5; i++) {
            const angle = (Math.PI * 2 * i) / 5 - Math.PI / 2
            const labelRadius = radius + 28
            const x = center + Math.cos(angle) * labelRadius
            const y = center + Math.sin(angle) * labelRadius
            const value = traits[TRAIT_KEYS[i]]

            ctx.fillStyle = "rgba(100, 116, 139, 1)"
            ctx.fillText(TRAIT_LABELS[i], x, y - 8)
            ctx.font = "bold 12px system-ui, -apple-system, sans-serif"
            ctx.fillStyle = `rgba(${color}, 1)`
            ctx.fillText(value.toFixed(1), x, y + 8)
            ctx.font = "12px system-ui, -apple-system, sans-serif"
        }
    }, [traits, size, canvasSize, center, radius, color, disabled, getPoint])

    useEffect(() => {
        draw()
    }, [draw])

    const snapToInterval = (val: number) => {
        return Math.round(val * 10) / 10
    }

    const handleInteraction = useCallback(
        (clientX: number, clientY: number) => {
            if (disabled || !onChange) return
            const canvas = canvasRef.current
            if (!canvas) return

            const rect = canvas.getBoundingClientRect()
            const x = clientX - rect.left
            const y = clientY - rect.top

            // Find closest trait point
            let closestTrait: keyof OceanTraits | null = dragging
            let closestDist = Number.POSITIVE_INFINITY

            if (!dragging) {
                for (let i = 0; i < 5; i++) {
                    const value = traits[TRAIT_KEYS[i]]
                    const pt = getPoint(i, value)
                    const dist = Math.sqrt((x - pt.x) ** 2 + (y - pt.y) ** 2)
                    if (dist < closestDist && dist < 30) {
                        closestDist = dist
                        closestTrait = TRAIT_KEYS[i]
                    }
                }
            }

            if (closestTrait) {
                const idx = TRAIT_KEYS.indexOf(closestTrait)
                const angle = (Math.PI * 2 * idx) / 5 - Math.PI / 2
                const dx = x - center
                const dy = y - center
                const projLength = dx * Math.cos(angle) + dy * Math.sin(angle)
                let newValue = snapToInterval(Math.max(0, Math.min(1, projLength / radius)))
                if (newValue < 0.1) newValue = 0
                if (newValue > 0.9) newValue = 1

                onChange({
                    ...traits,
                    [closestTrait]: newValue,
                })
            }
        },
        [disabled, onChange, traits, dragging, center, radius, getPoint],
    )

    const handleMouseDown = (e: React.MouseEvent) => {
        if (disabled || !onChange) return
        const canvas = canvasRef.current
        if (!canvas) return
        const rect = canvas.getBoundingClientRect()
        const x = e.clientX - rect.left
        const y = e.clientY - rect.top

        for (let i = 0; i < 5; i++) {
            const value = traits[TRAIT_KEYS[i]]
            const pt = getPoint(i, value)
            const dist = Math.sqrt((x - pt.x) ** 2 + (y - pt.y) ** 2)
            if (dist < 30) {
                setDragging(TRAIT_KEYS[i])
                break
            }
        }
    }

    const handleMouseMove = (e: React.MouseEvent) => {
        if (dragging) {
            handleInteraction(e.clientX, e.clientY)
        }
    }

    const handleMouseUp = () => {
        setDragging(null)
    }

    const handleTouchStart = (e: React.TouchEvent) => {
        if (disabled || !onChange) return
        const touch = e.touches[0]
        const canvas = canvasRef.current
        if (!canvas) return
        const rect = canvas.getBoundingClientRect()
        const x = touch.clientX - rect.left
        const y = touch.clientY - rect.top

        for (let i = 0; i < 5; i++) {
            const value = traits[TRAIT_KEYS[i]]
            const pt = getPoint(i, value)
            const dist = Math.sqrt((x - pt.x) ** 2 + (y - pt.y) ** 2)
            if (dist < 30) {
                setDragging(TRAIT_KEYS[i])
                e.preventDefault()
                break
            }
        }
    }

    const handleTouchMove = (e: React.TouchEvent) => {
        if (dragging) {
            e.preventDefault()
            handleInteraction(e.touches[0].clientX, e.touches[0].clientY)
        }
    }

    return (
        <canvas
            ref={canvasRef}
            width={canvasSize}
            height={canvasSize}
            style={{ width: canvasSize, height: canvasSize }}
            className={`${disabled ? "opacity-60" : "cursor-pointer"} touch-none max-w-full h-auto`}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleMouseUp}
        />
    )
}

interface OceanSlidersProps {
    traits: OceanTraits
    onChange: (traits: OceanTraits) => void
    disabled?: boolean
}

export function OceanSliders({ traits, onChange, disabled = false }: OceanSlidersProps) {
    const handleSliderChange = (key: keyof OceanTraits, value: number) => {
        onChange({ ...traits, [key]: Math.round(value * 10) / 10 })
    }

    return (
        <div className="space-y-4">
            {TRAIT_KEYS.map((key, i) => (
                <div key={key} className="flex items-center gap-4">
                    <label className="text-sm text-muted-foreground w-36 shrink-0">{TRAIT_LABELS[i]}</label>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={traits[key]}
                        onChange={(e) => handleSliderChange(key, Number.parseFloat(e.target.value))}
                        disabled={disabled}
                        className="flex-1 h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary disabled:opacity-50 disabled:cursor-not-allowed"
                    />
                    <span className="text-sm font-medium text-foreground w-8 text-right">{traits[key].toFixed(1)}</span>
                </div>
            ))}
        </div>
    )
}

export const DEFAULT_OCEAN_TRAITS: OceanTraits = {
    openness: 0.5,
    conscientiousness: 0.5,
    extraversion: 0.5,
    agreeableness: 0.5,
    neuroticism: 0.5,
}
