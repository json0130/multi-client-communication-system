"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { useParams } from "next/navigation"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

interface ClientHealth {
  client_id: string
  enabled_modules: string[]
  components_initialized: boolean
  last_activity: number
  current_emotion: string
  current_confidence: number
  components: Record<string, any>
}

export default function RobotDetailPage() {
  const params = useParams()
  const clientId = params.id as string
  const [health, setHealth] = useState<ClientHealth | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedRole, setSelectedRole] = useState<string>("guide")
  const [selectedCharacter, setSelectedCharacter] = useState<string>("male")
  const [moduleStates, setModuleStates] = useState<Record<string, boolean>>({})
  const [saving, setSaving] = useState(false)

  const roles = ["guide", "cooking_robot", "assistant", "greeter"]
  const characters = ["male", "female", "neutral"]

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await fetch(`/api/client/${clientId}/health`, {
          headers: { "Content-Type": "application/json" },
        })

        if (!response.ok) {
          throw new Error("Failed to fetch client health")
        }

        const data = await response.json()
        setHealth(data)

        const states: Record<string, boolean> = {}
        data.enabled_modules.forEach((module: string) => {
          states[module] = true
        })
        setModuleStates(states)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load client")
        console.error("[v0] Error fetching health:", err)
      } finally {
        setLoading(false)
      }
    }

    fetchHealth()
    const interval = setInterval(fetchHealth, 5000)
    return () => clearInterval(interval)
  }, [clientId])

  const handleModuleToggle = async (module: string) => {
    const newState = !moduleStates[module]
    setModuleStates((prev) => ({ ...prev, [module]: newState }))

    try {
      setSaving(true)
      const response = await fetch(`/api/client/${clientId}/modules`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          module,
          enabled: newState,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to update module")
      }
    } catch (err) {
      console.error("[v0] Error updating module:", err)
      setModuleStates((prev) => ({ ...prev, [module]: !newState }))
    } finally {
      setSaving(false)
    }
  }

  const handleRoleChange = async (role: string) => {
    setSelectedRole(role)

    try {
      setSaving(true)
      const response = await fetch(`/api/client/${clientId}/role`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      })

      if (!response.ok) {
        throw new Error("Failed to update role")
      }
    } catch (err) {
      console.error("[v0] Error updating role:", err)
    } finally {
      setSaving(false)
    }
  }

  const handleCharacterChange = async (character: string) => {
    setSelectedCharacter(character)

    try {
      setSaving(true)
      const response = await fetch(`/api/client/${clientId}/character`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ character }),
      })

      if (!response.ok) {
        throw new Error("Failed to update character")
      }
    } catch (err) {
      console.error("[v0] Error updating character:", err)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background p-4">
        <div className="max-w-2xl mx-auto">
          <p className="text-muted-foreground">Loading robot details...</p>
        </div>
      </div>
    )
  }

  if (error || !health) {
    return (
      <div className="min-h-screen bg-background p-4">
        <div className="max-w-2xl mx-auto">
          <Link href="/">
            <Button variant="outline">Back to Overview</Button>
          </Link>
          <Card className="mt-6 bg-card border-destructive/50">
            <CardContent className="pt-8">
              <p className="text-destructive">Error: {error || "Robot not found"}</p>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  const currentTime = Date.now() / 1000
  const isOnline = health.components_initialized && currentTime - health.last_activity < 30

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-2xl mx-auto px-4 py-8">
        <Link href="/">
          <Button variant="outline" className="mb-8 bg-transparent">
            Back to Overview
          </Button>
        </Link>

        <Card className="bg-card border-border mb-6">
          <CardContent className="pt-8">
            <div className="flex items-start justify-between gap-4 mb-6">
              <div>
                <h1 className="text-3xl font-bold text-foreground mb-2">{health.client_id}</h1>
                <p className="text-muted-foreground text-sm">Current Emotion: {health.current_emotion}</p>
              </div>
              <div className="text-right">
                <Badge className={isOnline ? "bg-green-500/20 text-green-300" : "bg-red-500/20 text-red-300"}>
                  {isOnline ? "Online" : "Offline"}
                </Badge>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border">
              <div>
                <p className="text-xs text-muted-foreground mb-1">Last Activity</p>
                <p className="text-sm text-foreground">{Math.round(currentTime - health.last_activity)}s ago</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Emotion Confidence</p>
                <p className="text-sm text-foreground">{(health.current_confidence * 100).toFixed(1)}%</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border mb-6">
          <CardHeader>
            <CardTitle className="text-lg">Enabled Modules</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {health.enabled_modules.map((module) => (
                <div
                  key={module}
                  className="flex items-center justify-between p-3 rounded-lg bg-muted/50 border border-border"
                >
                  <span className="text-sm font-medium text-foreground capitalize">{module}</span>
                  <button
                    onClick={() => handleModuleToggle(module)}
                    disabled={saving || !isOnline}
                    className={`w-12 h-6 rounded-full flex items-center transition-colors ${
                      moduleStates[module] ? "bg-primary/50" : "bg-muted"
                    } ${!isOnline ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    <div
                      className={`w-5 h-5 rounded-full bg-foreground transition-transform ${
                        moduleStates[module] ? "translate-x-6" : "translate-x-0.5"
                      }`}
                    />
                  </button>
                </div>
              ))}
            </div>
            {!isOnline && (
              <p className="text-xs text-muted-foreground mt-3">Robot must be online to modify module settings</p>
            )}
          </CardContent>
        </Card>

        <Card className="bg-card border-border mb-6">
          <CardHeader>
            <CardTitle className="text-lg">Role Selection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              {roles.map((role) => (
                <button
                  key={role}
                  onClick={() => handleRoleChange(role)}
                  disabled={saving || !isOnline}
                  className={`p-3 rounded-lg text-sm font-medium transition-colors border ${
                    selectedRole === role
                      ? "bg-primary/50 border-primary text-foreground"
                      : "bg-muted/50 border-border text-muted-foreground hover:border-primary/50"
                  } ${!isOnline ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  {role.replace("_", " ")}
                </button>
              ))}
            </div>
            {!isOnline && <p className="text-xs text-muted-foreground mt-3">Robot must be online to change role</p>}
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="text-lg">Character Selection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-3">
              {characters.map((character) => (
                <button
                  key={character}
                  onClick={() => handleCharacterChange(character)}
                  disabled={saving || !isOnline}
                  className={`p-3 rounded-lg text-sm font-medium transition-colors border ${
                    selectedCharacter === character
                      ? "bg-primary/50 border-primary text-foreground"
                      : "bg-muted/50 border-border text-muted-foreground hover:border-primary/50"
                  } ${!isOnline ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  {character}
                </button>
              ))}
            </div>
            {!isOnline && (
              <p className="text-xs text-muted-foreground mt-3">Robot must be online to change character</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
