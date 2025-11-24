"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { useParams } from "next/navigation"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

interface ClientDetail {
  client_id: string
  display_name: string
  robot_name: string
  status: string
  inactive_minutes: number
  last_activity: number
  modules: string[]
  registration_time: number
}

export default function RobotDetailPage() {
  const params = useParams()
  const clientId = params.id as string
  const [client, setClient] = useState<ClientDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedRole, setSelectedRole] = useState<string>("guide")
  const [selectedCharacter, setSelectedCharacter] = useState<string>("male")
  const [moduleStates, setModuleStates] = useState<Record<string, boolean>>({})
  const [saving, setSaving] = useState(false)

  const roles = ["guide", "cooking_robot", "assistant", "greeter"]
  const characters = ["male", "female", "neutral"]

  useEffect(() => {
    const fetchClient = async () => {
      try {
        const response = await fetch("/api/clients", {
          headers: { "Content-Type": "application/json" },
        })

        if (!response.ok) {
          throw new Error("Failed to fetch clients")
        }

        const data = await response.json()
        const foundClient = data.clients?.find((c: ClientDetail) => c.client_id === clientId)
        if (!foundClient) {
          throw new Error("Client not found")
        }

        setClient(foundClient)

        // Initialize module states
        const states: Record<string, boolean> = {}
        foundClient.modules.forEach((module: string) => {
          states[module] = true
        })
        setModuleStates(states)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load client")
      } finally {
        setLoading(false)
      }
    }

    fetchClient()
    const interval = setInterval(fetchClient, 5000)
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
      console.error("Error updating module:", err)
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
      console.error("Error updating role:", err)
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
      console.error("Error updating character:", err)
    } finally {
      setSaving(false)
    }
  }

  const isOnline = client && client.status === "active" && client.inactive_minutes < 1

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-2xl mx-auto px-4 py-8">
        <Link href="/">
          <Button variant="outline" className="mb-8 bg-transparent">
            Back to Overview
          </Button>
        </Link>

        {loading ? (
          <p className="text-muted-foreground">Loading robot details...</p>
        ) : error || !client ? (
          <Card className="bg-card border-destructive/50">
            <CardContent className="pt-8">
              <p className="text-destructive">Error: {error || "Robot not found"}</p>
            </CardContent>
          </Card>
        ) : (
          <>
            <Card className="bg-card border-border mb-6">
              <CardContent className="pt-8">
                <div className="flex items-start justify-between gap-4 mb-6">
                  <div>
                    <h1 className="text-3xl font-bold text-foreground mb-2">{client.robot_name}</h1>
                    <p className="text-muted-foreground text-sm">ID: {client.client_id}</p>
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
                    <p className="text-sm text-foreground">{Math.round(client.inactive_minutes)}m ago</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Registration Time</p>
                    <p className="text-sm text-foreground">
                      {new Date(client.registration_time * 1000).toLocaleDateString()}
                    </p>
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
                  {client.modules.map((module) => (
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
          </>
        )}
      </div>
    </div>
  )
}
