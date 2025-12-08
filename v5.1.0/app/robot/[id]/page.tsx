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
  const [hasChanges, setHasChanges] = useState(false)
  const [saveMessage, setSaveMessage] = useState<string | null>(null)

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

  const handleModuleToggle = (module: string) => {
    setModuleStates((prev) => ({ ...prev, [module]: !prev[module] }))
    setHasChanges(true)
  }

  const handleRoleChange = (role: string) => {
    setSelectedRole(role)
    setHasChanges(true)
  }

  const handleCharacterChange = (character: string) => {
    setSelectedCharacter(character)
    setHasChanges(true)
  }

  const handleSaveAll = async () => {
    try {
      setSaving(true)
      setSaveMessage(null)

      // Collect all changes
      const changes = {
        modules: moduleStates,
        role: selectedRole,
        character: selectedCharacter,
      }

      // Send all changes in a single request
      const response = await fetch(`/api/client/${clientId}/save-all`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      })

      if (!response.ok) {
        throw new Error("Failed to save changes")
      }

      setSaveMessage("All changes saved successfully!")
      setHasChanges(false)

      // Clear success message after 3 seconds
      setTimeout(() => setSaveMessage(null), 3000)
    } catch (err) {
      console.error("Error saving changes:", err)
      setSaveMessage("Failed to save changes. Please try again.")
    } finally {
      setSaving(false)
    }
  }

  const isOnline = client && client.status === "active" && client.inactive_minutes < 1

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="max-w-3xl mx-auto px-6 py-6">
          <h1 className="text-3xl font-bold text-foreground">Robot Central Hub</h1>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-8">
        {/* Back Button */}
        <Link href="/" className="inline-block mb-8">
          <Button variant="ghost" className="text-primary hover:bg-primary/5 gap-2 pl-0">
            <span>←</span> Back to Overview
          </Button>
        </Link>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <p className="text-muted-foreground">Loading robot details...</p>
          </div>
        ) : error || !client ? (
          <Card className="border-destructive/30 bg-destructive/5">
            <CardContent className="pt-8">
              <p className="text-destructive font-medium mb-2">Error loading robot</p>
              <p className="text-sm text-muted-foreground">{error || "Robot not found"}</p>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Robot Header Card */}
            <Card className="border-border mb-6">
              <CardContent className="pt-8">
                <div className="flex items-start justify-between gap-4 mb-6">
                  <div className="flex items-start gap-4">
                    <div className="w-16 h-16 rounded-lg bg-primary/10 flex items-center justify-center text-3xl">
                      🤖
                    </div>
                    <div>
                      <h1 className="text-3xl font-bold text-foreground">{client.robot_name}</h1>
                      <p className="text-sm text-muted-foreground mt-1">ID: {client.client_id}</p>
                    </div>
                  </div>
                  <Badge
                    className={
                      isOnline
                        ? "bg-green-100 text-green-700 hover:bg-green-100"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-100"
                    }
                  >
                    {isOnline ? "Online" : "Offline"}
                  </Badge>
                </div>

                <div className="grid grid-cols-2 gap-4 pt-6 border-t border-border">
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-1">Status</p>
                    <p className="text-sm text-foreground">
                      {isOnline ? "Active" : "Inactive"} ({Math.round(client.inactive_minutes)} min ago)
                    </p>
                  </div>
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-1">Registered</p>
                    <p className="text-sm text-foreground">
                      {new Date(client.registration_time * 1000).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Modules Section */}
            <Card className="border-border mb-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <span>⚙️</span>
                  Modules
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {client.modules.map((module) => (
                    <div
                      key={module}
                      className="flex items-center justify-between p-4 rounded-lg bg-secondary border border-border hover:bg-secondary/80 transition-colors"
                    >
                      <span className="text-sm font-medium text-foreground capitalize">{module}</span>
                      <button
                        onClick={() => handleModuleToggle(module)}
                        disabled={!isOnline}
                        className={`relative w-12 h-7 rounded-full transition-colors ${
                          moduleStates[module] ? "bg-primary" : "bg-muted"
                        } ${!isOnline ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                        aria-label={`Toggle ${module}`}
                      >
                        <div
                          className={`absolute top-1 w-5 h-5 rounded-full bg-white transition-transform ${
                            moduleStates[module] ? "translate-x-6" : "translate-x-1"
                          }`}
                        />
                      </button>
                    </div>
                  ))}
                </div>
                {!isOnline && (
                  <p className="text-xs text-muted-foreground mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-100">
                    ⚠️ Robot must be online to modify module settings
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Role Selection */}
            <Card className="border-border mb-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <span>👤</span>
                  Role Selection
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-3">
                  {roles.map((role) => (
                    <button
                      key={role}
                      onClick={() => handleRoleChange(role)}
                      disabled={!isOnline}
                      className={`p-3 rounded-lg text-sm font-medium transition-all border ${
                        selectedRole === role
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-secondary text-foreground border-border hover:border-primary/50"
                      } ${!isOnline ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                    >
                      {role.replace("_", " ")}
                    </button>
                  ))}
                </div>
                {!isOnline && (
                  <p className="text-xs text-muted-foreground mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-100">
                    ⚠️ Robot must be online to change role
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Character Selection */}
            <Card className="border-border mb-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <span>😊</span>
                  Character Selection
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-3">
                  {characters.map((character) => (
                    <button
                      key={character}
                      onClick={() => handleCharacterChange(character)}
                      disabled={!isOnline}
                      className={`p-3 rounded-lg text-sm font-medium transition-all border capitalize ${
                        selectedCharacter === character
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-secondary text-foreground border-border hover:border-primary/50"
                      } ${!isOnline ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                    >
                      {character}
                    </button>
                  ))}
                </div>
                {!isOnline && (
                  <p className="text-xs text-muted-foreground mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-100">
                    ⚠️ Robot must be online to change character
                  </p>
                )}
              </CardContent>
            </Card>

            {hasChanges && (
              <Card className="border-primary/30 bg-primary/5 mb-6">
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between gap-4">
                    <p className="text-sm text-foreground font-medium">You have unsaved changes</p>
                    <div className="flex gap-3">
                      <Button
                        variant="outline"
                        onClick={() => {
                          // Reset to original state
                          const states: Record<string, boolean> = {}
                          client.modules.forEach((module: string) => {
                            states[module] = true
                          })
                          setModuleStates(states)
                          setSelectedRole("guide")
                          setSelectedCharacter("male")
                          setHasChanges(false)
                        }}
                        disabled={saving}
                      >
                        Cancel
                      </Button>
                      <Button onClick={handleSaveAll} disabled={saving || !isOnline} className="gap-2">
                        {saving ? "Saving..." : "Save Changes"}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {saveMessage && (
              <Card
                className={`mb-6 border-2 ${
                  saveMessage.includes("successfully") ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"
                }`}
              >
                <CardContent className="pt-4">
                  <p
                    className={`text-sm font-medium ${
                      saveMessage.includes("successfully") ? "text-green-700" : "text-red-700"
                    }`}
                  >
                    {saveMessage}
                  </p>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  )
}
