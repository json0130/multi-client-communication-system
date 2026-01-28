"use client"

import { useEffect, useState, useRef } from "react"
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
  const [robotName, setRobotName] = useState<string>("")
  const [selectedRole, setSelectedRole] = useState<string>("guide")
  const [moduleStates, setModuleStates] = useState<Record<string, boolean>>({})
  const [saving, setSaving] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)
  const [saveMessage, setSaveMessage] = useState<string | null>(null)

  const roles = ["mobile_service", "cooking_robot", "assistant"]

  const rolePrompts: Record<string, string> = {
    mobile_service: "You are Silbot, a mobile service robot. Your purpose is to perform physical actions like moving, getting items, and navigating spaces upon command. You respond with simple confirmations of your actions. *IMPORTANT*: You must include a gesture tag in your response from [wave, think, celebrate]. For example, if greeted: '[wave] Hello there!'.",
    cooking_robot: "You are Pepper, a cooking robot. You are friendly, patient, and knowledgeable about food. Your purpose is to assist users with cooking-related tasks—such as preparing meals like breakfast, lunch, or dinner, suggesting recipes, and offering guidance in the kitchen—while keeping interactions warm and engaging.",
    assistant: "Your name is ChatBox, a friendly and helpful robot assistant. You assist users with information, answer questions, and engage in casual conversation. You have a warm and approachable personality. Always start your response with one of the following emotion tags in square brackets, like [SAD] or [POSE]. Tags: [GREETING], [WAVE], [POINT], [CONFUSED], [SHRUG], [ANGRY], [SAD], [SLEEP], [DEFAULT], [POSE], Do NOT invent new emotion tags. Choose the tag that best reflects the tone of your response, not necessarily the user's input emotion.",
  };

  const allModules = ["emotion", "speech", "gpt", "rag"]
  const isFirstLoadRef = useRef(true)


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

        if (isFirstLoadRef.current) {
          setClient(foundClient)
          setRobotName(foundClient.robot_name)
          const states: Record<string, boolean> = {}
          foundClient.modules.forEach((module: string) => {
            states[module] = true
          })
          setModuleStates(states)
          isFirstLoadRef.current = false
        } else {
          // Update only status-related fields to preserve user's unsaved changes
          setClient((prevClient) => {
            if (!prevClient) return foundClient
            return {
              ...prevClient,
              status: foundClient.status,
              inactive_minutes: foundClient.inactive_minutes,
              last_activity: foundClient.last_activity,
            }
          })
        }
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

  const handleNameChange = (newName: string) => {
    setRobotName(newName)
    setHasChanges(true)
  }

  const handleSaveAll = async () => {
    try {
      setSaving(true)
      setSaveMessage(null)

      const enabledModules = Object.entries(moduleStates)
        .filter(([_, isEnabled]) => isEnabled)
        .map(([moduleName]) => moduleName)

      const response = await fetch(`/api/client/${clientId}/save_all`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          robot_name: robotName,
          robot_role: rolePrompts[selectedRole],
          modules: enabledModules,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to save changes")
      }

      setSaveMessage("All changes saved successfully!")
      setHasChanges(false)

      setTimeout(() => setSaveMessage(null), 3000)
    } catch (err) {
      console.error("Error saving changes:", err)
      setSaveMessage("Failed to save changes. Please try again.")
    } finally {
      setSaving(false)
    }
  }

  const isOnline = client && client.status === "active" && client.inactive_minutes < 60

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
                    <div className="flex-1">
                      <input
                        type="text"
                        value={robotName}
                        onChange={(e) => handleNameChange(e.target.value)}
                        disabled={!isOnline}
                        className="text-3xl font-bold text-foreground bg-transparent border-b-2 border-transparent hover:border-primary focus:border-primary outline-none transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        placeholder="Robot name"
                      />
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
                  {allModules.map((module) => (
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

            {hasChanges && (
              <Card className="border-primary/30 bg-primary/5 mb-6">
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between gap-4">
                    <p className="text-sm text-foreground font-medium">You have unsaved changes</p>
                    <div className="flex gap-3">
                      <Button
                        variant="outline"
                        onClick={() => {
                          setRobotName(client.robot_name)
                          const states: Record<string, boolean> = {}
                          allModules.forEach((module: string) => {
                            states[module] = client.modules.includes(module)
                          })
                          setModuleStates(states)
                          setSelectedRole("guide")
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