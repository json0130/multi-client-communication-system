"use client"

import { useEffect, useState, useRef } from "react"
import Link from "next/link"
import { useParams } from "next/navigation"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { OceanRadarChart, OceanSliders, DEFAULT_OCEAN_TRAITS } from "@/components/ui/ocean-radar-chart"
import type { OceanTraits } from "@/components/ui/ocean-radar-chart"

interface ClientDetail {
  client_id: string
  display_name: string
  robot_name: string
  role: string
  rolePrompt: string
  character: string
  status: string
  inactive_minutes: number
  last_activity: number
  modules: string[]
  oceanTraits: OceanTraits
  registration_time: number
}

interface RobotTemplate {
  id: string
  name: string
  role: string
  rolePrompt: string
  character: string
  modules: string[]
  description: string
  oceanTraits: OceanTraits
  createdAt: number
}

const AVAILABLE_MODULES = ["gpt", "speech", "rag", "vision", "navigation", "manipulation"]
const AVAILABLE_ROLES = ["guide", "cooking_robot", "assistant", "greeter", "security", "cleaning"]
const AVAILABLE_CHARACTERS = [
  { id: "male_friendly", name: "Male Friendly", voice: "en-US-GuyNeural" },
  { id: "female_friendly", name: "Female Friendly", voice: "en-US-JennyNeural" },
  { id: "male_professional", name: "Male Professional", voice: "en-US-DavisNeural" },
  { id: "female_professional", name: "Female Professional", voice: "en-US-AriaNeural" },
  { id: "child_friendly", name: "Child Friendly", voice: "en-US-AnaNeural" },
  { id: "elderly_friendly", name: "Elderly Friendly", voice: "en-US-SaraNeural" },
]

export default function RobotDetailPage() {
  const params = useParams()
  const clientId = params.id as string
  const [client, setClient] = useState<ClientDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [robotName, setRobotName] = useState<string>("")
  const [selectedRole, setSelectedRole] = useState<string>("")
  const [selectedRolePrompt, setSelectedRolePrompt] = useState<string>("")
  const [selectedCharacter, setSelectedCharacter] = useState<string>("")
  const [selectedModules, setSelectedModules] = useState<string[]>([])
  const [selectedOceanTraits, setSelectedOceanTraits] = useState<OceanTraits>({ ...DEFAULT_OCEAN_TRAITS })
  const [saving, setSaving] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)
  const [saveMessage, setSaveMessage] = useState<string | null>(null)
  const [templates, setTemplates] = useState<RobotTemplate[]>([])
  const [showTemplates, setShowTemplates] = useState(false)

  const isFirstLoadRef = useRef(true)

  // Load templates from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("robot-templates")
    if (saved) {
      setTemplates(JSON.parse(saved))
    }
  }, [])

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
          setRobotName(foundClient.robot_name || "")
          setSelectedRole(foundClient.role || "")
          setSelectedRolePrompt(foundClient.rolePrompt || "")
          setSelectedCharacter(foundClient.character || "")
          setSelectedModules(foundClient.modules || [])
          setSelectedOceanTraits(foundClient.oceanTraits || { ...DEFAULT_OCEAN_TRAITS })
          isFirstLoadRef.current = false
        } else {
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

  const isUnconfigured = () => {
    return (!client?.modules || client.modules.length === 0) && (!client?.role || client.role === "")
  }

  const handleModuleToggle = (module: string) => {
    setSelectedModules((prev) =>
      prev.includes(module) ? prev.filter((m) => m !== module) : [...prev, module]
    )
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

  const handleNameChange = (newName: string) => {
    setRobotName(newName)
    setHasChanges(true)
  }

  const handleOceanChange = (traits: OceanTraits) => {
    setSelectedOceanTraits(traits)
    setHasChanges(true)
  }

  const handleRolePromptChange = (prompt: string) => {
    setSelectedRolePrompt(prompt)
    setHasChanges(true)
  }

  const handleApplyTemplate = (template: RobotTemplate) => {
    setSelectedRole(template.role)
    setSelectedRolePrompt(template.rolePrompt || "")
    setSelectedCharacter(template.character || "")
    setSelectedModules(template.modules)
    setSelectedOceanTraits(template.oceanTraits || { ...DEFAULT_OCEAN_TRAITS })
    setHasChanges(true)
    setShowTemplates(false)
  }

  const handleSaveAll = async () => {
    try {
      setSaving(true)
      setSaveMessage(null)

      const response = await fetch(`/api/client/${clientId}/save_all`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          robot_name: robotName,
          robot_role: selectedRole,
          role_prompt: selectedRolePrompt,
          character: selectedCharacter,
          modules: selectedModules,
          ocean_traits: selectedOceanTraits,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to save changes")
      }

      setSaveMessage("All changes saved successfully!")
      setHasChanges(false)

      setClient((prev) =>
        prev
          ? {
            ...prev,
            robot_name: robotName,
            role: selectedRole,
            rolePrompt: selectedRolePrompt,
            character: selectedCharacter,
            modules: selectedModules,
            oceanTraits: selectedOceanTraits,
          }
          : null
      )

      setTimeout(() => setSaveMessage(null), 3000)
    } catch (err) {
      console.error("Error saving changes:", err)
      setSaveMessage("Failed to save changes. Please try again.")
    } finally {
      setSaving(false)
    }
  }

  const handleCancel = () => {
    if (client) {
      setRobotName(client.robot_name || "")
      setSelectedRole(client.role || "")
      setSelectedRolePrompt(client.rolePrompt || "")
      setSelectedCharacter(client.character || "")
      setSelectedModules(client.modules || [])
      setSelectedOceanTraits(client.oceanTraits || { ...DEFAULT_OCEAN_TRAITS })
      setHasChanges(false)
    }
  }

  const isOnline = client && client.status === "active"

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
            <span>{"<-"}</span> Back to Overview
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
            <Card className={`border-border mb-6 ${isUnconfigured() ? "border-orange-200 bg-orange-50/30" : ""}`}>
              <CardContent className="pt-8">
                <div className="flex items-start justify-between gap-4 mb-6">
                  <div className="flex items-start gap-4">
                    <div
                      className={`w-16 h-16 rounded-lg flex items-center justify-center text-3xl ${isUnconfigured() ? "bg-orange-100 text-orange-500" : "bg-primary/10 text-primary"
                        }`}
                    >
                      {isUnconfigured() ? (
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="12" cy="12" r="10" />
                          <line x1="12" y1="8" x2="12" y2="12" />
                          <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <rect x="3" y="11" width="18" height="10" rx="2" />
                          <circle cx="12" cy="5" r="2" />
                          <path d="M12 7v4" />
                          <line x1="8" y1="16" x2="8" y2="16" />
                          <line x1="16" y1="16" x2="16" y2="16" />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1">
                      <input
                        type="text"
                        value={robotName}
                        onChange={(e) => handleNameChange(e.target.value)}
                        disabled={!isOnline}
                        className="text-3xl font-bold text-foreground bg-transparent border-b-2 border-transparent hover:border-primary focus:border-primary outline-none transition-colors disabled:opacity-50 disabled:cursor-not-allowed w-full"
                        placeholder="Robot name"
                      />
                      <p className="text-sm text-muted-foreground mt-1">ID: {client.client_id}</p>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <Badge
                      className={
                        isOnline
                          ? client.inactive_minutes < 1
                            ? "bg-green-100 text-green-700 hover:bg-green-100"
                            : "bg-yellow-100 text-yellow-700 hover:bg-yellow-100"
                          : "bg-gray-100 text-gray-700 hover:bg-gray-100"
                      }
                    >
                      {isOnline ? (client.inactive_minutes < 1 ? "Active" : "Idle") : "Offline"}
                    </Badge>
                    {isUnconfigured() && (
                      <Badge variant="outline" className="bg-orange-100 text-orange-600 border-orange-200">
                        Unconfigured
                      </Badge>
                    )}
                  </div>
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

            {/* Apply Template Section */}
            <Card className={`mb-6 ${isUnconfigured() ? "border-primary/30 bg-primary/5" : "border-border"}`}>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                  {isUnconfigured() ? "Quick Setup with Template" : "Apply Template"}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  {isUnconfigured()
                    ? "Apply a pre-defined template to quickly configure this robot:"
                    : "Swap the current configuration by applying a template:"}
                </p>
                {templates.length > 0 ? (
                  <>
                    {!showTemplates ? (
                      <Button onClick={() => setShowTemplates(true)} disabled={!isOnline} variant={isUnconfigured() ? "default" : "outline"}>
                        Choose Template
                      </Button>
                    ) : (
                      <div className="space-y-3">
                        {templates.map((template) => {
                          const charName = AVAILABLE_CHARACTERS.find((c) => c.id === template.character)?.name
                          return (
                            <div
                              key={template.id}
                              className="flex items-center justify-between p-4 rounded-lg bg-card border border-border hover:border-primary/50 transition-colors"
                            >
                              <div className="flex items-center gap-4">
                                {template.oceanTraits && (
                                  <div className="shrink-0 hidden sm:block">
                                    <OceanRadarChart traits={template.oceanTraits} disabled size={80} color="59, 130, 246" />
                                  </div>
                                )}
                                <div>
                                  <p className="font-medium text-foreground">{template.name}</p>
                                  <p className="text-sm text-muted-foreground">
                                    {template.role.replace("_", " ")}
                                    {charName ? ` / ${charName}` : ""}
                                    {" - "}
                                    {template.modules.length} module{template.modules.length !== 1 ? "s" : ""}
                                  </p>
                                  {template.rolePrompt && (
                                    <p className="text-xs text-muted-foreground mt-1 line-clamp-1">{template.rolePrompt}</p>
                                  )}
                                </div>
                              </div>
                              <Button size="sm" onClick={() => handleApplyTemplate(template)} disabled={!isOnline}>
                                Apply
                              </Button>
                            </div>
                          )
                        })}
                        <Button variant="outline" size="sm" onClick={() => setShowTemplates(false)}>
                          Cancel
                        </Button>
                      </div>
                    )}
                  </>
                ) : (
                  <div>
                    <p className="text-sm text-muted-foreground mb-3">
                      No templates available yet. Create one to quickly configure robots.
                    </p>
                    <Link href="/templates">
                      <Button variant="outline" size="sm">
                        Create Template
                      </Button>
                    </Link>
                  </div>
                )}
                {!isOnline && templates.length > 0 && (
                  <p className="text-xs text-muted-foreground mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-100">
                    Robot must be online to apply a template
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Role Selection */}
            <Card className="border-border mb-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                    <circle cx="12" cy="7" r="4" />
                  </svg>
                  Role Selection
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {AVAILABLE_ROLES.map((role) => (
                    <button
                      key={role}
                      type="button"
                      onClick={() => handleRoleChange(role)}
                      disabled={!isOnline}
                      className={`p-3 rounded-lg text-sm font-medium transition-all border ${selectedRole === role
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
                    Robot must be online to change role
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Role Prompt */}
            <Card className="border-border mb-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  Role Prompt
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  System prompt given to the robot to define its behavior and understand its role.
                </p>
                <textarea
                  value={selectedRolePrompt}
                  onChange={(e) => handleRolePromptChange(e.target.value)}
                  disabled={!isOnline}
                  placeholder="e.g., You are a helpful kitchen assistant robot. You help users find recipes, guide them through cooking steps, and provide nutritional information."
                  rows={4}
                  className="flex w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring resize-y disabled:opacity-50 disabled:cursor-not-allowed"
                />
                {!isOnline && (
                  <p className="text-xs text-muted-foreground mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-100">
                    Robot must be online to change role prompt
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Character Selection */}
            <Card className="border-border mb-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                    <line x1="12" x2="12" y1="19" y2="22" />
                  </svg>
                  Character (Voice)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Select a character personality that determines the robot{"'"}s voice for TTS:
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {AVAILABLE_CHARACTERS.map((char) => (
                    <button
                      key={char.id}
                      type="button"
                      onClick={() => handleCharacterChange(char.id)}
                      disabled={!isOnline}
                      className={`p-4 rounded-lg text-left transition-all border ${selectedCharacter === char.id
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-secondary text-foreground border-border hover:border-primary/50"
                        } ${!isOnline ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                    >
                      <span className="block font-medium">{char.name}</span>
                      <span className="block text-xs opacity-70 mt-1">{char.voice}</span>
                    </button>
                  ))}
                </div>
                {!isOnline && (
                  <p className="text-xs text-muted-foreground mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-100">
                    Robot must be online to change character
                  </p>
                )}
              </CardContent>
            </Card>

            {/* OCEAN Personality Traits */}
            <Card className="border-border mb-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                  </svg>
                  OCEAN Personality Traits
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Configure the Big Five personality dimensions. Drag points on the chart or adjust the sliders (0.0 - 1.0 scale, 0.1 intervals).
                </p>
                <div className="flex flex-col items-center gap-6">
                  <div>
                    <OceanRadarChart
                      traits={selectedOceanTraits}
                      onChange={isOnline ? handleOceanChange : undefined}
                      disabled={!isOnline}
                      size={240}
                      color="59, 130, 246"
                    />
                  </div>
                  <div className="w-full">
                    <OceanSliders
                      traits={selectedOceanTraits}
                      onChange={handleOceanChange}
                      disabled={!isOnline}
                    />
                  </div>
                </div>
                {!isOnline && (
                  <p className="text-xs text-muted-foreground mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-100">
                    Robot must be online to change personality traits
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Modules Section */}
            <Card className="border-border mb-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                  Modules
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {AVAILABLE_MODULES.map((module) => (
                    <button
                      key={module}
                      type="button"
                      onClick={() => handleModuleToggle(module)}
                      disabled={!isOnline}
                      className={`p-3 rounded-lg text-sm font-medium transition-all border ${selectedModules.includes(module)
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-secondary text-foreground border-border hover:border-primary/50"
                        } ${!isOnline ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                    >
                      {module}
                    </button>
                  ))}
                </div>
                {!isOnline && (
                  <p className="text-xs text-muted-foreground mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-100">
                    Robot must be online to modify modules
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Unsaved Changes Banner */}
            {hasChanges && (
              <Card className="border-primary/30 bg-primary/5 mb-6">
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between gap-4">
                    <p className="text-sm text-foreground font-medium">You have unsaved changes</p>
                    <div className="flex gap-3">
                      <Button variant="outline" onClick={handleCancel} disabled={saving}>
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

            {/* Save Message */}
            {saveMessage && (
              <Card
                className={`mb-6 border-2 ${saveMessage.includes("successfully") ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"
                  }`}
              >
                <CardContent className="pt-4">
                  <p
                    className={`text-sm font-medium ${saveMessage.includes("successfully") ? "text-green-700" : "text-red-700"
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
