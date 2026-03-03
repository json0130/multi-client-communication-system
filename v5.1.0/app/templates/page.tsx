"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { OceanRadarChart, OceanSliders, DEFAULT_OCEAN_TRAITS } from "@/components/ui/ocean-radar-chart"
import type { OceanTraits } from "@/components/ui/ocean-radar-chart"

export interface RobotTemplate {
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

export default function TemplatesPage() {
  const [templates, setTemplates] = useState<RobotTemplate[]>([])
  const [isCreating, setIsCreating] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)

  // Form state
  const [formName, setFormName] = useState("")
  const [formRole, setFormRole] = useState("")
  const [formRolePrompt, setFormRolePrompt] = useState("")
  const [formCharacter, setFormCharacter] = useState("")
  const [formModules, setFormModules] = useState<string[]>([])
  const [formDescription, setFormDescription] = useState("")
  const [formOceanTraits, setFormOceanTraits] = useState<OceanTraits>({ ...DEFAULT_OCEAN_TRAITS })

  useEffect(() => {
    const saved = localStorage.getItem("robot-templates")
    if (saved) {
      setTemplates(JSON.parse(saved))
    }
  }, [])

  const saveTemplates = (newTemplates: RobotTemplate[]) => {
    localStorage.setItem("robot-templates", JSON.stringify(newTemplates))
    setTemplates(newTemplates)
  }

  const resetForm = () => {
    setFormName("")
    setFormRole("")
    setFormRolePrompt("")
    setFormCharacter("")
    setFormModules([])
    setFormDescription("")
    setFormOceanTraits({ ...DEFAULT_OCEAN_TRAITS })
    setIsCreating(false)
    setEditingId(null)
  }

  const handleCreateTemplate = () => {
    if (!formName || !formRole) return

    const newTemplate: RobotTemplate = {
      id: editingId || `template-${Date.now()}`,
      name: formName,
      role: formRole,
      rolePrompt: formRolePrompt,
      character: formCharacter,
      modules: formModules,
      description: formDescription,
      oceanTraits: formOceanTraits,
      createdAt: editingId ? templates.find((t) => t.id === editingId)?.createdAt || Date.now() : Date.now(),
    }

    if (editingId) {
      saveTemplates(templates.map((t) => (t.id === editingId ? newTemplate : t)))
    } else {
      saveTemplates([...templates, newTemplate])
    }

    resetForm()
  }

  const handleEditTemplate = (template: RobotTemplate) => {
    setFormName(template.name)
    setFormRole(template.role)
    setFormRolePrompt(template.rolePrompt || "")
    setFormCharacter(template.character || "")
    setFormModules(template.modules)
    setFormDescription(template.description)
    setFormOceanTraits(template.oceanTraits || { ...DEFAULT_OCEAN_TRAITS })
    setEditingId(template.id)
    setIsCreating(true)
  }

  const handleDeleteTemplate = (id: string) => {
    saveTemplates(templates.filter((t) => t.id !== id))
  }

  const toggleModule = (module: string) => {
    setFormModules((prev) => (prev.includes(module) ? prev.filter((m) => m !== module) : [...prev, module]))
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="max-w-5xl mx-auto px-6 py-6">
          <h1 className="text-3xl font-bold text-foreground">Robot Central Hub</h1>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Back Button */}
        <Link href="/" className="inline-block mb-8">
          <Button variant="ghost" className="text-primary hover:bg-primary/5 gap-2 pl-0">
            <span>{"<-"}</span> Back to Overview
          </Button>
        </Link>

        {/* Page Title */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold text-foreground mb-2">Character Templates</h2>
            <p className="text-muted-foreground">Create pre-defined configurations for your robots</p>
          </div>
          {!isCreating && (
            <Button onClick={() => setIsCreating(true)} className="gap-2">
              <span>+</span> New Template
            </Button>
          )}
        </div>

        {/* Create/Edit Form */}
        {isCreating && (
          <Card className="border-primary/30 bg-primary/5 mb-8">
            <CardHeader>
              <CardTitle className="text-lg">{editingId ? "Edit Template" : "Create New Template"}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Template Name */}
              <div className="space-y-2">
                <Label htmlFor="name">Template Name</Label>
                <Input
                  id="name"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  placeholder="e.g., Kitchen Assistant"
                  className="max-w-md"
                />
              </div>

              {/* Description */}
              <div className="space-y-2">
                <Label htmlFor="description">Description (optional)</Label>
                <Input
                  id="description"
                  value={formDescription}
                  onChange={(e) => setFormDescription(e.target.value)}
                  placeholder="Brief description of this template"
                  className="max-w-md"
                />
              </div>

              {/* Role Selection */}
              <div className="space-y-2">
                <Label>Role</Label>
                <div className="flex flex-wrap gap-2">
                  {AVAILABLE_ROLES.map((role) => (
                    <button
                      key={role}
                      type="button"
                      onClick={() => setFormRole(role)}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all border ${formRole === role
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-card text-foreground border-border hover:border-primary/50"
                        }`}
                    >
                      {role.replace("_", " ")}
                    </button>
                  ))}
                </div>
              </div>

              {/* Role Prompt */}
              <div className="space-y-2">
                <Label htmlFor="rolePrompt">Role Prompt</Label>
                <p className="text-xs text-muted-foreground">
                  System prompt given to the robot to define its behavior and role context.
                </p>
                <textarea
                  id="rolePrompt"
                  value={formRolePrompt}
                  onChange={(e) => setFormRolePrompt(e.target.value)}
                  placeholder="e.g., You are a helpful kitchen assistant robot. You help users find recipes, guide them through cooking steps, and provide nutritional information."
                  rows={4}
                  className="flex w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring resize-y"
                />
              </div>

              {/* Character Selection */}
              <div className="space-y-2">
                <Label>Character (Voice)</Label>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {AVAILABLE_CHARACTERS.map((char) => (
                    <button
                      key={char.id}
                      type="button"
                      onClick={() => setFormCharacter(char.id)}
                      className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border text-left ${formCharacter === char.id
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-card text-foreground border-border hover:border-primary/50"
                        }`}
                    >
                      <span className="block">{char.name}</span>
                      <span className="block text-xs opacity-70">{char.voice}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* OCEAN Personality Traits */}
              <div className="space-y-4">
                <div>
                  <Label>OCEAN Personality Traits</Label>
                  <p className="text-xs text-muted-foreground mt-1">
                    Drag points on the chart or use sliders to set personality dimensions (0.0 - 1.0).
                  </p>
                </div>
                <div className="flex flex-col items-center gap-6">
                  <div>
                    <OceanRadarChart
                      traits={formOceanTraits}
                      onChange={setFormOceanTraits}
                      size={240}
                      color="59, 130, 246"
                    />
                  </div>
                  <div className="w-full">
                    <OceanSliders traits={formOceanTraits} onChange={setFormOceanTraits} />
                  </div>
                </div>
              </div>

              {/* Modules Selection */}
              <div className="space-y-2">
                <Label>Modules</Label>
                <div className="flex flex-wrap gap-2">
                  {AVAILABLE_MODULES.map((module) => (
                    <button
                      key={module}
                      type="button"
                      onClick={() => toggleModule(module)}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all border ${formModules.includes(module)
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-card text-foreground border-border hover:border-primary/50"
                        }`}
                    >
                      {module}
                    </button>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-3 pt-4">
                <Button onClick={handleCreateTemplate} disabled={!formName || !formRole}>
                  {editingId ? "Update Template" : "Create Template"}
                </Button>
                <Button variant="outline" onClick={resetForm}>
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Templates List */}
        {templates.length === 0 && !isCreating ? (
          <Card className="border-border">
            <CardContent className="py-12 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-muted-foreground"
                >
                  <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="12" y1="18" x2="12" y2="12" />
                  <line x1="9" y1="15" x2="15" y2="15" />
                </svg>
              </div>
              <p className="text-muted-foreground mb-4">No templates created yet</p>
              <Button onClick={() => setIsCreating(true)} className="gap-2">
                <span>+</span> Create Your First Template
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {templates.map((template) => (
              <Card key={template.id} className="border-border hover:border-primary/30 transition-all">
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="20"
                          height="20"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                          <polyline points="14 2 14 8 20 8" />
                        </svg>
                      </div>
                      <div>
                        <h3 className="font-semibold text-foreground">{template.name}</h3>
                        {template.description && (
                          <p className="text-sm text-muted-foreground">{template.description}</p>
                        )}
                      </div>
                    </div>
                    <Badge variant="outline" className="capitalize">
                      {template.role.replace("_", " ")}
                    </Badge>
                  </div>

                  {/* Character */}
                  {template.character && (
                    <div className="mb-3">
                      <p className="text-xs font-medium text-muted-foreground mb-1">Character:</p>
                      <Badge variant="secondary" className="capitalize">
                        {AVAILABLE_CHARACTERS.find((c) => c.id === template.character)?.name || template.character}
                      </Badge>
                    </div>
                  )}

                  {/* Role Prompt Preview */}
                  {template.rolePrompt && (
                    <div className="mb-3">
                      <p className="text-xs font-medium text-muted-foreground mb-1">Role Prompt:</p>
                      <p className="text-xs text-muted-foreground bg-muted/50 rounded-md p-2 line-clamp-2">
                        {template.rolePrompt}
                      </p>
                    </div>
                  )}

                  {/* OCEAN Mini Chart */}
                  {template.oceanTraits && (
                    <div className="mb-3 flex items-center gap-4">
                      <div className="shrink-0">
                        <OceanRadarChart
                          traits={template.oceanTraits}
                          disabled
                          size={120}
                          color="59, 130, 246"
                        />
                      </div>
                      <div className="text-xs text-muted-foreground space-y-0.5">
                        <p>O: {template.oceanTraits.openness.toFixed(1)}</p>
                        <p>C: {template.oceanTraits.conscientiousness.toFixed(1)}</p>
                        <p>E: {template.oceanTraits.extraversion.toFixed(1)}</p>
                        <p>A: {template.oceanTraits.agreeableness.toFixed(1)}</p>
                        <p>N: {template.oceanTraits.neuroticism.toFixed(1)}</p>
                      </div>
                    </div>
                  )}

                  {/* Modules */}
                  <div className="space-y-2 mb-4">
                    <p className="text-xs font-medium text-muted-foreground">Modules ({template.modules.length}):</p>
                    <div className="flex flex-wrap gap-2">
                      {template.modules.map((module) => (
                        <Badge
                          key={module}
                          variant="outline"
                          className="border-primary/30 text-primary bg-primary/5"
                        >
                          {module}
                        </Badge>
                      ))}
                      {template.modules.length === 0 && (
                        <span className="text-sm text-muted-foreground">No modules</span>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2 pt-4 border-t border-border">
                    <Button variant="outline" size="sm" onClick={() => handleEditTemplate(template)}>
                      Edit
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-destructive hover:bg-destructive/10 bg-transparent"
                      onClick={() => handleDeleteTemplate(template.id)}
                    >
                      Delete
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
