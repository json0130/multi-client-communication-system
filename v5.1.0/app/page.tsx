"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface Client {
  client_id: string
  robot_name: string
  enabled_modules: string[]
  last_activity: number
  components_initialized: boolean
  current_emotion: string
}

export default function RobotOverviewPage() {
  const [clients, setClients] = useState<Client[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchClients = async () => {
      try {
        const response = await fetch("/api/clients", {
          headers: {
            "Content-Type": "application/json",
          },
        })

        if (!response.ok) {
          throw new Error("Failed to fetch clients")
        }

        const data = await response.json()
        setClients(data.clients || [])
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load clients")
        console.error("[v0] Error fetching clients:", err)
      } finally {
        setLoading(false)
      }
    }

    fetchClients()
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchClients, 5000)
    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (initialized: boolean, lastActivity: number) => {
    const currentTime = Date.now() / 1000
    const timeSinceActivity = currentTime - lastActivity
    const isOnline = initialized && timeSinceActivity < 30

    if (isOnline) return "bg-green-500/20 text-green-300"
    return "bg-red-500/20 text-red-300"
  }

  const getStatusDot = (initialized: boolean, lastActivity: number) => {
    const currentTime = Date.now() / 1000
    const timeSinceActivity = currentTime - lastActivity
    const isOnline = initialized && timeSinceActivity < 30

    if (isOnline) return "bg-green-500"
    return "bg-red-500"
  }

  const getLastActivityText = (lastActivity: number) => {
    const currentTime = Date.now() / 1000
    const secondsAgo = Math.round(currentTime - lastActivity)

    if (secondsAgo < 60) return "Just now"
    const minutesAgo = Math.floor(secondsAgo / 60)
    if (minutesAgo < 60) return `${minutesAgo}m ago`
    const hoursAgo = Math.floor(minutesAgo / 60)
    return `${hoursAgo}h ago`
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 py-12">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-2">Robot Central Hub</h1>
          <p className="text-muted-foreground">Monitor and manage all connected robots</p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-12">
          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Robots</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{clients.length}</div>
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Online</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-accent">
                {
                  clients.filter((c) => {
                    const currentTime = Date.now() / 1000
                    return c.components_initialized && currentTime - c.last_activity < 30
                  }).length
                }
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Offline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-destructive">
                {
                  clients.filter((c) => {
                    const currentTime = Date.now() / 1000
                    return !c.components_initialized || currentTime - c.last_activity >= 30
                  }).length
                }
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Client List */}
        {loading ? (
          <Card className="bg-card border-border">
            <CardContent className="pt-8 text-center">
              <p className="text-muted-foreground">Loading robots...</p>
            </CardContent>
          </Card>
        ) : error ? (
          <Card className="bg-card border-border border-destructive/50">
            <CardContent className="pt-8">
              <p className="text-destructive">Error: {error}</p>
              <p className="text-sm text-muted-foreground mt-2">
                Make sure the Python server is running at http://localhost:5000
              </p>
            </CardContent>
          </Card>
        ) : clients.length === 0 ? (
          <Card className="bg-card border-border">
            <CardContent className="pt-8 text-center">
              <p className="text-muted-foreground">No robots connected</p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {clients.map((client) => {
              const currentTime = Date.now() / 1000
              const isOnline = client.components_initialized && currentTime - client.last_activity < 30

              return (
                <Link key={client.client_id} href={`/robot/${client.client_id}`}>
                  <Card className="bg-card border-border hover:border-primary/50 transition-colors cursor-pointer">
                    <CardContent className="pt-6">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-3">
                            <div
                              className={`w-3 h-3 rounded-full ${getStatusDot(client.components_initialized, client.last_activity)}`}
                            />
                            <h3 className="text-lg font-semibold text-foreground">{client.robot_name}</h3>
                            <Badge className={getStatusColor(client.components_initialized, client.last_activity)}>
                              {isOnline ? "Online" : "Offline"}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mb-3">ID: {client.client_id}</p>
                          <div className="flex flex-wrap gap-2">
                            {client.enabled_modules.map((module) => (
                              <Badge key={module} variant="outline" className="border-primary/30 text-accent">
                                {module}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-xs text-muted-foreground">Last activity</p>
                          <p className="text-sm text-foreground">{getLastActivityText(client.last_activity)}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
