"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface Client {
  client_id: string
  display_name: string
  robot_name: string
  status: string
  inactive_minutes: number
  last_activity: number
  modules: string[]
  registration_time: number
}

export default function RobotOverviewPage() {
  const [clients, setClients] = useState<Client[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState({ total_clients: 0, active_servers: 0 })

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
        setStats({
          total_clients: data.total_clients || 0,
          active_servers: data.active_servers || 0,
        })
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load clients")
      } finally {
        setLoading(false)
      }
    }

    fetchClients()
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchClients, 5000)
    return () => clearInterval(interval)
  }, [])

  const isOnline = (client: Client) => {
    return client.status === "active" && client.inactive_minutes < 1
  }

  const getStatusColor = (client: Client) => {
    if (isOnline(client)) return "bg-green-500/20 text-green-300"
    return "bg-red-500/20 text-red-300"
  }

  const getStatusDot = (client: Client) => {
    if (isOnline(client)) return "bg-green-500"
    return "bg-red-500"
  }

  const getLastActivityText = (inactive_minutes: number) => {
    if (inactive_minutes < 1) return "Just now"
    if (inactive_minutes < 60) return `${Math.round(inactive_minutes)}m ago`
    const hoursAgo = Math.floor(inactive_minutes / 60)
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
              <div className="text-2xl font-bold text-foreground">{stats.total_clients}</div>
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Online</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-accent">{clients.filter(isOnline).length}</div>
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Active Servers</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-accent">{stats.active_servers}</div>
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
                Make sure the Python server is running at http://130.216.238.6:5000
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
            {clients.map((client) => (
              <Link key={client.client_id} href={`/robot/${client.client_id}`}>
                <Card className="bg-card border-border hover:border-primary/50 transition-colors cursor-pointer">
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          <div className={`w-3 h-3 rounded-full ${getStatusDot(client)}`} />
                          <h3 className="text-lg font-semibold text-foreground">{client.robot_name}</h3>
                          <Badge className={getStatusColor(client)}>{isOnline(client) ? "Online" : "Offline"}</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-3">ID: {client.client_id}</p>
                        <div className="flex flex-wrap gap-2">
                          {client.modules.map((module) => (
                            <Badge key={module} variant="outline" className="border-primary/30 text-accent">
                              {module}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-muted-foreground">Last activity</p>
                        <p className="text-sm text-foreground">{getLastActivityText(client.inactive_minutes)}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
