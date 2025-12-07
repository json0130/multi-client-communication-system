"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Card, CardContent } from "@/components/ui/card"
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

  const getStatusBadge = (client: Client) => {
    if (isOnline(client)) {
      return <Badge className="bg-green-100 text-green-700 hover:bg-green-100">Active</Badge>
    }
    if (client.inactive_minutes < 60) {
      return <Badge className="bg-yellow-100 text-yellow-700 hover:bg-yellow-100">Idle</Badge>
    }
    return <Badge className="bg-gray-100 text-gray-700 hover:bg-gray-100">Offline</Badge>
  }

  const getLastActivityText = (inactive_minutes: number) => {
    if (inactive_minutes < 1) return "Just now"
    if (inactive_minutes < 60) return `${Math.round(inactive_minutes)} minutes ago`
    const hoursAgo = Math.floor(inactive_minutes / 60)
    if (hoursAgo < 24) return `${hoursAgo} hours ago`
    const daysAgo = Math.floor(hoursAgo / 24)
    return `${daysAgo} days ago`
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <h1 className="text-3xl font-bold text-foreground">Robot Central Hub</h1>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Page Title */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-2">Robot Overview</h2>
          <p className="text-muted-foreground">Monitor and manage all your robots</p>
        </div>

        {/* Content */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <p className="text-muted-foreground">Loading robots...</p>
          </div>
        ) : error ? (
          <Card className="border-destructive/30 bg-destructive/5">
            <CardContent className="pt-8">
              <p className="text-destructive font-medium mb-2">Error loading robots</p>
              <p className="text-sm text-muted-foreground">
                {error}. Make sure the Python server is running at http://130.216.238.6:5000
              </p>
            </CardContent>
          </Card>
        ) : clients.length === 0 ? (
          <Card className="border-border">
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">No robots connected yet</p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {clients.map((client) => (
              <Link key={client.client_id} href={`/robot/${client.client_id}`} className="group">
                <Card className="border-border hover:border-primary/50 transition-all hover:shadow-md cursor-pointer h-full">
                  <CardContent className="pt-6">
                    {/* Header with Icon and Status */}
                    <div className="flex items-start justify-between mb-4">
                      <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary text-xl">
                        🤖
                      </div>
                      {getStatusBadge(client)}
                    </div>

                    {/* Robot Name and ID */}
                    <h3 className="text-lg font-bold text-foreground mb-1">{client.robot_name}</h3>
                    <p className="text-sm text-muted-foreground mb-4">ID: {client.client_id}</p>

                    {/* Last Activity */}
                    <div className="flex items-center gap-1 text-sm text-muted-foreground mb-4">
                      <span>⏱</span>
                      <span>Last active: {getLastActivityText(client.inactive_minutes)}</span>
                    </div>

                    {/* Modules */}
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-muted-foreground">Modules ({client.modules.length}):</p>
                      <div className="flex flex-wrap gap-2">
                        {client.modules.map((module) => (
                          <Badge
                            key={module}
                            variant="outline"
                            className="border-primary/30 text-primary bg-primary/5 hover:bg-primary/10"
                          >
                            {module}
                          </Badge>
                        ))}
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
