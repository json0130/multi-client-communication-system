"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

interface Client {
  client_id: string
  display_name: string
  robot_name: string
  role: string
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
    const interval = setInterval(fetchClients, 5000)
    return () => clearInterval(interval)
  }, [])

  const isOnline = (client: Client) => {
    return client.status === "active" && client.inactive_minutes < 5
  }

  const isUnconfigured = (client: Client) => {
    return (!client.modules || client.modules.length === 0) && (!client.role || client.role === "")
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

  const configuredClients = clients.filter((c) => !isUnconfigured(c))
  const unconfiguredClients = clients.filter((c) => isUnconfigured(c))

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <h1 className="text-3xl font-bold text-foreground">Robot Central Hub</h1>
          <Link href="/templates">
            <Button variant="outline" className="gap-2 bg-transparent">
              <span>+</span> Manage Templates
            </Button>
          </Link>
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
                {error}. Make sure the Python server is running.
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
          <>
            {/* Unconfigured Robots Section */}
            {unconfiguredClients.length > 0 && (
              <div className="mb-12">
                <div className="flex items-center gap-3 mb-4">
                  <h3 className="text-lg font-semibold text-foreground">Needs Configuration</h3>
                  <Badge variant="outline" className="bg-orange-50 text-orange-600 border-orange-200">
                    {unconfiguredClients.length} robot{unconfiguredClients.length > 1 ? "s" : ""}
                  </Badge>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {unconfiguredClients.map((client) => (
                    <Link key={client.client_id} href={`/client/${client.client_id}`} className="group">
                      <Card className="border-orange-200 bg-orange-50/50 hover:border-orange-300 transition-all hover:shadow-md cursor-pointer h-full">
                        <CardContent className="pt-6">
                          {/* Header with Icon and Status */}
                          <div className="flex items-start justify-between mb-4">
                            <div className="w-12 h-12 rounded-lg bg-orange-100 flex items-center justify-center text-orange-500 text-xl">
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                width="24"
                                height="24"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" y1="8" x2="12" y2="12" />
                                <line x1="12" y1="16" x2="12.01" y2="16" />
                              </svg>
                            </div>
                            <div className="flex flex-col items-end gap-1">
                              {getStatusBadge(client)}
                              <Badge variant="outline" className="bg-orange-100 text-orange-600 border-orange-200 text-xs">
                                Unconfigured
                              </Badge>
                            </div>
                          </div>

                          {/* Robot Name and ID */}
                          <h3 className="text-lg font-bold text-foreground mb-1">{client.robot_name || client.client_id}</h3>
                          <p className="text-sm text-muted-foreground mb-4">ID: {client.client_id}</p>

                          {/* Last Activity */}
                          <div className="flex items-center gap-1 text-sm text-muted-foreground mb-4">
                            <span>Last active: {getLastActivityText(client.inactive_minutes)}</span>
                          </div>

                          {/* Configuration Needed Message */}
                          <div className="p-3 rounded-lg bg-orange-100/50 border border-orange-200">
                            <p className="text-sm text-orange-700">
                              Click to configure this robot with a template or custom settings
                            </p>
                          </div>
                        </CardContent>
                      </Card>
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Configured Robots Section */}
            {configuredClients.length > 0 && (
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <h3 className="text-lg font-semibold text-foreground">Configured Robots</h3>
                  <Badge variant="outline" className="bg-primary/5 text-primary border-primary/20">
                    {configuredClients.length} robot{configuredClients.length > 1 ? "s" : ""}
                  </Badge>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {configuredClients.map((client) => (
                    <Link key={client.client_id} href={`/client/${client.client_id}`} className="group">
                      <Card className="border-border hover:border-primary/50 transition-all hover:shadow-md cursor-pointer h-full">
                        <CardContent className="pt-6">
                          {/* Header with Icon and Status */}
                          <div className="flex items-start justify-between mb-4">
                            <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary text-xl">
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                width="24"
                                height="24"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <rect x="3" y="11" width="18" height="10" rx="2" />
                                <circle cx="12" cy="5" r="2" />
                                <path d="M12 7v4" />
                                <line x1="8" y1="16" x2="8" y2="16" />
                                <line x1="16" y1="16" x2="16" y2="16" />
                              </svg>
                            </div>
                            {getStatusBadge(client)}
                          </div>

                          {/* Robot Name and ID */}
                          <h3 className="text-lg font-bold text-foreground mb-1">{client.robot_name}</h3>
                          <p className="text-sm text-muted-foreground mb-2">ID: {client.client_id}</p>

                          {/* Role Badge */}
                          {client.role && (
                            <Badge variant="outline" className="mb-4 capitalize">
                              {client.role.replace("_", " ")}
                            </Badge>
                          )}

                          {/* Last Activity */}
                          <div className="flex items-center gap-1 text-sm text-muted-foreground mb-4">
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
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
