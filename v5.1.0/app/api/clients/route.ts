export async function GET() {
  try {
    const response = await fetch("http://130.216.239.118:5000/clients", {
      headers: { "Content-Type": "application/json" },
    })

    if (!response.ok) {
      return Response.json({ error: "Failed to fetch from Python server" }, { status: 500 })
    }

    const data = await response.json()
    // The server returns { clients: { client_id: {...} } }
    const formattedClients = Object.entries(data.clients || {}).map(([clientId, clientData]: [string, any]) => ({
      client_id: clientId,
      display_name: clientData.display_name || clientData.robot_name || clientId,
      robot_name: clientData.robot_name || clientId,
      status: clientData.status || "inactive",
      inactive_minutes: clientData.inactive_minutes || 0,
      last_activity: clientData.last_activity || Date.now() / 1000,
      modules: clientData.modules || [],
      registration_time: clientData.registration_time || 0,
    }))

    return Response.json({
      clients: formattedClients,
      total_clients: data.total_clients || 0,
      active_servers: data.active_servers || 0,
      timestamp: data.timestamp || Date.now() / 1000,
    })
  } catch (error) {
    console.error("API Error:", error)
    return Response.json({ error: "Failed to connect to Python server" }, { status: 500 })
  }
}
