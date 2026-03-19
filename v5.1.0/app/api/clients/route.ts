export async function GET() {
  try {
    const response = await fetch("http://130.216.238.51:5000/clients", {
      headers: { "Content-Type": "application/json" },
    })

    if (!response.ok) {
      return Response.json({ error: "Failed to fetch from Python server" }, { status: 500 })
    }

    const text = await response.text()
    console.log("[v0] Raw response from Python server:", text.substring(0, 200))

    let data
    try {
      data = JSON.parse(text)
    } catch (parseError) {
      console.error("[v0] Failed to parse response as JSON:", text.substring(0, 500))
      return Response.json(
        { error: "Python server returned invalid JSON. Check if /clients endpoint exists on the Python server." },
        { status: 500 },
      )
    }

    // The server returns { clients: { client_id: {...} } }
    const formattedClients = Object.entries(data.clients || {}).map(([clientId, clientData]: [string, any]) => ({
      client_id: clientId,
      display_name: clientData.display_name || clientData.robot_name || clientId,
      robot_name: clientData.robot_name || clientId,
      role: clientData.role || clientData.robot_role || "",
      rolePrompt: clientData.role_prompt || "",
      character: clientData.character || "",
      status: clientData.status || "inactive",
      inactive_minutes: clientData.inactive_minutes || 0,
      last_activity: clientData.last_activity || Date.now() / 1000,
      modules: clientData.modules || [],
      oceanTraits: clientData.ocean_traits || null,
      registration_time: clientData.registration_time || 0,
    }))

    return Response.json({
      clients: formattedClients,
      total_clients: data.total_clients || 0,
      active_servers: data.active_servers || 0,
      timestamp: data.timestamp || Date.now() / 1000,
    })
  } catch (error) {
    console.error("[v0] API Error:", error)
    return Response.json({ error: "Failed to connect to Python server" }, { status: 500 })
  }
}
