export async function GET() {
  try {
    const response = await fetch("http://localhost:5000/api/registry", {
      headers: { "Content-Type": "application/json" },
    })

    if (!response.ok) {
      return Response.json({ error: "Failed to fetch from Python server" }, { status: 500 })
    }

    const data = await response.json()
    // Format the response to match frontend expectations
    return Response.json({
      clients: Object.values(data).map((client: any) => ({
        client_id: client.client_id,
        robot_name: client.robot_name || client.client_id,
        enabled_modules: client.enabled_modules || [],
        last_activity: client.last_activity || Date.now() / 1000,
        components_initialized: client.components_initialized || false,
        current_emotion: client.current_emotion || "neutral",
      })),
    })
  } catch (error) {
    console.error("API Error:", error)
    return Response.json({ error: "Failed to connect to Python server" }, { status: 500 })
  }
}
