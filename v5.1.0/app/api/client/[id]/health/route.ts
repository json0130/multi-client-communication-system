export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id: clientId } = await params;
    const response = await fetch(`http://130.216.238.11:5000/client/${clientId}/health`, {
      headers: { "Content-Type": "application/json" },
    })

    if (!response.ok) {
      return Response.json({ error: "Failed to fetch client health" }, { status: 500 })
    }

    const data = await response.json()
    return Response.json(data)
  } catch (error) {
    console.error("API Error:", error)
    return Response.json({ error: "Failed to connect to Python server" }, { status: 500 })
  }
}
