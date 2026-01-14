export async function POST(req: Request, { params }: { params: { id: string } }) {
  try {
    const clientId = params.id
    const body = await req.json()

    const response = await fetch(`http://130.216.238.11:5000/client/${clientId}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ character: body.character }),
    })

    if (!response.ok) {
      return Response.json({ error: "Failed to update character" }, { status: 500 })
    }

    const data = await response.json()
    return Response.json(data)
  } catch (error) {
    console.error("API Error:", error)
    return Response.json({ error: "Failed to connect to Python server" }, { status: 500 })
  }
}
