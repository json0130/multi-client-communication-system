export async function GET() {
  try {
    const response = await fetch("http://130.216.239.118:5000/api/robots", {
      headers: { "Content-Type": "application/json" },
    })

    if (!response.ok) {
      return Response.json({ error: "Failed to fetch from Python server" }, { status: 500 })
    }

    const data = await response.json()
    return Response.json(data)
  } catch (error) {
    console.error("API Error:", error)
    return Response.json({ error: "Failed to connect to Python server" }, { status: 500 })
  }
}
