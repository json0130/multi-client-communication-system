export async function POST(request: Request, { params }: { params: { id: string } }) {
  const clientId = params.id

  try {
    const body = await request.json()
    const { modules, role, character } = body

    // Send all changes to Python server
    // You may want to adjust these endpoints based on your actual server implementation
    const promises = []

    // Update modules
    if (modules) {
      for (const [moduleName, isEnabled] of Object.entries(modules)) {
        promises.push(
          fetch(`http://130.216.239.248:5000/client/${clientId}/modules`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              module: moduleName,
              enabled: isEnabled,
            }),
          }),
        )
      }
    }

    // Update role
    if (role) {
      promises.push(
        fetch(`http://130.216.239.248:5000/client/${clientId}/role`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ role }),
        }),
      )
    }

    // Update character
    if (character) {
      promises.push(
        fetch(`http://130.216.239.248:5000/client/${clientId}/character`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ character }),
        }),
      )
    }

    // Wait for all requests to complete
    const responses = await Promise.all(promises)

    // Check if any request failed
    const allSuccessful = responses.every((res) => res.ok)

    if (!allSuccessful) {
      return Response.json({ error: "Some changes failed to save" }, { status: 400 })
    }

    return Response.json({ success: true, message: "All changes saved" }, { status: 200 })
  } catch (error) {
    console.error("Error saving changes:", error)
    return Response.json({ error: "Failed to save changes" }, { status: 500 })
  }
}
