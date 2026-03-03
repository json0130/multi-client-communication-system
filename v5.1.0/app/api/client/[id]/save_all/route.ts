import { type NextRequest, NextResponse } from "next/server"

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: clientId } = await params

  try {
    const body = await request.json()
    const { robot_name, robot_role, role_prompt, character, modules, ocean_traits } = body

    // Send all changes to Python server in a single PATCH request
    const response = await fetch(`http://130.216.239.118:5000/client/${clientId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        robot_name,
        robot_role,
        role_prompt,
        character,
        modules,
        ocean_traits,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      return Response.json({ error: errorData.error || "Failed to save changes" }, { status: response.status })
    }

    const result = await response.json()
    return Response.json({ success: true, message: "All changes saved", data: result }, { status: 200 })
  } catch (error) {
    console.error("Error saving changes:", error)
    return Response.json({ error: "Failed to save changes" }, { status: 500 })
  }
}
