// app/api/client/[id]/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }  // Add Promise type
) {
  const { id: clientId } = await params;  // Await the params Promise and destructure

  try {
    const body = await request.json();
    const { robot_name, robot_role, modules } = body;

    // Send all changes to Python server in a single PATCH request
    const response = await fetch(`http://130.216.238.11:5000/client/${clientId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        robot_name,
        robot_role,
        modules,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json({ error: errorData.error || "Failed to save changes" }, { status: response.status });
    }

    const result = await response.json();
    return NextResponse.json({ success: true, message: "All changes saved", data: result }, { status: 200 });
  } catch (error) {
    console.error("Error saving changes:", error);
    return NextResponse.json({ error: "Failed to save changes" }, { status: 500 });
  }
}