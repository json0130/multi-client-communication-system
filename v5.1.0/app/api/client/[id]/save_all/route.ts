// app/api/client/[id]/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: clientId } = await params;

  try {
    const body = await request.json();

    // Destructure everything the frontend now sends
    const { robot_name, robot_role, character, modules } = body;

    // Forward ALL fields to the Python backend
    const response = await fetch(`http://130.216.239.118:5000/client/${clientId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        robot_name,
        robot_role,
        character,           // ← NEW
        modules,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: errorData.error || "Failed to save changes" },
        { status: response.status }
      );
    }

    const result = await response.json();
    return NextResponse.json(
      { success: true, message: "All changes saved", data: result },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error saving changes:", error);
    return NextResponse.json(
      { error: "Failed to save changes" },
      { status: 500 }
    );
  }
}