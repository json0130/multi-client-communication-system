import { NextResponse } from "next/server"

const PYTHON_SERVER_URL = "http://130.216.238.51:5000"

export async function DELETE(request: Request, { params }: { params: Promise<{ id: string }> }) {
    try {
        const { id } = await params;
        const response = await fetch(`${PYTHON_SERVER_URL}/templates/${id}`, {
            method: "DELETE",
        })

        if (!response.ok) {
            return NextResponse.json({ error: "Failed to delete template" }, { status: response.status })
        }

        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        console.error(`[v0] API Error deleting template:`, error)
        return NextResponse.json({ error: "Failed to connect to Python server" }, { status: 500 })
    }
}
