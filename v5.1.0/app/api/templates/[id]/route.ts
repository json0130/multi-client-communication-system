import { NextResponse } from "next/server"

const PYTHON_SERVER_URL = "http://130.216.239.118:5000"

export async function DELETE(request: Request, { params }: { params: { id: string } }) {
    try {
        const response = await fetch(`${PYTHON_SERVER_URL}/templates/${params.id}`, {
            method: "DELETE",
        })

        if (!response.ok) {
            return NextResponse.json({ error: "Failed to delete template" }, { status: response.status })
        }

        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        console.error(`[v0] API Error deleting template ${params.id}:`, error)
        return NextResponse.json({ error: "Failed to connect to Python server" }, { status: 500 })
    }
}
