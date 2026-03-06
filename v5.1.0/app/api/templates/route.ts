import { NextResponse } from "next/server"

const PYTHON_SERVER_URL = "http://130.216.239.118:5000"

export async function GET() {
    try {
        const response = await fetch(`${PYTHON_SERVER_URL}/templates`, {
            headers: { "Content-Type": "application/json" },
        })

        if (!response.ok) {
            return NextResponse.json({ error: "Failed to fetch top-level templates" }, { status: response.status })
        }

        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        console.error("[v0] API Error fetching templates:", error)
        return NextResponse.json({ error: "Failed to connect to Python server" }, { status: 500 })
    }
}

export async function POST(request: Request) {
    try {
        const body = await request.json()

        // Convert 'createdAt' to integer just to be safe if it's sent as float or string
        if (body.createdAt) {
            body.createdAt = parseInt(body.createdAt, 10)
        }

        const response = await fetch(`${PYTHON_SERVER_URL}/templates`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        })

        if (!response.ok) {
            return NextResponse.json({ error: "Failed to create/update template" }, { status: response.status })
        }

        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        console.error("[v0] API Error fetching templates:", error)
        return NextResponse.json({ error: "Failed to connect to Python server" }, { status: 500 })
    }
}
