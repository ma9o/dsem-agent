import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

const SESSIONS_PATH = join(process.cwd(), "..", "data-pipeline", "results", "sessions.json");

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ code: string }> },
) {
  const { code } = await params;
  const normalizedCode = code.toUpperCase();

  try {
    const data = await readFile(SESSIONS_PATH, "utf-8");
    const sessions = JSON.parse(data);
    const session = sessions[normalizedCode];

    if (!session) {
      return NextResponse.json({ error: "Session not found" }, { status: 404 });
    }

    return NextResponse.json(session);
  } catch {
    return NextResponse.json({ error: "Session not found" }, { status: 404 });
  }
}
