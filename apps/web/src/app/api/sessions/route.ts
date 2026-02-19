import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

const SESSIONS_PATH = join(process.cwd(), "..", "data-pipeline", "results", "sessions.json");

interface Session {
  runId: string;
  question: string;
  createdAt: string;
}

async function readSessions(): Promise<Record<string, Session>> {
  try {
    const data = await readFile(SESSIONS_PATH, "utf-8");
    return JSON.parse(data);
  } catch {
    return {};
  }
}

export async function POST(request: Request) {
  const body = await request.json();
  const { code, runId, question } = body as {
    code?: string;
    runId?: string;
    question?: string;
  };

  if (!code || !runId || !question) {
    return NextResponse.json(
      { error: "code, runId, and question are required" },
      { status: 400 },
    );
  }

  const normalizedCode = code.toUpperCase();
  const sessions = await readSessions();
  sessions[normalizedCode] = {
    runId,
    question,
    createdAt: new Date().toISOString(),
  };

  await writeFile(SESSIONS_PATH, JSON.stringify(sessions, null, 2));
  return NextResponse.json({ ok: true });
}
