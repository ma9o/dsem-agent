import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string; stage: string }> },
) {
  const { runId, stage } = await params;

  // Try real pipeline results first, fall back to fixtures (for mock/dev mode)
  const paths = [
    join(process.cwd(), "..", "data-pipeline", "results", runId, `${stage}.json`),
    join(process.cwd(), "test", "fixtures", `${stage}.json`),
  ];

  for (const filePath of paths) {
    try {
      const data = await readFile(filePath, "utf-8");
      return NextResponse.json(JSON.parse(data));
    } catch {
      // Try next path
    }
  }

  return NextResponse.json({ error: `No data for ${stage}` }, { status: 404 });
}
