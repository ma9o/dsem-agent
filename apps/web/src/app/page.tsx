"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { uploadFile } from "@/lib/api/endpoints";
import { MOCK_RUN_ID, isMockMode } from "@/lib/api/mock-provider";
import { getDeploymentId, triggerRun } from "@/lib/api/prefect";
import { ArrowRight, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";

export default function LandingPage() {
  const router = useRouter();
  const [question, setQuestion] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }, []);

  const handleSubmit = async () => {
    if (!question.trim()) {
      setError("Please enter a research question.");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      if (isMockMode()) {
        router.push(`/analysis/${MOCK_RUN_ID}`);
        return;
      }

      const userId = `user-${Date.now()}`;

      if (file) {
        await uploadFile(file, userId);
      }

      const deploymentId = await getDeploymentId();
      const runId = await triggerRun(deploymentId, {
        query: question,
        user_id: userId,
      });

      router.push(`/analysis/${runId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start analysis");
      setIsSubmitting(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="w-full max-w-2xl space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Causal Inference Pipeline</h1>
          <p className="text-lg text-muted-foreground">
            From research question to quantified treatment effects
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Research Question</CardTitle>
            <CardDescription>What causal relationship do you want to investigate?</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <textarea
              className="w-full rounded-md border bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-h-[100px] resize-y"
              placeholder="e.g., How does my daily screen time affect my sleep quality and mood?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Data Upload</CardTitle>
            <CardDescription>Upload your Google Takeout export (ZIP or JSON)</CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
                dragOver
                  ? "border-primary bg-primary/5"
                  : "border-muted-foreground/25 hover:border-muted-foreground/50"
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <Upload className="h-8 w-8 text-muted-foreground mb-2" />
              {file ? (
                <p className="text-sm font-medium">{file.name}</p>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Drag and drop or{" "}
                  <button
                    type="button"
                    className="text-primary underline"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    browse
                  </button>
                </p>
              )}
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept=".zip,.json"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setFile(f);
                }}
              />
            </div>
          </CardContent>
        </Card>

        {error && <p className="text-sm text-destructive text-center">{error}</p>}

        <Button
          className="w-full"
          size="lg"
          onClick={handleSubmit}
          disabled={isSubmitting || !question.trim()}
        >
          {isSubmitting ? "Starting Analysis..." : "Start Analysis"}
          <ArrowRight className="h-4 w-4 ml-2" />
        </Button>
      </div>
    </main>
  );
}
