"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useStageData } from "@/lib/hooks/use-stage-data";
import { formatNumber } from "@/lib/utils/format";
import type { Stage5Data } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Download, TrendingUp } from "lucide-react";
import { useCallback } from "react";

export function CompletionSummary({ runId }: { runId: string }) {
  const { data } = useStageData<Stage5Data>(runId, "stage-5", true);
  const queryClient = useQueryClient();

  const handleExport = useCallback(() => {
    const allData: Record<string, unknown> = {};
    for (const stage of STAGES) {
      const stageData = queryClient.getQueryData(["pipeline", runId, "stage", stage.id]);
      if (stageData) {
        allData[stage.id] = stageData;
      }
    }
    const blob = new Blob([JSON.stringify(allData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analysis-${runId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [runId, queryClient]);

  if (!data) return null;

  const results = data.intervention_results;
  const sorted = [...results].sort((a, b) => Math.abs(b.beta_hat) - Math.abs(a.beta_hat));
  const top = sorted[0];
  const identifiableCount = results.filter((r) => r.identifiable).length;
  const sensitiveCount = results.filter((r) => r.sensitivity_flag).length;

  return (
    <Card className="animate-fade-in-up border-success/30 bg-success-soft/50">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <CheckCircle2 className="h-5 w-5 text-success" />
            Analysis Complete
          </CardTitle>
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="mr-1.5 h-3.5 w-3.5" />
            Export JSON
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {top && (
          <div className="flex items-start gap-2">
            <TrendingUp className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <p className="text-sm">
              <span className="font-medium">Strongest effect:</span>{" "}
              <span className="font-mono">{top.treatment}</span> with{" "}
              <span className="font-mono">
                {"\u03B2\u0302"} = {formatNumber(top.beta_hat)}
              </span>{" "}
              <span className="text-muted-foreground">
                [{formatNumber(top.ci_lower)}, {formatNumber(top.ci_upper)}]
              </span>
            </p>
          </div>
        )}
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">
            {results.length} treatment{results.length !== 1 && "s"} analyzed
          </Badge>
          <Badge variant={identifiableCount === results.length ? "success" : "warning"}>
            {identifiableCount}/{results.length} identifiable
          </Badge>
          {sensitiveCount > 0 && (
            <Badge variant="warning">
              {sensitiveCount} sensitivity warning{sensitiveCount !== 1 && "s"}
            </Badge>
          )}
          {data.ppc.overall_passed ? (
            <Badge variant="success">PPC passed</Badge>
          ) : (
            <Badge variant="destructive">PPC failed</Badge>
          )}
        </div>
        {data.inference_metadata && (
          <div className="flex flex-wrap gap-2 pt-1 border-t">
            <Badge variant="secondary">{data.inference_metadata.method}</Badge>
            <Badge variant="secondary">
              {data.inference_metadata.n_samples.toLocaleString()} samples
            </Badge>
            <Badge variant="secondary">{formatNumber(data.inference_metadata.duration_seconds)}s</Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
