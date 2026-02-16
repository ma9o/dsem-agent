"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { evaluatePdf } from "@/lib/utils/distributions";
import { formatNumber, formatPercent } from "@/lib/utils/format";
import type { PriorProposal } from "@causal-ssm/api-types";
import { ExternalLink } from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";

interface PriorCardProps {
  prior: PriorProposal;
}

function confidenceVariant(
  prior: PriorProposal,
): "success" | "warning" | "destructive" | "secondary" {
  if (prior.confidence_level) {
    return prior.confidence_level === "high"
      ? "success"
      : prior.confidence_level === "medium"
        ? "warning"
        : "destructive";
  }
  // No pipeline-provided level — show neutral badge
  return "secondary";
}

export function PriorCard({ prior }: PriorCardProps) {
  // Prefer pipeline-provided density points; fall back to client-side approximation
  const pdfData = prior.density_points ?? evaluatePdf(prior.distribution, prior.params);
  const isApproximate = !prior.density_points;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-mono">{prior.parameter}</CardTitle>
            <CardDescription>{prior.distribution}</CardDescription>
          </div>
          <Badge variant={confidenceVariant(prior)}>
            Confidence: {formatPercent(prior.confidence)}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* PDF Chart */}
        <div className="relative h-40 w-full">
          {isApproximate && (
            <span className="absolute top-1 right-2 z-10 rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
              approx.
            </span>
          )}
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={pdfData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="x"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v: number) => formatNumber(v, 2)}
                tick={{ fontSize: 11 }}
              />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => formatNumber(v, 2)} />
              <Line
                type="monotone"
                dataKey="y"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Distribution params as pills */}
        <div className="flex flex-wrap gap-2">
          {Object.entries(prior.params).map(([key, value]) => (
            <Badge key={key} variant="secondary">
              {key} = {formatNumber(value)}
            </Badge>
          ))}
        </div>

        {/* Reasoning */}
        <p className="text-sm text-muted-foreground">{prior.reasoning}</p>

        {/* Sources */}
        {prior.sources.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-medium">Sources</p>
            <div className="space-y-2">
              {prior.sources.map((source, i) => (
                <div
                  key={`source-${
                    // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                    i
                  }`}
                  className="rounded-md border p-3 text-sm"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{source.title}</span>
                    {source.url && (
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary hover:underline"
                      >
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                  </div>
                  <p className="mt-1 text-muted-foreground">{source.snippet}</p>
                  {source.effect_size && (
                    <Badge variant="outline" className="mt-1">
                      Effect size: {source.effect_size}
                    </Badge>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
