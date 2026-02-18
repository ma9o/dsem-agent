"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { formatNumber } from "@/lib/utils/format";
import type { Extraction, LikelihoodSpec } from "@causal-ssm/api-types";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

interface ObservationModelCardProps {
  likelihood: LikelihoodSpec;
  extractions: Extraction[];
  priorSamples?: number[];
}

/** Bin numeric values into a histogram. */
function buildHistogram(
  values: number[],
  nBins: number,
): Array<{ binCenter: number; count: number }> {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / nBins;

  const bins = Array.from({ length: nBins }, (_, i) => ({
    binCenter: Math.round((min + (i + 0.5) * binWidth) * 100) / 100,
    count: 0,
  }));

  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), nBins - 1);
    bins[idx].count++;
  }

  return bins;
}

/** Build count-frequency table for discrete data. */
function buildCountFrequency(values: number[]): Array<{ binCenter: number; count: number }> {
  const freq = new Map<number, number>();
  for (const v of values) {
    freq.set(v, (freq.get(v) ?? 0) + 1);
  }
  return Array.from(freq.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([k, count]) => ({ binCenter: k, count }));
}

/**
 * Bin prior predictive samples into the same bin structure as the data histogram,
 * scaled so the total area matches the data histogram (n_data counts).
 */
function binPriorSamples(
  priorSamples: number[],
  dataBins: Array<{ binCenter: number }>,
  nData: number,
  isDiscrete: boolean,
): Array<{ binCenter: number; prior: number }> {
  if (priorSamples.length === 0 || dataBins.length === 0) return [];

  if (isDiscrete) {
    // Count frequency at each discrete value
    const freq = new Map<number, number>();
    for (const v of priorSamples) {
      freq.set(v, (freq.get(v) ?? 0) + 1);
    }
    // Scale so total matches nData
    const scale = nData / priorSamples.length;
    return dataBins.map((b) => ({
      binCenter: b.binCenter,
      prior: (freq.get(b.binCenter) ?? 0) * scale,
    }));
  }

  // Continuous: use same bin edges as data histogram
  if (dataBins.length < 2) return [];
  const binWidth = dataBins[1].binCenter - dataBins[0].binCenter;
  const firstEdge = dataBins[0].binCenter - binWidth / 2;

  const counts = new Array(dataBins.length).fill(0);
  for (const v of priorSamples) {
    const idx = Math.min(Math.max(Math.floor((v - firstEdge) / binWidth), 0), dataBins.length - 1);
    counts[idx]++;
  }

  // Scale so total count matches nData
  const scale = nData / priorSamples.length;
  return dataBins.map((b, i) => ({
    binCenter: b.binCenter,
    prior: counts[i] * scale,
  }));
}

/** Format link function for display. */
function linkLabel(link: string): string {
  switch (link) {
    case "identity":
      return "E[y] = \u03BC";
    case "log":
      return "E[y] = exp(\u03BC)";
    case "logit":
      return "E[y] = \u03C3(\u03BC)";
    case "probit":
      return "E[y] = \u03A6(\u03BC)";
    default:
      return "g\u207B\u00B9(\u03BC)";
  }
}

export function ObservationModelCard({
  likelihood,
  extractions,
  priorSamples,
}: ObservationModelCardProps) {
  const numericValues = extractions
    .map((e) => (typeof e.value === "boolean" ? (e.value ? 1 : 0) : Number(e.value)))
    .filter((v) => !Number.isNaN(v));

  if (numericValues.length === 0) return null;

  const isDiscrete =
    likelihood.distribution === "poisson" || likelihood.distribution === "bernoulli";

  const bins = isDiscrete
    ? buildCountFrequency(numericValues)
    : buildHistogram(numericValues, Math.min(15, Math.ceil(Math.sqrt(numericValues.length))));

  const prior =
    priorSamples && priorSamples.length > 0
      ? binPriorSamples(priorSamples, bins, numericValues.length, isDiscrete)
      : [];

  const hasPrior = prior.length > 0;

  const chartData = bins.map((b) => {
    const p = prior.find((o) => o.binCenter === b.binCenter);
    return { ...b, ...(hasPrior ? { prior: p?.prior ?? 0 } : {}) };
  });

  const mean = numericValues.reduce((s, v) => s + v, 0) / numericValues.length;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono">{likelihood.variable}</CardTitle>
          <div className="flex items-center gap-1.5">
            <Badge variant="outline">{likelihood.distribution}</Badge>
            <Badge variant="secondary">{linkLabel(likelihood.link)}</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-36 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="binCenter"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v: number) => formatNumber(v, 1)}
                tick={{ fontSize: 10 }}
              />
              <YAxis tick={{ fontSize: 10 }} />
              <Bar
                dataKey="count"
                fill="var(--muted-foreground)"
                opacity={0.3}
                barSize={isDiscrete ? 20 : undefined}
              />
              {hasPrior && (
                <Line
                  type="monotone"
                  dataKey="prior"
                  stroke="var(--primary)"
                  strokeWidth={2}
                  dot={false}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>
            n={numericValues.length} &middot; mean={formatNumber(mean, 2)}
          </span>
          {hasPrior && (
            <span className="inline-flex items-center gap-2">
              <span className="inline-flex items-center gap-1">
                <span className="inline-block h-0.5 w-3 bg-primary" />
                prior pred.
              </span>
              <StatTooltip explanation="Marginal prior predictive — observations simulated from the model's priors before seeing data. Compare against the empirical histogram to check whether the priors imply a plausible data scale." />
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
