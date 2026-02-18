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

/** Poisson PMF. */
function poissonPmf(k: number, lambda: number): number {
  if (k < 0 || !Number.isInteger(k)) return 0;
  let logP = -lambda + k * Math.log(lambda);
  for (let i = 2; i <= k; i++) logP -= Math.log(i);
  return Math.exp(logP);
}

/** Normal PDF. */
function normalPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

/** Beta PDF (using Stirling approx for log-gamma). */
function betaPdf(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) return 0;
  const lnGamma = (v: number) =>
    0.5 * Math.log((2 * Math.PI) / v) + v * (Math.log(v + 1 / (12 * v - 1 / (10 * v))) - 1);
  const logB = lnGamma(a) + lnGamma(b) - lnGamma(a + b);
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
}

/** Fit distribution to data via method of moments & return overlay points. */
function fittedOverlay(
  dist: string,
  values: number[],
  bins: Array<{ binCenter: number }>,
  n: number,
): Array<{ binCenter: number; fitted: number }> {
  if (values.length < 2) return [];

  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1);
  const std = Math.sqrt(variance);

  switch (dist) {
    case "gaussian": {
      // For continuous: scale PDF to match histogram counts
      const binWidth = bins.length > 1 ? bins[1].binCenter - bins[0].binCenter : 1;
      return bins.map((b) => ({
        binCenter: b.binCenter,
        fitted: normalPdf(b.binCenter, mean, std) * n * binWidth,
      }));
    }
    case "poisson": {
      const lambda = mean;
      return bins.map((b) => ({
        binCenter: b.binCenter,
        fitted: poissonPmf(b.binCenter, lambda) * n,
      }));
    }
    case "bernoulli": {
      const p = mean;
      return [
        { binCenter: 0, fitted: (1 - p) * n },
        { binCenter: 1, fitted: p * n },
      ];
    }
    case "beta": {
      // Method of moments for Beta
      const m = Math.max(0.01, Math.min(0.99, mean));
      const v = Math.min(variance, m * (1 - m) - 0.001);
      const alpha = m * ((m * (1 - m)) / Math.max(v, 0.001) - 1);
      const beta = (1 - m) * ((m * (1 - m)) / Math.max(v, 0.001) - 1);
      const binWidth = bins.length > 1 ? bins[1].binCenter - bins[0].binCenter : 0.1;
      return bins.map((b) => ({
        binCenter: b.binCenter,
        fitted: betaPdf(b.binCenter, Math.max(alpha, 0.1), Math.max(beta, 0.1)) * n * binWidth,
      }));
    }
    default:
      return [];
  }
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

export function ObservationModelCard({ likelihood, extractions }: ObservationModelCardProps) {
  const numericValues = extractions
    .map((e) => (typeof e.value === "boolean" ? (e.value ? 1 : 0) : Number(e.value)))
    .filter((v) => !Number.isNaN(v));

  if (numericValues.length === 0) return null;

  const isDiscrete =
    likelihood.distribution === "poisson" || likelihood.distribution === "bernoulli";

  const bins = isDiscrete
    ? buildCountFrequency(numericValues)
    : buildHistogram(numericValues, Math.min(15, Math.ceil(Math.sqrt(numericValues.length))));

  const overlay = fittedOverlay(likelihood.distribution, numericValues, bins, numericValues.length);

  // Merge bins and overlay
  const chartData = bins.map((b) => {
    const fit = overlay.find((o) => o.binCenter === b.binCenter);
    return { ...b, fitted: fit?.fitted ?? 0 };
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
              <Line
                type="monotone"
                dataKey="fitted"
                stroke="var(--primary)"
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>
            n={numericValues.length} &middot; mean={formatNumber(mean, 2)}
          </span>
          <span className="inline-flex items-center gap-1">
            MoM fit
            <StatTooltip explanation="The curve shows the chosen distribution family fitted to the observed data via method of moments. This helps verify the distribution choice — if the curve doesn't match the histogram shape, the family may be inappropriate." />
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
