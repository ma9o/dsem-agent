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

/** MoM fit: fit distribution to data via method of moments. */
function momOverlay(
  dist: string,
  values: number[],
  bins: Array<{ binCenter: number }>,
  n: number,
): Array<{ binCenter: number; mom: number }> {
  if (values.length < 2) return [];

  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1);
  const std = Math.sqrt(variance);

  switch (dist) {
    case "gaussian": {
      const binWidth = bins.length > 1 ? bins[1].binCenter - bins[0].binCenter : 1;
      return bins.map((b) => ({
        binCenter: b.binCenter,
        mom: normalPdf(b.binCenter, mean, std) * n * binWidth,
      }));
    }
    case "poisson": {
      return bins.map((b) => ({
        binCenter: b.binCenter,
        mom: poissonPmf(b.binCenter, mean) * n,
      }));
    }
    case "bernoulli": {
      return [
        { binCenter: 0, mom: (1 - mean) * n },
        { binCenter: 1, mom: mean * n },
      ];
    }
    case "beta": {
      const m = Math.max(0.01, Math.min(0.99, mean));
      const v = Math.min(variance, m * (1 - m) - 0.001);
      const alpha = m * ((m * (1 - m)) / Math.max(v, 0.001) - 1);
      const beta = (1 - m) * ((m * (1 - m)) / Math.max(v, 0.001) - 1);
      const binWidth = bins.length > 1 ? bins[1].binCenter - bins[0].binCenter : 0.1;
      return bins.map((b) => ({
        binCenter: b.binCenter,
        mom: betaPdf(b.binCenter, Math.max(alpha, 0.1), Math.max(beta, 0.1)) * n * binWidth,
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

  const mom = momOverlay(likelihood.distribution, numericValues, bins, numericValues.length);

  const chartData = bins.map((b) => {
    const m = mom.find((o) => o.binCenter === b.binCenter);
    return { ...b, mom: m?.mom ?? 0 };
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
                dataKey="mom"
                stroke="var(--primary)"
                strokeWidth={1.5}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>
            n={numericValues.length} &middot; mean={formatNumber(mean, 2)}
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-0.5 w-3 bg-primary" />
              MoM fit
            </span>
            <StatTooltip explanation="Method of moments fit: best-fit distribution of the chosen family to the observed data. Shows whether the distribution family is a good match for the data shape." />
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
