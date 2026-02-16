"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber } from "@/lib/utils/format";
import type { TreatmentEffect } from "@causal-ssm/api-types";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

interface ForestPlotPanelProps {
  results: TreatmentEffect[];
}

interface ForestDatum {
  treatment: string;
  beta_hat: number;
  ci_lower: number;
  ci_upper: number;
  /** Offset from ci_lower to beta_hat for stacked rendering */
  lower_to_point: number;
  /** Offset from beta_hat to ci_upper for stacked rendering */
  point_to_upper: number;
  /** Invisible base segment from data min to ci_lower */
  base: number;
}

export function ForestPlotPanel({ results }: ForestPlotPanelProps) {
  if (results.length === 0) return null;

  // Sort by effect size descending
  const sorted = [...results].sort((a, b) => b.beta_hat - a.beta_hat);

  // Compute global min/max for axis domain
  const allLower = sorted.map((r) => r.ci_lower);
  const allUpper = sorted.map((r) => r.ci_upper);
  const domainMin = Math.min(...allLower, 0) * 1.1;
  const domainMax = Math.max(...allUpper, 0) * 1.1;

  const data: ForestDatum[] = sorted.map((r) => ({
    treatment: r.treatment,
    beta_hat: r.beta_hat,
    ci_lower: r.ci_lower,
    ci_upper: r.ci_upper,
    base: r.ci_lower - domainMin,
    lower_to_point: r.beta_hat - r.ci_lower,
    point_to_upper: r.ci_upper - r.beta_hat,
  }));

  const tooltipFormatter = (value: number | undefined, name: string) => {
    if (name === "base" || value === undefined) return [null, null];
    return [formatNumber(Number(value), 3), name];
  };

  const tooltipLabelFormatter = (label: string | number) => {
    const labelStr = String(label);
    const item = data.find((d) => d.treatment === labelStr);
    if (!item) return labelStr;
    return `${labelStr}: ${formatNumber(item.beta_hat, 3)} [${formatNumber(item.ci_lower, 3)}, ${formatNumber(item.ci_upper, 3)}]`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Forest Plot</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="w-full" style={{ height: Math.max(200, sorted.length * 48 + 60) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" horizontal={false} />
              <XAxis
                type="number"
                domain={[domainMin, domainMax]}
                tick={{ fontSize: 11 }}
                tickFormatter={(v: number) => formatNumber(v, 2)}
              />
              <YAxis type="category" dataKey="treatment" tick={{ fontSize: 12 }} width={90} />
              <RechartsTooltip
                formatter={
                  // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                  tooltipFormatter as any
                }
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                labelFormatter={tooltipLabelFormatter as any}
              />
              <Legend />
              <ReferenceLine
                x={0}
                stroke="hsl(var(--destructive))"
                strokeDasharray="3 3"
                strokeWidth={2}
              />
              {/* Invisible base to position CI at correct offset */}
              <Bar dataKey="base" stackId="ci" fill="transparent" legendType="none" />
              {/* Lower CI whisker to point estimate */}
              <Bar
                dataKey="lower_to_point"
                stackId="ci"
                fill="hsl(var(--muted-foreground))"
                name="CI (lower)"
                radius={[0, 0, 0, 0]}
              />
              {/* Point estimate to upper CI whisker */}
              <Bar
                dataKey="point_to_upper"
                stackId="ci"
                fill="hsl(var(--primary))"
                name="CI (upper)"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
