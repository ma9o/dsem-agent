"use client";

import { Badge } from "@/components/ui/badge";
import { formatNumber } from "@/lib/utils/format";
import type { PPCTestStat } from "@causal-ssm/api-types";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

interface PPCTestStatChartProps {
  stat: PPCTestStat;
}

function buildHistogram(values: number[], nBins: number = 20) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / nBins;

  const bins = Array.from({ length: nBins }, (_, i) => ({
    x: min + (i + 0.5) * binWidth,
    count: 0,
    binStart: min + i * binWidth,
    binEnd: min + (i + 1) * binWidth,
  }));

  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), nBins - 1);
    bins[idx].count++;
  }

  return bins;
}

export function PPCTestStatChart({ stat }: PPCTestStatChartProps) {
  const bins = buildHistogram(stat.rep_values);
  const isExtreme = stat.p_value < 0.05 || stat.p_value > 0.95;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-muted-foreground">
          T = {stat.stat_name}({stat.variable})
        </span>
        <Badge variant={isExtreme ? "destructive" : "success"} className="text-[10px]">
          p = {formatNumber(stat.p_value, 2)}
        </Badge>
      </div>
      <div className="h-36 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" vertical={false} />
            <XAxis
              dataKey="x"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fontSize: 10 }}
              tickFormatter={(v: number) => formatNumber(v, 1)}
            />
            <YAxis tick={{ fontSize: 10 }} hide />
            <RechartsTooltip
              formatter={
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                ((value: number) => [value, "Count"]) as any
              }
              // biome-ignore lint/suspicious/noExplicitAny: recharts overload
              labelFormatter={
                ((label: number) => `T(y_rep) \u2248 ${formatNumber(label, 2)}`) as any
              }
            />
            <Bar dataKey="count" fill="var(--primary)" fillOpacity={0.5} />
            <ReferenceLine
              x={stat.observed_value}
              stroke="var(--foreground)"
              strokeWidth={2}
              strokeDasharray="0"
              label={{
                value: `T(y) = ${formatNumber(stat.observed_value, 2)}`,
                position: "top",
                fontSize: 10,
                fill: "var(--foreground)",
              }}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
