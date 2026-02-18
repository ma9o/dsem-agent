"use client";

import { formatNumber } from "@/lib/utils/format";
import type { PosteriorPair } from "@causal-ssm/api-types";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

interface PosteriorPairsChartProps {
  pair: PosteriorPair;
}

export function PosteriorPairsChart({ pair }: PosteriorPairsChartProps) {
  const data = pair.x_values.map((x, i) => ({
    x,
    y: pair.y_values[i],
  }));

  return (
    <div className="space-y-1">
      <span className="text-xs font-mono text-muted-foreground">
        {pair.param_x} vs {pair.param_y}
      </span>
      <div className="h-36 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="x"
              type="number"
              tick={{ fontSize: 9 }}
              tickFormatter={(v: number) => formatNumber(v, 2)}
              label={{ value: pair.param_x, position: "insideBottom", offset: -2, fontSize: 9 }}
            />
            <YAxis
              dataKey="y"
              type="number"
              tick={{ fontSize: 9 }}
              tickFormatter={(v: number) => formatNumber(v, 2)}
              label={{ value: pair.param_y, angle: -90, position: "insideLeft", offset: 10, fontSize: 9 }}
            />
            <RechartsTooltip
              formatter={
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                ((value: number) => [formatNumber(value, 3)]) as any
              }
            />
            <Scatter data={data} fill="var(--primary)" fillOpacity={0.3} r={2} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
