"use client";

import { Badge } from "@/components/ui/badge";
import { evaluatePdf } from "@/lib/utils/distributions";
import { formatNumber } from "@/lib/utils/format";
import {
  CartesianGrid,
  Line,
  LineChart,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

interface PriorDensityChartProps {
  distribution: string;
  params: Record<string, number>;
  /** Pre-computed density points from the pipeline (preferred over client-side approximation). */
  densityPoints?: Array<{ x: number; y: number }>;
}

export function PriorDensityChart({ distribution, params, densityPoints }: PriorDensityChartProps) {
  const pdfData = densityPoints ?? evaluatePdf(distribution, params);
  const isApproximate = !densityPoints;

  // Find the mode (peak) of the distribution for annotation
  const peak = pdfData.reduce((max, point) => (point.y > max.y ? point : max), pdfData[0]);

  // Determine mean reference line if applicable
  const mean = params.mu ?? params.loc ?? null;

  const tooltipFormatter = (value: number | undefined) =>
    value !== undefined ? [formatNumber(Number(value), 4), "Density"] : [null, null];

  const tooltipLabelFormatter = (label: string | number) => `x = ${formatNumber(Number(label), 3)}`;

  return (
    <div className="space-y-3">
      <div className="relative h-48 w-full">
        {isApproximate && (
          <span className="absolute top-1 right-2 z-10 rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
            approx.
          </span>
        )}
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={pdfData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="x"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v: number) => formatNumber(v, 2)}
              tick={{ fontSize: 11 }}
            />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => formatNumber(v, 2)} />
            <RechartsTooltip
              formatter={
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                tooltipFormatter as any
              }
              // biome-ignore lint/suspicious/noExplicitAny: recharts overload
              labelFormatter={tooltipLabelFormatter as any}
            />
            {mean !== null && (
              <ReferenceLine
                x={mean}
                stroke="var(--muted-foreground)"
                strokeDasharray="4 4"
                label={{ value: `\u03BC=${formatNumber(mean, 2)}`, position: "top", fontSize: 11 }}
              />
            )}
            <Line
              type="monotone"
              dataKey="y"
              stroke="var(--primary)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Param annotations */}
      <div className="flex flex-wrap gap-2">
        <Badge variant="outline" className="text-xs">
          {distribution}
        </Badge>
        {Object.entries(params).map(([key, value]) => (
          <Badge key={key} variant="secondary" className="text-xs">
            {key} = {formatNumber(value)}
          </Badge>
        ))}
        <Badge variant="secondary" className="text-xs">
          mode {"\u2248"} {formatNumber(peak?.x ?? 0, 2)}
        </Badge>
      </div>
    </div>
  );
}
