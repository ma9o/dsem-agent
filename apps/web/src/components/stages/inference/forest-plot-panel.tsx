"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber } from "@/lib/utils/format";
import type { TreatmentEffect } from "@causal-ssm/api-types";
import { useMemo } from "react";
import {
  CartesianGrid,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ticks as d3Ticks, min as d3Min, max as d3Max } from "d3-array";

interface ForestPlotPanelProps {
  results: TreatmentEffect[];
}

const ROW_HEIGHT = 32;
const PADDING_TOP = 4;
const DIAMOND_H = 5;
const DIAMOND_W = 6;

interface ChartRow {
  x: number;
  y: number;
  errorX: [number, number];
  treatment: string;
  beta_hat: number;
  ci_lower: number;
  ci_upper: number;
  se: number;
  crossesZero: boolean;
}

function DiamondDot({ cx, cy, fill }: { cx?: number; cy?: number; fill: string }) {
  if (cx == null || cy == null) return null;
  return (
    <polygon
      points={`${cx},${cy - DIAMOND_H} ${cx + DIAMOND_W},${cy} ${cx},${cy + DIAMOND_H} ${cx - DIAMOND_W},${cy}`}
      fill={fill}
    />
  );
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ChartRow }> }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-md border bg-popover px-3 py-2 text-sm shadow-md">
      <div className="mb-1 font-medium">{d.treatment}</div>
      <div className="space-y-0.5 font-mono text-xs text-muted-foreground">
        <div>{"\u03B2\u0302"} = {formatNumber(d.beta_hat, 3)}</div>
        <div>95% CI [{formatNumber(d.ci_lower, 3)}, {formatNumber(d.ci_upper, 3)}]</div>
        <div>SE = {formatNumber(d.se, 3)}</div>
      </div>
    </div>
  );
}

export function ForestPlotPanel({ results }: ForestPlotPanelProps) {
  if (results.length === 0) return null;

  const sorted = useMemo(
    () => [...results].sort((a, b) => b.beta_hat - a.beta_hat),
    [results],
  );

  const { domainMin, domainMax } = useMemo(() => {
    const allVals = sorted.flatMap((r) => [r.ci_lower, r.ci_upper, 0]);
    const lo = d3Min(allVals) ?? 0;
    const hi = d3Max(allVals) ?? 0;
    const pad = (hi - lo) * 0.15;
    return { domainMin: lo - pad, domainMax: hi + pad };
  }, [sorted]);

  const ticks = useMemo(
    () => d3Ticks(domainMin, domainMax, 5),
    [domainMin, domainMax],
  );

  const { significant, notSignificant } = useMemo(() => {
    const sig: ChartRow[] = [];
    const notSig: ChartRow[] = [];
    sorted.forEach((r, i) => {
      const row: ChartRow = {
        x: r.beta_hat,
        y: i,
        errorX: [r.beta_hat - r.ci_lower, r.ci_upper - r.beta_hat],
        treatment: r.treatment,
        beta_hat: r.beta_hat,
        ci_lower: r.ci_lower,
        ci_upper: r.ci_upper,
        se: r.se,
        crossesZero: r.ci_lower <= 0 && r.ci_upper >= 0,
      };
      if (row.crossesZero) {
        notSig.push(row);
      } else {
        sig.push(row);
      }
    });
    return { significant: sig, notSignificant: notSig };
  }, [sorted]);

  const svgHeight = PADDING_TOP + sorted.length * ROW_HEIGHT + 20;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Forest Plot</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex w-full gap-0">
          {/* Treatment labels column */}
          <div className="shrink-0 pr-3" style={{ paddingTop: PADDING_TOP }}>
            {sorted.map((r) => (
              <div
                key={r.treatment}
                className="flex items-center text-sm text-muted-foreground"
                style={{ height: ROW_HEIGHT }}
              >
                <span className="max-w-[140px] truncate">{r.treatment}</span>
              </div>
            ))}
          </div>

          {/* Chart area */}
          <div className="min-w-0 flex-1">
            <ResponsiveContainer width="100%" height={svgHeight}>
              <ScatterChart margin={{ top: PADDING_TOP, right: 0, bottom: 0, left: 0 }}>
                <CartesianGrid
                  vertical
                  horizontal={false}
                  strokeDasharray="none"
                  className="stroke-muted"
                />
                <XAxis
                  type="number"
                  dataKey="x"
                  domain={[domainMin, domainMax]}
                  ticks={ticks}
                  tickFormatter={(t: number) => formatNumber(t, 2)}
                  fontSize={11}
                  className="fill-muted-foreground"
                  axisLine={false}
                  tickLine={{ className: "stroke-muted-foreground" }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  hide
                  domain={[-0.5, sorted.length - 0.5]}
                  reversed
                />
                <ReferenceLine
                  x={0}
                  strokeDasharray="4 3"
                  className="stroke-muted-foreground/60"
                />
                <RechartTooltip
                  content={<CustomTooltip />}
                  cursor={false}
                />
                {significant.length > 0 && (
                  <Scatter
                    data={significant}
                    shape={<DiamondDot fill="var(--primary)" />}
                    isAnimationActive={false}
                  >
                    <ErrorBar
                      dataKey="errorX"
                      direction="x"
                      width={10}
                      stroke="var(--primary)"
                      strokeWidth={1.5}
                    />
                  </Scatter>
                )}
                {notSignificant.length > 0 && (
                  <Scatter
                    data={notSignificant}
                    shape={<DiamondDot fill="var(--muted-foreground)" />}
                    isAnimationActive={false}
                  >
                    <ErrorBar
                      dataKey="errorX"
                      direction="x"
                      width={10}
                      stroke="var(--muted-foreground)"
                      strokeWidth={1.5}
                    />
                  </Scatter>
                )}
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Numeric summary column */}
          <div className="shrink-0 pl-3" style={{ paddingTop: PADDING_TOP }}>
            {sorted.map((r) => {
              const crossesZero = r.ci_lower <= 0 && r.ci_upper >= 0;
              return (
                <div
                  key={r.treatment}
                  className={`flex items-center font-mono text-xs ${crossesZero ? "text-muted-foreground" : "text-foreground"}`}
                  style={{ height: ROW_HEIGHT }}
                >
                  <span className="tabular-nums whitespace-nowrap">
                    {formatNumber(r.beta_hat, 2)}{" "}
                    <span className="text-muted-foreground">
                      [{formatNumber(r.ci_lower, 2)},{" "}
                      {formatNumber(r.ci_upper, 2)}]
                    </span>
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
