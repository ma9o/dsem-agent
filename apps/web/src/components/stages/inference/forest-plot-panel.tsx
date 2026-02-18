"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber } from "@/lib/utils/format";
import type { TreatmentEffect } from "@causal-ssm/api-types";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface ForestPlotPanelProps {
  results: TreatmentEffect[];
}

const ROW_HEIGHT = 32;
const PADDING_TOP = 4;
const AXIS_HEIGHT = 20;
const CAP_HALF = 5;
const DIAMOND_H = 5;
const DIAMOND_W = 6;

function niceAxisTicks(min: number, max: number, targetCount = 5): number[] {
  const range = max - min;
  const rawStep = range / targetCount;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const residual = rawStep / magnitude;
  const niceStep =
    residual <= 1.5
      ? magnitude
      : residual <= 3
        ? 2 * magnitude
        : residual <= 7
          ? 5 * magnitude
          : 10 * magnitude;

  const start = Math.ceil(min / niceStep) * niceStep;
  const ticks: number[] = [];
  for (let t = start; t <= max + niceStep * 0.01; t += niceStep) {
    ticks.push(Math.round(t * 1e9) / 1e9);
  }
  return ticks;
}

interface HoverState {
  treatment: string;
  x: number;
  y: number;
}

export function ForestPlotPanel({ results }: ForestPlotPanelProps) {
  if (results.length === 0) return null;

  const sorted = useMemo(
    () => [...results].sort((a, b) => b.beta_hat - a.beta_hat),
    [results],
  );

  const { domainMin, domainMax } = useMemo(() => {
    const allVals = sorted.flatMap((r) => [r.ci_lower, r.ci_upper, 0]);
    const pad = (Math.max(...allVals) - Math.min(...allVals)) * 0.15;
    return {
      domainMin: Math.min(...allVals) - pad,
      domainMax: Math.max(...allVals) + pad,
    };
  }, [sorted]);

  const ticks = useMemo(
    () => niceAxisTicks(domainMin, domainMax),
    [domainMin, domainMax],
  );

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [svgWidth, setSvgWidth] = useState(400);
  const [hover, setHover] = useState<HoverState | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setSvgWidth(entry.contentRect.width);
      }
    });
    obs.observe(svgRef.current);
    return () => obs.disconnect();
  }, []);

  // Pixel-based x scale
  const xPx = useCallback(
    (value: number) => {
      return ((value - domainMin) / (domainMax - domainMin)) * svgWidth;
    },
    [domainMin, domainMax, svgWidth],
  );

  const svgHeight = PADDING_TOP + sorted.length * ROW_HEIGHT + AXIS_HEIGHT;

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGGElement>, treatment: string) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      setHover({
        treatment,
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    },
    [],
  );

  const handleMouseLeave = useCallback(() => setHover(null), []);

  const hoveredItem = hover
    ? sorted.find((r) => r.treatment === hover.treatment)
    : null;

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

          {/* SVG chart area */}
          <div className="relative min-w-0 flex-1" ref={containerRef}>
            <svg
              ref={svgRef}
              width="100%"
              height={svgHeight}
              className="overflow-visible"
            >
              {/* Gridlines */}
              {ticks.map((t) => {
                const x = xPx(t);
                return (
                  <line
                    key={`grid-${t}`}
                    x1={x}
                    x2={x}
                    y1={PADDING_TOP}
                    y2={PADDING_TOP + sorted.length * ROW_HEIGHT}
                    className="stroke-muted"
                    strokeWidth={1}
                  />
                );
              })}

              {/* Null-effect reference line at 0 */}
              <line
                x1={xPx(0)}
                x2={xPx(0)}
                y1={PADDING_TOP - 2}
                y2={PADDING_TOP + sorted.length * ROW_HEIGHT + 2}
                className="stroke-muted-foreground/60"
                strokeWidth={1}
                strokeDasharray="4 3"
              />

              {/* Treatment rows */}
              {sorted.map((r, i) => {
                const cy = PADDING_TOP + i * ROW_HEIGHT + ROW_HEIGHT / 2;
                const x1 = xPx(r.ci_lower);
                const x2 = xPx(r.ci_upper);
                const xPt = xPx(r.beta_hat);
                const crossesZero = r.ci_lower <= 0 && r.ci_upper >= 0;
                const isHovered = hover?.treatment === r.treatment;

                const colorClass = crossesZero
                  ? "stroke-muted-foreground"
                  : "stroke-primary";
                const fillClass = crossesZero
                  ? "fill-muted-foreground"
                  : "fill-primary";

                return (
                  <g
                    key={r.treatment}
                    className="cursor-pointer"
                    onMouseMove={(e) => handleMouseMove(e, r.treatment)}
                    onMouseLeave={handleMouseLeave}
                  >
                    {/* Hover target */}
                    <rect
                      x={0}
                      y={cy - ROW_HEIGHT / 2}
                      width={svgWidth}
                      height={ROW_HEIGHT}
                      fill="transparent"
                    />
                    {/* Row highlight on hover */}
                    {isHovered && (
                      <rect
                        x={0}
                        y={cy - ROW_HEIGHT / 2}
                        width={svgWidth}
                        height={ROW_HEIGHT}
                        className="fill-muted"
                        opacity={0.4}
                      />
                    )}
                    {/* CI whisker line */}
                    <line
                      x1={x1}
                      x2={x2}
                      y1={cy}
                      y2={cy}
                      className={colorClass}
                      strokeWidth={isHovered ? 2 : 1.5}
                      strokeLinecap="round"
                    />
                    {/* CI end caps */}
                    <line
                      x1={x1}
                      x2={x1}
                      y1={cy - CAP_HALF}
                      y2={cy + CAP_HALF}
                      className={colorClass}
                      strokeWidth={1.5}
                    />
                    <line
                      x1={x2}
                      x2={x2}
                      y1={cy - CAP_HALF}
                      y2={cy + CAP_HALF}
                      className={colorClass}
                      strokeWidth={1.5}
                    />
                    {/* Point estimate diamond */}
                    <polygon
                      points={`${xPt},${cy - DIAMOND_H} ${xPt + DIAMOND_W},${cy} ${xPt},${cy + DIAMOND_H} ${xPt - DIAMOND_W},${cy}`}
                      className={fillClass}
                    />
                  </g>
                );
              })}

              {/* X-axis ticks */}
              {ticks.map((t) => {
                const x = xPx(t);
                const tickY = PADDING_TOP + sorted.length * ROW_HEIGHT;
                return (
                  <g key={`tick-${t}`}>
                    <line
                      x1={x}
                      x2={x}
                      y1={tickY}
                      y2={tickY + 4}
                      className="stroke-muted-foreground"
                      strokeWidth={1}
                    />
                    <text
                      x={x}
                      y={tickY + 16}
                      textAnchor="middle"
                      className="fill-muted-foreground"
                      fontSize={11}
                    >
                      {formatNumber(t, 2)}
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Floating tooltip */}
            {hover && hoveredItem && (
              <div
                className="pointer-events-none absolute z-50 rounded-md border bg-popover px-3 py-2 text-sm shadow-md"
                style={{
                  left: hover.x,
                  top: hover.y - 8,
                  transform: "translate(-50%, -100%)",
                }}
              >
                <div className="mb-1 font-medium">{hoveredItem.treatment}</div>
                <div className="space-y-0.5 font-mono text-xs text-muted-foreground">
                  <div>
                    {"\u03B2\u0302"} = {formatNumber(hoveredItem.beta_hat, 3)}
                  </div>
                  <div>
                    95% CI [{formatNumber(hoveredItem.ci_lower, 3)},{" "}
                    {formatNumber(hoveredItem.ci_upper, 3)}]
                  </div>
                  <div>SE = {formatNumber(hoveredItem.se, 3)}</div>
                </div>
              </div>
            )}
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
