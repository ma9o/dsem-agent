"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils/cn";
import type { Construct, Indicator } from "@causal-ssm/api-types";
import { Handle, type NodeProps, Position } from "@xyflow/react";
import { Star } from "lucide-react";
import { memo } from "react";

interface ConstructNodeData extends Construct {
  indicators?: Indicator[];
}

function ConstructNodeInner({ data, selected }: NodeProps) {
  const construct = data as unknown as ConstructNodeData;
  const indicators = construct.indicators ?? [];

  return (
    <div
      className={cn(
        "rounded-lg border-2 bg-card shadow-sm transition-all duration-200 cursor-pointer",
        "hover:shadow-md hover:-translate-y-0.5",
        construct.role === "endogenous"
          ? "border-blue-400 dark:border-blue-600"
          : "border-slate-300 dark:border-slate-600",
        construct.is_outcome && "ring-2 ring-amber-400 ring-offset-1 dark:ring-amber-500",
        selected && "shadow-lg ring-2 ring-primary ring-offset-2",
      )}
    >
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground !w-2 !h-2" />

      <div className="px-4 py-3">
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-semibold leading-tight">{construct.name}</span>
          {construct.is_outcome && (
            <Star className="h-3.5 w-3.5 shrink-0 fill-amber-400 text-amber-400" />
          )}
        </div>

        <div className="mt-1.5 flex flex-wrap gap-1">
          <Badge
            variant={construct.role === "endogenous" ? "default" : "secondary"}
            className="px-1.5 py-0 text-[10px]"
          >
            {construct.role === "endogenous" ? "endo" : "exo"}
          </Badge>
          <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
            {construct.temporal_status === "time_varying" ? "varying" : "invariant"}
          </Badge>
          {construct.causal_granularity && (
            <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
              {construct.causal_granularity}
            </Badge>
          )}
        </div>
      </div>

      {indicators.length > 0 && (
        <div className="border-t border-dashed border-slate-200 dark:border-slate-600 px-3 py-1.5">
          {indicators.map((ind) => (
            <div key={ind.name} className="flex items-center justify-between gap-2 py-0.5">
              <span className="text-[11px] text-muted-foreground truncate">
                {ind.name}
              </span>
              <span className="text-[9px] text-muted-foreground shrink-0">
                {ind.measurement_dtype}
              </span>
            </div>
          ))}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-muted-foreground !w-2 !h-2"
      />
    </div>
  );
}

export const ConstructNode = memo(ConstructNodeInner);
