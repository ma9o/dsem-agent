"use client";

import type { Indicator } from "@causal-ssm/api-types";
import { Handle, type NodeProps, Position } from "@xyflow/react";
import { memo } from "react";

function IndicatorNodeInner({ data }: NodeProps) {
  const indicator = data as unknown as Indicator;

  return (
    <div className="rounded-md border border-muted-foreground bg-muted px-3 py-1.5 shadow-sm">
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground" />

      <span className="text-xs font-medium text-muted-foreground">
        {indicator.name}
      </span>

      <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground" />
    </div>
  );
}

export const IndicatorNode = memo(IndicatorNodeInner);
