"use client";

import { layoutDag } from "@/lib/utils/dag-layout";
import type { CausalEdge, Construct, Indicator } from "@causal-ssm/api-types";
import { Background, BackgroundVariant, Controls, type NodeTypes, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useMemo } from "react";
import { ConstructNode } from "./construct-node";
import { IndicatorNode } from "./indicator-node";

interface CausalDagProps {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  onNodeClick?: (constructName: string) => void;
}

const nodeTypes: NodeTypes = {
  construct: ConstructNode,
  indicator: IndicatorNode,
};

export function CausalDag({ constructs, edges, indicators, onNodeClick }: CausalDagProps) {
  const { nodes, edges: flowEdges } = useMemo(
    () => layoutDag(constructs, edges, indicators),
    [constructs, edges, indicators],
  );

  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: { id: string; type?: string }) => {
      if (node.type === "construct" && onNodeClick) {
        onNodeClick(node.id);
      }
    },
    [onNodeClick],
  );

  return (
    <div className="h-[500px] w-full rounded-lg border bg-card">
      <ReactFlow
        nodes={nodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        fitView
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
