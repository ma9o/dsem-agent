"use client";

import { layoutDag } from "@/lib/utils/dag-layout";
import type { CausalEdge, Construct, Indicator } from "@causal-ssm/api-types";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  type NodeTypes,
  ReactFlow,
} from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import { ConstructNode } from "./construct-node";
import { IndicatorNode } from "./indicator-node";

interface CausalDagProps {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  onNodeClick?: (constructName: string) => void;
  height?: string;
}

const nodeTypes: NodeTypes = {
  construct: ConstructNode,
  indicator: IndicatorNode,
};

export function CausalDag({
  constructs,
  edges,
  indicators,
  onNodeClick,
  height = "500px",
}: CausalDagProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const { nodes, edges: flowEdges } = useMemo(
    () => layoutDag(constructs, edges, indicators),
    [constructs, edges, indicators],
  );

  // Dim nodes/edges not connected to selection
  const styledNodes = useMemo(() => {
    if (!selectedNode) return nodes;
    const connectedIds = new Set<string>([selectedNode]);
    for (const e of flowEdges) {
      if (e.source === selectedNode || e.target === selectedNode) {
        connectedIds.add(e.source);
        connectedIds.add(e.target);
      }
    }
    return nodes.map((n) => ({
      ...n,
      style: connectedIds.has(n.id) ? {} : { opacity: 0.3 },
    }));
  }, [nodes, flowEdges, selectedNode]);

  const styledEdges = useMemo(() => {
    if (!selectedNode) return flowEdges;
    return flowEdges.map((e) => ({
      ...e,
      style: {
        ...e.style,
        opacity: e.source === selectedNode || e.target === selectedNode ? 1 : 0.15,
      },
    }));
  }, [flowEdges, selectedNode]);

  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: { id: string; type?: string }) => {
      if (node.type === "construct") {
        setSelectedNode((prev) => (prev === node.id ? null : node.id));
        onNodeClick?.(node.id);
      }
    },
    [onNodeClick],
  );

  const handlePaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  return (
    <div className="w-full rounded-lg border bg-card" style={{ height }}>
      <ReactFlow
        nodes={styledNodes}
        edges={styledEdges}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        onPaneClick={handlePaneClick}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        nodesDraggable={false}
        nodesConnectable={false}
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
          style: { strokeWidth: 2 },
        }}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        <Controls showInteractive={false} />
        <MiniMap
          nodeStrokeWidth={3}
          zoomable
          pannable
          className="!bg-secondary/50 !border-border"
        />
      </ReactFlow>
    </div>
  );
}
