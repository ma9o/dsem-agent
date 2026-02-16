import type { CausalEdge, Construct, Indicator } from "@causal-ssm/api-types";
import dagre from "@dagrejs/dagre";
import type { Edge, Node } from "@xyflow/react";

interface LayoutResult {
  nodes: Node[];
  edges: Edge[];
}

export function layoutDag(
  constructs: Construct[],
  causalEdges: CausalEdge[],
  indicators?: Indicator[],
): LayoutResult {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 80, ranksep: 100 });

  // Add construct nodes
  for (const c of constructs) {
    g.setNode(c.name, { width: 200, height: 60 });
  }

  // Add causal edges
  for (const e of causalEdges) {
    g.setEdge(e.cause, e.effect);
  }

  // Add indicator nodes if provided
  if (indicators) {
    for (const ind of indicators) {
      g.setNode(ind.name, { width: 160, height: 40 });
      g.setEdge(ind.construct_name, ind.name);
    }
  }

  dagre.layout(g);

  const nodes: Node[] = [];

  for (const c of constructs) {
    const pos = g.node(c.name);
    nodes.push({
      id: c.name,
      type: "construct",
      position: { x: pos.x - 100, y: pos.y - 30 },
      data: { ...c },
    });
  }

  if (indicators) {
    for (const ind of indicators) {
      const pos = g.node(ind.name);
      nodes.push({
        id: ind.name,
        type: "indicator",
        position: { x: pos.x - 80, y: pos.y - 20 },
        data: { ...ind },
      });
    }
  }

  const edges: Edge[] = causalEdges.map((e, i) => ({
    id: `causal-${i}`,
    source: e.cause,
    target: e.effect,
    data: { ...e },
    style: {
      stroke: e.lagged ? "var(--edge-lagged)" : "var(--edge-contemporary)",
      strokeWidth: 2,
      strokeDasharray: e.lagged ? undefined : "5,5",
    },
    label: e.lagged ? "t-1 → t" : "t → t",
    labelStyle: { fontSize: 10, fill: "var(--edge-label)" },
    animated: !e.lagged,
    markerEnd: {
      type: "arrowclosed" as const,
      color: e.lagged ? "var(--edge-lagged)" : "var(--edge-contemporary)",
      width: 16,
      height: 16,
    },
  }));

  if (indicators) {
    for (const ind of indicators) {
      edges.push({
        id: `loading-${ind.name}`,
        source: ind.construct_name,
        target: ind.name,
        style: { stroke: "var(--edge-indicator)", strokeWidth: 1.5, strokeDasharray: "3,3" },
        markerEnd: {
          type: "arrowclosed" as const,
          color: "var(--edge-indicator)",
          width: 12,
          height: 12,
        },
      });
    }
  }

  return { nodes, edges };
}
