import type { CausalEdge, Construct, Indicator } from "@causal-ssm/api-types";
import ELK, { type ElkNode } from "elkjs/lib/elk.bundled.js";
import type { Edge, Node } from "@xyflow/react";

export interface LayoutResult {
  nodes: Node[];
  edges: Edge[];
}

const elk = new ELK();

const CONSTRUCT_WIDTH = 200;
const CONSTRUCT_HEIGHT = 60;
const INDICATOR_WIDTH = 160;
const INDICATOR_HEIGHT = 40;

const ELK_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.spacing.nodeNode": "120",
  "elk.layered.spacing.nodeNodeBetweenLayers": "150",
  "elk.spacing.edgeNode": "40",
  "elk.spacing.edgeEdge": "20",
  "elk.edgeRouting": "SPLINES",
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
};

export async function layoutDag(
  constructs: Construct[],
  causalEdges: CausalEdge[],
  indicators?: Indicator[],
): Promise<LayoutResult> {
  const children: ElkNode[] = [];
  const elkEdges: Array<{ id: string; sources: string[]; targets: string[] }> = [];

  // Only contemporaneous edges define the hierarchy for layout.
  // Lagged edges (temporal feedback) are overlaid after layout so they
  // don't distort the layering with back-edges.
  const contemporaneousEdges = causalEdges.filter((e) => !e.lagged);
  const laggedEdges = causalEdges.filter((e) => e.lagged);

  for (const c of constructs) {
    children.push({ id: c.name, width: CONSTRUCT_WIDTH, height: CONSTRUCT_HEIGHT });
  }

  for (let i = 0; i < contemporaneousEdges.length; i++) {
    const e = contemporaneousEdges[i];
    elkEdges.push({ id: `causal-contemp-${i}`, sources: [e.cause], targets: [e.effect] });
  }

  if (indicators) {
    for (const ind of indicators) {
      children.push({ id: ind.name, width: INDICATOR_WIDTH, height: INDICATOR_HEIGHT });
      elkEdges.push({ id: `loading-${ind.name}`, sources: [ind.construct_name], targets: [ind.name] });
    }
  }

  const graph: ElkNode = {
    id: "root",
    layoutOptions: ELK_OPTIONS,
    children,
    edges: elkEdges,
  };

  const layouted = await elk.layout(graph);

  const posMap = new Map<string, { x: number; y: number }>();
  for (const n of layouted.children ?? []) {
    posMap.set(n.id, { x: n.x ?? 0, y: n.y ?? 0 });
  }

  const nodes: Node[] = [];

  for (const c of constructs) {
    const pos = posMap.get(c.name);
    if (!pos) continue;
    nodes.push({ id: c.name, type: "construct", position: pos, data: { ...c } });
  }

  if (indicators) {
    for (const ind of indicators) {
      const pos = posMap.get(ind.name);
      if (!pos) continue;
      nodes.push({ id: ind.name, type: "indicator", position: pos, data: { ...ind } });
    }
  }

  // Build edges — contemporaneous edges use smoothstep (orthogonal feel),
  // lagged edges use bezier (curved arcs that float over the layout).
  const edges: Edge[] = [];

  for (let i = 0; i < causalEdges.length; i++) {
    const e = causalEdges[i];
    edges.push({
      id: `causal-${i}`,
      source: e.cause,
      target: e.effect,
      type: e.lagged ? "default" : "smoothstep",
      data: { ...e },
      style: {
        stroke: e.lagged ? "var(--edge-lagged)" : "var(--edge-contemporary)",
        strokeWidth: e.lagged ? 1.5 : 2,
        strokeDasharray: e.lagged ? "6,4" : undefined,
      },
      animated: false,
      markerEnd: {
        type: "arrowclosed" as const,
        color: e.lagged ? "var(--edge-lagged)" : "var(--edge-contemporary)",
        width: 14,
        height: 14,
      },
    });
  }

  if (indicators) {
    for (const ind of indicators) {
      edges.push({
        id: `loading-${ind.name}`,
        source: ind.construct_name,
        target: ind.name,
        type: "smoothstep",
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
