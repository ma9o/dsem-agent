export interface PriorSource {
  title: string;
  url: string | null;
  snippet: string;
  effect_size: string | null;
}

export interface PriorProposal {
  parameter: string;
  distribution: string;
  params: Record<string, number>;
  sources: PriorSource[];
  confidence: number;
  /** Pipeline-determined confidence interpretation (avoids frontend hardcoding thresholds). */
  confidence_level?: "high" | "medium" | "low";
  reasoning: string;
  /** Pre-computed density points from the pipeline (preferred over client-side approximation). */
  density_points?: Array<{ x: number; y: number }>;
}

export interface PriorValidationResult {
  parameter: string;
  is_valid: boolean;
  issue: string | null;
  suggested_adjustment: string | null;
}

export interface RawPriorSample {
  paraphrase_id: number;
  mu: number;
  sigma: number;
  confidence: number;
  reasoning: string;
}

export interface AggregatedPrior {
  method: "simple" | "gmm";
  mu: number;
  sigma: number;
  mixture_weights: number[] | null;
  mixture_means: number[] | null;
  mixture_stds: number[] | null;
  n_samples: number;
}

export interface PriorResearchResult {
  parameter: string;
  proposal: PriorProposal;
  literature_found: boolean;
  raw_response: string;
  aggregation: AggregatedPrior | null;
}
