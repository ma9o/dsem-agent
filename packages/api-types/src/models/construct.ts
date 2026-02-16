export type Role = "endogenous" | "exogenous";
export type TemporalStatus = "time_varying" | "time_invariant";
export type CausalGranularity = "hourly" | "daily" | "weekly" | "monthly" | "yearly";

export interface Construct {
  name: string;
  description: string;
  role: Role;
  is_outcome: boolean;
  temporal_status: TemporalStatus;
  causal_granularity: CausalGranularity | null;
}

export interface CausalEdge {
  cause: string;
  effect: string;
  description: string;
  lagged: boolean;
}

export interface LatentModel {
  constructs: Construct[];
  edges: CausalEdge[];
}
