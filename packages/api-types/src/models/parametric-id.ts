export interface TRuleResult {
  satisfies: boolean;
  n_free_params: number;
  n_moments: number;
  param_counts: Record<string, number>;
}

export type ParameterClassification = "structurally_identified" | "boundary" | "weak";

export interface ParameterIdentification {
  name: string;
  classification: ParameterClassification;
  contraction_ratio: number | null;
}

export interface ParametricIdResult {
  t_rule: TRuleResult;
  per_param_classification: ParameterIdentification[];
  weak_params: string[];
}
