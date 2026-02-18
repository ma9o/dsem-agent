export type ValidationSeverity = "error" | "warning" | "info";

export interface ValidationIssue {
  indicator: string;
  issue_type: string;
  severity: ValidationSeverity;
  message: string;
}

export interface IndicatorHealth {
  indicator: string;
  n_obs: number;
  variance: number | null;
  time_coverage_ratio: number | null;
  max_gap_ratio: number | null;
  dtype_violations: number;
  duplicate_pct: number;
  arithmetic_sequence_detected: boolean;
}

export interface ValidationReport {
  is_valid: boolean;
  issues: ValidationIssue[];
  per_indicator_health: IndicatorHealth[];
}
