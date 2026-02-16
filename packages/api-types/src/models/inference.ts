export interface TreatmentEffect {
  treatment: string;
  beta_hat: number;
  se: number;
  ci_lower: number;
  ci_upper: number;
  p_positive: number;
  identifiable: boolean;
  sensitivity_flag: boolean;
}

export type PowerScalingDiagnosis = "prior_dominated" | "well_identified" | "prior_data_conflict";

export interface PowerScalingResult {
  parameter: string;
  diagnosis: PowerScalingDiagnosis;
  prior_sensitivity: number;
  likelihood_sensitivity: number;
}

export interface PPCWarning {
  variable: string;
  check_type: "calibration" | "autocorrelation" | "variance";
  message: string;
  passed: boolean;
}

export interface PPCResult {
  per_variable_warnings: PPCWarning[];
  overall_passed: boolean;
}

export interface InferenceMetadata {
  method: string;
  n_samples: number;
  duration_seconds: number;
}
