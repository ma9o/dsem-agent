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
  value: number;
  passed: boolean;
}

/** Per-variable quantile bands for PPC ribbon/overlay plots (Gabry's ppc_dens_overlay). */
export interface PPCOverlay {
  variable: string;
  /** Observed data, null for missing timesteps. Length = T. */
  observed: (number | null)[];
  /** 2.5th percentile of y_rep per timestep. */
  q025: number[];
  /** 25th percentile. */
  q25: number[];
  /** 50th percentile (median). */
  median: number[];
  /** 75th percentile. */
  q75: number[];
  /** 97.5th percentile. */
  q975: number[];
}

/** Distribution of a test statistic across y_rep vs observed (Gabry's ppc_stat). */
export interface PPCTestStat {
  variable: string;
  stat_name: "mean" | "sd" | "min" | "max";
  observed_value: number;
  /** T(y_rep) values, one per posterior draw. */
  rep_values: number[];
  /** Posterior predictive p-value: P(T(y_rep) >= T(y_obs)). */
  p_value: number;
}

export interface PPCResult {
  per_variable_warnings: PPCWarning[];
  overall_passed: boolean;
  overlays: PPCOverlay[];
  test_stats: PPCTestStat[];
}

export interface InferenceMetadata {
  method: string;
  n_samples: number;
  duration_seconds: number;
}

/** Per-parameter convergence diagnostic from MCMC. */
export interface MCMCParamDiagnostic {
  parameter: string;
  r_hat: number | number[];
  ess_bulk: number | number[];
}

/** MCMC sampler-level diagnostics (NUTS/HMC). */
export interface MCMCDiagnostics {
  per_parameter: MCMCParamDiagnostic[];
  num_divergences: number;
  divergence_rate: number;
  tree_depth_mean: number;
  tree_depth_max: number;
  accept_prob_mean: number;
  num_chains: number | null;
  num_samples: number | null;
}

/** SVI diagnostics (ELBO loss curve). */
export interface SVIDiagnostics {
  /** ELBO loss per optimization step (thinned to ~500 points). */
  elbo_losses: number[];
}
