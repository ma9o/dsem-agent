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
  psis_k_hat?: number;
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
  /** Individual y_rep trajectories for spaghetti plots (each length T). */
  spaghetti_draws?: number[][];
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

// ---------------------------------------------------------------------------
// MCMC Diagnostics
// ---------------------------------------------------------------------------

/** Per-parameter convergence diagnostic from MCMC. */
export interface MCMCParamDiagnostic {
  parameter: string;
  r_hat: number | number[];
  ess_bulk: number | number[];
  /** Tail ESS (via ArviZ). */
  ess_tail?: number | number[];
  /** Monte Carlo standard error of the mean. */
  mcse_mean?: number | number[];
}

/** Chain-level trace data for trace plots. */
export interface TraceChain {
  chain: number;
  values: number[];
}

export interface TraceData {
  parameter: string;
  chains: TraceChain[];
}

/** Chain-level rank histogram for mixing assessment. */
export interface RankHistogramChain {
  chain: number;
  counts: number[];
}

export interface RankHistogram {
  parameter: string;
  n_bins: number;
  expected_per_bin: number;
  chains: RankHistogramChain[];
}

/** NUTS energy histogram data (Betancourt 2017). */
export interface EnergyHistogram {
  bin_centers: number[];
  density: number[];
}

/** NUTS energy diagnostics for detecting geometric pathologies. */
export interface EnergyDiagnostics {
  /** Marginal energy distribution histogram. */
  energy_hist: EnergyHistogram;
  /** Energy transition (dE) distribution histogram. */
  energy_transition_hist: EnergyHistogram;
  /** Bayesian Fraction of Missing Information per chain (should be > 0.3). */
  bfmi: number[];
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
  /** Thinned chain-level samples for trace plots. */
  trace_data?: TraceData[];
  /** Rank histograms per parameter for mixing assessment. */
  rank_histograms?: RankHistogram[];
  /** NUTS energy diagnostics (Betancourt 2017). */
  energy?: EnergyDiagnostics;
}

/** SVI diagnostics (ELBO loss curve). */
export interface SVIDiagnostics {
  /** ELBO loss per optimization step (thinned to ~500 points). */
  elbo_losses: number[];
}

// ---------------------------------------------------------------------------
// LOO Diagnostics
// ---------------------------------------------------------------------------

/** Leave-one-out cross-validation diagnostics via PSIS. */
export interface LOODiagnostics {
  /** Expected log pointwise predictive density. */
  elpd_loo: number;
  /** Effective number of parameters. */
  p_loo: number;
  /** Standard error of ELPD estimate. */
  se: number;
  /** Number of data points. */
  n_data_points: number;
  /** Per-observation Pareto k values (length = n_data_points). */
  pareto_k?: number[];
  /** Number of observations with Pareto k > 0.7 (problematic). */
  n_bad_k?: number;
  /** LOO-PIT values for calibration (length = n_data_points). */
  loo_pit?: number[];
}

// ---------------------------------------------------------------------------
// Posterior Visualization Data
// ---------------------------------------------------------------------------

/** Marginal posterior density for one parameter. */
export interface PosteriorMarginal {
  parameter: string;
  /** Bin centers for density plot. */
  x_values: number[];
  /** Density values (normalized). */
  density: number[];
  mean: number;
  sd: number;
  /** 3% HDI bound. */
  hdi_3: number;
  /** 97% HDI bound. */
  hdi_97: number;
}

/** Pairwise posterior scatter data for two parameters. */
export interface PosteriorPair {
  param_x: string;
  param_y: string;
  /** Thinned x-axis samples. */
  x_values: number[];
  /** Thinned y-axis samples. */
  y_values: number[];
  /** Per-sample divergence flag (only present when divergences exist). */
  divergent?: boolean[];
}
