// Run & stage metadata
export type { PipelineRun, RunStatus, StageState, StageStatus } from "./run";
export type { StageId, StageMeta } from "./stages";
export { STAGES, STAGE_IDS } from "./stages";

// Domain models
export type {
  Construct,
  CausalEdge,
  LatentModel,
  Role,
  TemporalStatus,
  CausalGranularity,
} from "./models/construct";

export type {
  Indicator,
  MeasurementModel,
  MeasurementDtype,
  AggregationFunction,
} from "./models/indicator";

export type {
  CausalSpec,
  IdentifiabilityStatus,
  IdentifiedTreatmentStatus,
  NonIdentifiableTreatmentStatus,
} from "./models/causal-spec";

export type { Extraction, WorkerOutput, WorkerStatus } from "./models/worker";

export type {
  ValidationReport,
  ValidationIssue,
  ValidationSeverity,
  IndicatorHealth,
} from "./models/validation";

export type {
  ModelSpec,
  LikelihoodSpec,
  ParameterSpec,
  DistributionFamily,
  LinkFunction,
  ParameterRole,
  ParameterConstraint,
} from "./models/model-spec";

export type {
  PriorProposal,
  PriorSource,
  PriorValidationResult,
  RawPriorSample,
  AggregatedPrior,
  PriorResearchResult,
} from "./models/prior";

export type {
  ParametricIdResult,
  TRuleResult,
  ParameterClassification,
  ParameterIdentification,
} from "./models/parametric-id";

export type { TraceMessage, TraceUsage, LLMTrace } from "./models/llm-trace";

export type {
  TreatmentEffect,
  PowerScalingResult,
  PowerScalingDiagnosis,
  PPCWarning,
  PPCOverlay,
  PPCTestStat,
  PPCResult,
  InferenceMetadata,
  MCMCParamDiagnostic,
  MCMCDiagnostics,
  SVIDiagnostics,
  TraceData,
  TraceChain,
  RankHistogram,
  RankHistogramChain,
  EnergyHistogram,
  EnergyDiagnostics,
  LOODiagnostics,
  PosteriorMarginal,
  PosteriorPair,
} from "./models/inference";

// Stage data envelope types
export interface StageData<T = unknown> {
  stage: string;
  data: T;
  context: string;
}

export interface Stage0Data {
  source_type: string;
  source_label: string;
  n_records: number;
  date_range: { start: string; end: string };
  sample: Array<Record<string, string | null>>;
}

export interface Stage1aData {
  latent_model: import("./models/construct").LatentModel;
  outcome_name: string;
  treatments: string[];
  graph_properties: {
    is_acyclic: boolean;
    n_constructs: number;
    n_edges: number;
    has_single_outcome: boolean;
  };
  llm_trace?: import("./models/llm-trace").LLMTrace;
}

export interface Stage1bData {
  causal_spec: import("./models/causal-spec").CausalSpec;
  filtered_treatments: string[];
  llm_trace?: import("./models/llm-trace").LLMTrace;
}

export interface Stage2Data {
  workers: import("./models/worker").WorkerStatus[];
  combined_extractions_sample: Array<Record<string, string | null>>;
  total_extractions: number;
  per_indicator_counts: Record<string, number>;
}

export interface Stage3Data {
  validation_report: import("./models/validation").ValidationReport;
}

export interface Stage4Data {
  model_spec: import("./models/model-spec").ModelSpec;
  priors: import("./models/prior").PriorProposal[];
  validation_retries: Array<{
    attempt: number;
    failed_params: string[];
    feedback: string;
  }>;
  ssm_equations: string[];
  llm_trace?: import("./models/llm-trace").LLMTrace;
  prior_predictive_samples?: Record<string, number[]>;
}

export interface Stage4bData {
  parametric_id: import("./models/parametric-id").ParametricIdResult;
}

export interface Stage5Data {
  intervention_results: import("./models/inference").TreatmentEffect[];
  power_scaling: import("./models/inference").PowerScalingResult[];
  ppc: import("./models/inference").PPCResult;
  inference_metadata: import("./models/inference").InferenceMetadata;
  mcmc_diagnostics?: import("./models/inference").MCMCDiagnostics | null;
  svi_diagnostics?: import("./models/inference").SVIDiagnostics | null;
  loo_diagnostics?: import("./models/inference").LOODiagnostics | null;
  posterior_marginals?: import("./models/inference").PosteriorMarginal[] | null;
  posterior_pairs?: import("./models/inference").PosteriorPair[] | null;
}
