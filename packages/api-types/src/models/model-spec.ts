export type DistributionFamily =
  | "gaussian"
  | "student_t"
  | "poisson"
  | "gamma"
  | "bernoulli"
  | "negative_binomial"
  | "beta"
  | "ordered_logistic"
  | "categorical";

export type LinkFunction = "identity" | "log" | "logit" | "probit" | "cumulative_logit" | "softmax";

export type ParameterRole =
  | "fixed_effect"
  | "ar_coefficient"
  | "residual_sd"
  | "correlation"
  | "loading";

export type ParameterConstraint = "none" | "positive" | "unit_interval" | "correlation";

export interface LikelihoodSpec {
  variable: string;
  distribution: DistributionFamily;
  link: LinkFunction;
  reasoning: string;
}

export interface ParameterSpec {
  name: string;
  role: ParameterRole;
  constraint: ParameterConstraint;
  description: string;
  search_context: string;
}

export interface ModelSpec {
  likelihoods: LikelihoodSpec[];
  parameters: ParameterSpec[];
  reasoning: string;
}
