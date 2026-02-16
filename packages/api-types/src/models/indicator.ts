export type MeasurementGranularity =
  | "finest"
  | "hourly"
  | "daily"
  | "weekly"
  | "monthly"
  | "yearly";

export type MeasurementDtype = "continuous" | "binary" | "count" | "ordinal" | "categorical";

export type AggregationFunction =
  | "mean"
  | "sum"
  | "min"
  | "max"
  | "std"
  | "var"
  | "last"
  | "first"
  | "count"
  | "median"
  | "p10"
  | "p25"
  | "p75"
  | "p90"
  | "p99"
  | "skew"
  | "kurtosis"
  | "iqr"
  | "range"
  | "cv"
  | "entropy"
  | "instability"
  | "trend"
  | "n_unique";

export interface Indicator {
  name: string;
  construct_name: string;
  how_to_measure: string;
  measurement_granularity: MeasurementGranularity;
  measurement_dtype: MeasurementDtype;
  aggregation: AggregationFunction;
  ordinal_levels: string[] | null;
}

export interface MeasurementModel {
  indicators: Indicator[];
}
