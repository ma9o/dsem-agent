import type { LatentModel } from "./construct";
import type { MeasurementModel } from "./indicator";

export interface IdentifiedTreatmentStatus {
  method: string;
  estimand: string;
  marginalized_confounders: string[];
  instruments: string[];
}

export interface NonIdentifiableTreatmentStatus {
  confounders: string[];
  notes: string | null;
}

export interface IdentifiabilityStatus {
  identifiable_treatments: Record<string, IdentifiedTreatmentStatus>;
  non_identifiable_treatments: Record<string, NonIdentifiableTreatmentStatus>;
}

export interface CausalSpec {
  latent: LatentModel;
  measurement: MeasurementModel;
  identifiability: IdentifiabilityStatus | null;
}
