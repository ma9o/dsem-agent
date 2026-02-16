export const STAGE_IDS = [
  "stage-0",
  "stage-1a",
  "stage-1b",
  "stage-2",
  "stage-3",
  "stage-4",
  "stage-4b",
  "stage-5",
] as const;

export type StageId = (typeof STAGE_IDS)[number];

export interface StageMeta {
  id: StageId;
  label: string;
  number: string;
  hasGate: boolean;
  prefectTaskName: string;
}

export const STAGES: StageMeta[] = [
  {
    id: "stage-0",
    label: "Preprocess",
    number: "0",
    hasGate: false,
    prefectTaskName: "preprocess_raw_input",
  },
  {
    id: "stage-1a",
    label: "Latent Model",
    number: "1a",
    hasGate: false,
    prefectTaskName: "propose_latent_model",
  },
  {
    id: "stage-1b",
    label: "Measurement & Identifiability",
    number: "1b",
    hasGate: true,
    prefectTaskName: "propose_measurement_with_identifiability_fix",
  },
  {
    id: "stage-2",
    label: "Data Extraction",
    number: "2",
    hasGate: false,
    prefectTaskName: "populate_indicators",
  },
  {
    id: "stage-3",
    label: "Validation",
    number: "3",
    hasGate: true,
    prefectTaskName: "validate_extraction",
  },
  {
    id: "stage-4",
    label: "Model Specification",
    number: "4",
    hasGate: false,
    prefectTaskName: "stage4_orchestrated_flow",
  },
  {
    id: "stage-4b",
    label: "Parametric Identifiability",
    number: "4b",
    hasGate: true,
    prefectTaskName: "stage4b_parametric_id_flow",
  },
  {
    id: "stage-5",
    label: "Inference & Results",
    number: "5",
    hasGate: false,
    prefectTaskName: "fit_model",
  },
];
