export interface Extraction {
  indicator: string;
  value: number | boolean | string | null;
  timestamp: string | null;
}

export interface WorkerOutput {
  extractions: Extraction[];
}

export interface WorkerStatus {
  worker_id: number;
  status: "pending" | "running" | "completed" | "failed";
  n_extractions: number;
  chunk_size: number;
  error?: string;
}
