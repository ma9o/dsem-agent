export interface TraceMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  reasoning?: string | null;
  tool_calls?: Array<{ name: string; arguments: Record<string, unknown> }> | null;
  tool_name?: string | null;
  tool_result?: string | null;
  tool_is_error?: boolean;
}

export interface TraceUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  reasoning_tokens?: number | null;
}

export interface LLMTrace {
  messages: TraceMessage[];
  model: string;
  total_time_seconds: number;
  usage: TraceUsage;
}
