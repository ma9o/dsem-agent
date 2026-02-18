"use client";

import { LLMTraceAccordion } from "@/components/ui/custom/llm-trace-accordion";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import type { StageRunStatus, StageTiming } from "@/lib/hooks/use-run-events";
import { useStageData } from "@/lib/hooks/use-stage-data";
import type {
  Stage0Data,
  Stage1aData,
  Stage1bData,
  Stage2Data,
  Stage3Data,
  Stage4Data,
  Stage4bData,
  Stage5Data,
  StageMeta,
} from "@causal-ssm/api-types";
import { Suspense, lazy } from "react";
import { StageSection } from "./stage-section";

const Stage0Content = lazy(() => import("./stage-contents/stage-0-content"));
const Stage1aContent = lazy(() => import("./stage-contents/stage-1a-content"));
const Stage1bContent = lazy(() => import("./stage-contents/stage-1b-content"));
const Stage2Content = lazy(() => import("./stage-contents/stage-2-content"));
const Stage3Content = lazy(() => import("./stage-contents/stage-3-content"));
const Stage4Content = lazy(() => import("./stage-contents/stage-4-content"));
const Stage4bContent = lazy(() => import("./stage-contents/stage-4b-content"));
const Stage5Content = lazy(() => import("./stage-contents/stage-5-content"));

export function StageSectionRouter({
  stage,
  runId,
  status,
  timing,
}: {
  stage: StageMeta;
  runId: string;
  status: StageRunStatus;
  timing?: StageTiming;
}) {
  const isCompleted = status === "completed";
  const elapsedMs =
    timing?.completedAt && timing?.startedAt ? timing.completedAt - timing.startedAt : undefined;

  // Read context from the stage data (shared query key — cache hit when stage content is loaded)
  const { data: stageData } = useStageData<{ context?: string }>(runId, stage.id, isCompleted);

  return (
    <StageSection
      id={stage.id}
      number={stage.number}
      title={stage.label}
      status={status}
      elapsedMs={elapsedMs}
      context={stageData?.context}
      hasGate={stage.hasGate}
      loadingHint={stage.loadingHint}
    >
      {isCompleted && (
        <ErrorBoundary>
          <Suspense fallback={null}>
            <StageContent stageId={stage.id} runId={runId} />
          </Suspense>
        </ErrorBoundary>
      )}
    </StageSection>
  );
}

function StageContent({ stageId, runId }: { stageId: string; runId: string }) {
  switch (stageId) {
    case "stage-0":
      return <Stage0Wrapper runId={runId} />;
    case "stage-1a":
      return <Stage1aWrapper runId={runId} />;
    case "stage-1b":
      return <Stage1bWrapper runId={runId} />;
    case "stage-2":
      return <Stage2Wrapper runId={runId} />;
    case "stage-3":
      return <Stage3Wrapper runId={runId} />;
    case "stage-4":
      return <Stage4Wrapper runId={runId} />;
    case "stage-4b":
      return <Stage4bWrapper runId={runId} />;
    case "stage-5":
      return <Stage5Wrapper runId={runId} />;
    default:
      return null;
  }
}

function Stage0Wrapper({ runId }: { runId: string }) {
  const { data } = useStageData<Stage0Data>(runId, "stage-0", true);
  if (!data) return null;
  return <Stage0Content data={data} />;
}

function Stage1aWrapper({ runId }: { runId: string }) {
  const { data } = useStageData<Stage1aData>(runId, "stage-1a", true);
  if (!data) return null;
  return (
    <>
      <Stage1aContent data={data} />
      {data.raw_completion && <LLMTraceAccordion rawCompletion={data.raw_completion} />}
    </>
  );
}

function Stage1bWrapper({ runId }: { runId: string }) {
  const { data } = useStageData<Stage1bData>(runId, "stage-1b", true);
  if (!data) return null;
  return (
    <>
      <Stage1bContent data={data} />
      {data.raw_completion && <LLMTraceAccordion rawCompletion={data.raw_completion} />}
    </>
  );
}

function Stage2Wrapper({ runId }: { runId: string }) {
  const { data } = useStageData<Stage2Data>(runId, "stage-2", true);
  if (!data) return null;
  return <Stage2Content data={data} />;
}

function Stage3Wrapper({ runId }: { runId: string }) {
  const { data } = useStageData<Stage3Data>(runId, "stage-3", true);
  if (!data) return null;
  return <Stage3Content data={data} />;
}

function Stage4Wrapper({ runId }: { runId: string }) {
  const { data } = useStageData<Stage4Data>(runId, "stage-4", true);
  const { data: stage2 } = useStageData<Stage2Data>(runId, "stage-2", true);
  if (!data) return null;
  return (
    <>
      <Stage4Content data={data} extractions={stage2?.combined_extractions_sample} />
      {data.raw_completion && <LLMTraceAccordion rawCompletion={data.raw_completion} />}
    </>
  );
}

function Stage4bWrapper({ runId }: { runId: string }) {
  const { data } = useStageData<Stage4bData>(runId, "stage-4b", true);
  if (!data) return null;
  return <Stage4bContent data={data} />;
}

function Stage5Wrapper({ runId }: { runId: string }) {
  const { data } = useStageData<Stage5Data>(runId, "stage-5", true);
  if (!data) return null;
  return <Stage5Content data={data} />;
}
