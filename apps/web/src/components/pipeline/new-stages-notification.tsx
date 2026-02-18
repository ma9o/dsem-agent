"use client";

import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import type { StageId } from "@causal-ssm/api-types";
import { ArrowDown } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Fixed bottom notification for stages that completed while the user
 * wasn't scrolled down to them. A stage is "unseen" until the user
 * scrolls it into the viewport at least once. Scrolling back up does
 * NOT re-add already-seen stages.
 */
export function NewStagesNotification({
  progress,
}: {
  progress: PipelineProgress;
}) {
  // Stages the user has scrolled past (seen) at least once
  const seenRef = useRef<Set<StageId>>(new Set());
  const [unseenIds, setUnseenIds] = useState<StageId[]>([]);

  // When progress changes, check for newly completed stages below viewport
  useEffect(() => {
    const newUnseen: StageId[] = [];
    for (const s of STAGES) {
      const status = progress.stages[s.id];
      if (status !== "completed" && status !== "failed") continue;
      if (seenRef.current.has(s.id)) continue;

      const el = document.getElementById(s.id);
      if (!el) {
        // Not rendered yet — unseen
        newUnseen.push(s.id);
        continue;
      }
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight) {
        // Already visible — mark as seen immediately
        seenRef.current.add(s.id);
      } else {
        newUnseen.push(s.id);
      }
    }
    setUnseenIds(newUnseen);
  }, [progress]);

  // On scroll, check if any unseen stages have entered the viewport
  useEffect(() => {
    if (unseenIds.length === 0) return;

    const handler = () => {
      let changed = false;
      for (const id of unseenIds) {
        const el = document.getElementById(id);
        if (!el) continue;
        const rect = el.getBoundingClientRect();
        if (rect.top < window.innerHeight) {
          seenRef.current.add(id);
          changed = true;
        }
      }
      if (changed) {
        setUnseenIds((prev) => prev.filter((id) => !seenRef.current.has(id)));
      }
    };

    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, [unseenIds]);

  const scrollToNext = useCallback(() => {
    if (unseenIds.length === 0) return;
    const el = document.getElementById(unseenIds[0]);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [unseenIds]);

  if (unseenIds.length === 0) return null;

  const next = STAGES.find((s) => s.id === unseenIds[0])!;
  const label =
    unseenIds.length === 1
      ? `Stage ${next.number}: ${next.label} completed`
      : `${unseenIds.length} new stages completed`;

  return (
    <button
      type="button"
      onClick={scrollToNext}
      className="animate-fade-in-up fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 items-center gap-2 rounded-full border bg-background px-4 py-2.5 shadow-lg transition-colors hover:bg-secondary"
    >
      <ArrowDown className="h-4 w-4" />
      <span className="text-sm font-medium">{label}</span>
    </button>
  );
}
