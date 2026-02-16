"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Bot } from "lucide-react";

export function LLMTraceAccordion({
  rawCompletion,
}: {
  rawCompletion: string;
}) {
  if (!rawCompletion) return null;
  return (
    <Accordion>
      <AccordionItem value="llm-trace">
        <AccordionTrigger value="llm-trace" className="text-sm text-muted-foreground">
          <span className="flex items-center gap-2">
            <Bot className="h-4 w-4" />
            Raw LLM Trace
          </span>
        </AccordionTrigger>
        <AccordionContent value="llm-trace">
          <pre className="max-h-96 overflow-auto rounded-md bg-muted p-4 text-xs font-mono whitespace-pre-wrap">
            {rawCompletion}
          </pre>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
