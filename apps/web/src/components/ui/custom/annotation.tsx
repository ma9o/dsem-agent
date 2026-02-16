import { cn } from "@/lib/utils/cn";
import { BookOpen } from "lucide-react";

export function Annotation({
  content,
  className,
}: {
  content: string;
  className?: string;
}) {
  if (!content) return null;
  return (
    <div
      className={cn(
        "rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm text-blue-900 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-200",
        className,
      )}
    >
      <div className="mb-2 flex items-center gap-2 font-medium">
        <BookOpen className="h-4 w-4" />
        Context
      </div>
      <div className="whitespace-pre-wrap leading-relaxed">{content}</div>
    </div>
  );
}
