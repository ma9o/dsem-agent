"use client";

import { cn } from "@/lib/utils/cn";
import { type ReactNode, useState } from "react";

export function Tooltip({
  children,
  content,
  className,
}: {
  children: ReactNode;
  content: ReactNode;
  className?: string;
}) {
  const [show, setShow] = useState(false);
  return (
    <div
      className="relative inline-flex"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div
          className={cn(
            "absolute bottom-full left-1/2 z-50 mb-2 -translate-x-1/2 rounded-md bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md border animate-in fade-in-0 zoom-in-95",
            className,
          )}
        >
          {content}
        </div>
      )}
    </div>
  );
}
