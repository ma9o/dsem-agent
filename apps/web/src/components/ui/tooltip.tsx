"use client";

import { cn } from "@/lib/utils/cn";
import { type ReactNode, useCallback, useRef, useState } from "react";
import { createPortal } from "react-dom";

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
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const handleEnter = useCallback(() => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    // Position above center by default
    let left = rect.left + rect.width / 2;
    let top = rect.top - 8;

    // Clamp to viewport
    const padding = 8;
    left = Math.max(padding + 80, Math.min(left, window.innerWidth - padding - 80));

    // If too close to top, position below
    if (top < 60) {
      top = rect.bottom + 8;
    }

    setCoords({ top, left });
    setShow(true);
  }, []);

  const handleLeave = useCallback(() => {
    setShow(false);
  }, []);

  return (
    <div
      ref={triggerRef}
      className="relative inline-flex"
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
      onFocus={handleEnter}
      onBlur={handleLeave}
    >
      {children}
      {show &&
        coords &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            ref={tooltipRef}
            role="tooltip"
            className={cn(
              "fixed z-[100] -translate-x-1/2 -translate-y-full rounded-md bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md border animate-fade-in pointer-events-none",
              className,
            )}
            style={{ top: coords.top, left: coords.left }}
          >
            {content}
          </div>,
          document.body,
        )}
    </div>
  );
}
