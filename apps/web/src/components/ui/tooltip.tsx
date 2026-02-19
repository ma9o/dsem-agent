"use client";

import { cn } from "@/lib/utils/cn";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export function Tooltip({
  children,
  content,
  className,
  triggerClassName,
}: {
  children: ReactNode;
  content: ReactNode;
  className?: string;
  triggerClassName?: string;
}) {
  const [show, setShow] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const handleEnter = useCallback(() => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    // Initial position: centered above trigger
    setCoords({ top: rect.top - 8, left: rect.left + rect.width / 2 });
    setShow(true);
  }, []);

  // After the tooltip renders, measure it and clamp to viewport
  useEffect(() => {
    if (!show || !tooltipRef.current || !triggerRef.current) return;
    const trigger = triggerRef.current.getBoundingClientRect();
    const tip = tooltipRef.current.getBoundingClientRect();
    const pad = 8;

    let left = trigger.left + trigger.width / 2;
    let top = trigger.top - 8;

    // Clamp horizontally using actual tooltip width
    const halfW = tip.width / 2;
    left = Math.max(pad + halfW, Math.min(left, window.innerWidth - pad - halfW));

    // If too close to top, position below
    if (top - tip.height < pad) {
      top = trigger.bottom + 8;
    }

    setCoords({ top, left });
  }, [show]);

  const handleLeave = useCallback(() => {
    setShow(false);
  }, []);

  return (
    <div
      ref={triggerRef}
      className={cn("relative inline-flex", triggerClassName)}
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
    >
      {children}
      {show &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            ref={tooltipRef}
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
