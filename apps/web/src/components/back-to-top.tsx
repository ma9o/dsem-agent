"use client";

import { ArrowUp } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

export function BackToTop() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const handler = () => {
      setVisible(window.scrollY > 400);
    };
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  const scrollToTop = useCallback(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  if (!visible) return null;

  return (
    <button
      type="button"
      onClick={scrollToTop}
      className="animate-fade-in fixed bottom-6 right-6 z-50 flex h-10 w-10 items-center justify-center rounded-full border bg-background shadow-lg transition-colors hover:bg-secondary"
      aria-label="Back to top"
    >
      <ArrowUp className="h-4 w-4" />
    </button>
  );
}
