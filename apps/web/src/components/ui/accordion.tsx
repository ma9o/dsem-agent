"use client";

import { cn } from "@/lib/utils/cn";
import { ChevronDown } from "lucide-react";
import { type HTMLAttributes, type ReactNode, createContext, useContext, useState } from "react";

const AccordionContext = createContext<{
  openItems: Set<string>;
  toggle: (value: string) => void;
}>({ openItems: new Set(), toggle: () => {} });

export function Accordion({
  children,
  className,
  defaultOpen,
  ...props
}: HTMLAttributes<HTMLDivElement> & { defaultOpen?: string[] }) {
  const [openItems, setOpenItems] = useState<Set<string>>(new Set(defaultOpen));
  const toggle = (value: string) => {
    setOpenItems((prev) => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
  };
  return (
    <AccordionContext.Provider value={{ openItems, toggle }}>
      <div className={cn("divide-y", className)} {...props}>
        {children}
      </div>
    </AccordionContext.Provider>
  );
}

export function AccordionItem({
  value,
  children,
  className,
}: {
  value: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("border-b", className)} data-value={value}>
      {children}
    </div>
  );
}

export function AccordionTrigger({
  children,
  value,
  className,
}: {
  children: ReactNode;
  value: string;
  className?: string;
}) {
  const { openItems, toggle } = useContext(AccordionContext);
  const isOpen = openItems.has(value);
  return (
    <button
      type="button"
      className={cn(
        "flex w-full items-center justify-between py-4 font-medium transition-all hover:underline",
        className,
      )}
      onClick={() => toggle(value)}
      aria-expanded={isOpen}
    >
      {children}
      <ChevronDown
        className={cn("h-4 w-4 shrink-0 transition-transform duration-200", isOpen && "rotate-180")}
      />
    </button>
  );
}

export function AccordionContent({
  children,
  value,
  className,
}: {
  children: ReactNode;
  value: string;
  className?: string;
}) {
  const { openItems } = useContext(AccordionContext);
  const isOpen = openItems.has(value);
  if (!isOpen) return null;
  return <div className={cn("pb-4 pt-0 text-sm", className)}>{children}</div>;
}
