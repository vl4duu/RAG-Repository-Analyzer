"use client";

import * as React from "react";

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "solid" | "ghost" | "outline";
  accent?: "orange" | "neutral";
  fullWidth?: boolean;
};

const base =
  "inline-flex items-center justify-center select-none border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-transform duration-100 active:translate-x-[2px] active:translate-y-[2px] active:shadow-none focus:outline-none focus-visible:ring-2 focus-visible:ring-black disabled:opacity-60 disabled:cursor-not-allowed font-semibold tracking-tight";

const size = "px-4 py-2 text-sm md:text-base";

const palette = {
  orange: {
    solid:
      "bg-amber-200 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none",
    ghost:
      "bg-transparent hover:bg-amber-100 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none",
    outline:
      "bg-white hover:bg-amber-50 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none",
  },
  neutral: {
    solid:
      "bg-white hover:bg-neutral-100 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none",
    ghost:
      "bg-transparent hover:bg-neutral-100 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none",
    outline:
      "bg-white hover:bg-neutral-100 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none",
  },
} as const;

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className = "",
      variant = "solid",
      accent = "orange",
      fullWidth,
      children,
      ...props
    },
    ref
  ) => {
    const paletteClasses = palette[accent][variant];
    const width = fullWidth ? "w-full" : "";
    return (
      <button
        ref={ref}
        className={[base, size, paletteClasses, width, className].join(" ")}
        {...props}
      >
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";

export default Button;
