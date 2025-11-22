"use client";

import * as React from "react";

type InputSize = "md" | "lg";

export type InputProps = Omit<React.InputHTMLAttributes<HTMLInputElement>, "size"> & {
  fullWidth?: boolean;
  size?: InputSize;
};

const base =
  "border-2 border-black bg-white font-mono placeholder:opacity-60 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-transform duration-100 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none focus:translate-x-[2px] focus:translate-y-[2px] focus:shadow-none focus:outline-none focus-visible:ring-2 focus-visible:ring-black";

const sizes: Record<InputSize, string> = {
  md: "px-4 py-2 text-sm md:text-base",
  lg: "px-5 py-3 text-base md:text-lg",
};

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = "", fullWidth, size = "lg", ...props }, ref) => {
    const width = fullWidth ? "w-full" : "";
    const sizeClasses = sizes[size];
    return (
      <input
        ref={ref}
        className={[base, sizeClasses, width, className].join(" ")}
        {...props}
      />
    );
  }
);

Input.displayName = "Input";

export default Input;
