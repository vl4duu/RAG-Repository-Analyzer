import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Enable App Router (default in Next 13+ with app directory present)
  experimental: {
    // Keep defaults minimal; can be extended later if needed
  },
};

export default nextConfig;
