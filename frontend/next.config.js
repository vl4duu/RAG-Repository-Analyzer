/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static export so `next build` outputs to `out/` without needing `next export`.
  output: 'export',
  // Disable image optimization since we're exporting a static site.
  images: { unoptimized: true },
  // Ensure all routes resolve with trailing slash which is friendlier for static hosting.
  trailingSlash: true,
};

module.exports = nextConfig;
