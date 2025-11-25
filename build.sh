#!/usr/bin/env bash
# Build script for Render deployment

set -e

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Attempting to build frontend (optional)..."
if command -v npm >/dev/null 2>&1; then
  echo "npm detected. Building Next.js frontend."
  (
    cd frontend || exit 0
    # Use npm install to avoid strict lockfile parity errors (EUSAGE) on Render
    # where package-lock may not include platform-specific optional deps (e.g., sharp).
    if command -v npm >/dev/null 2>&1; then
      npm install --no-audit --no-fund
      # Build the static export. With next.config.js using output: 'export',
      # `next build` will produce the static files in `frontend/out`.
      npm run build || {
        echo "Warning: 'npm run build' failed. Skipping frontend build but continuing backend build.";
        exit 0;
      }
    fi
  )
else
  echo "npm not found; skipping frontend build."
fi

echo "Build completed successfully!"
