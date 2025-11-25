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
      # Standard Next.js build
      npm run build || {
        echo "Warning: 'npm run build' failed. Skipping frontend export.";
        exit 0;
      }
      # Try to export static site (Next.js). If not configured, skip silently.
      if npm run | grep -q "export"; then
        npm run export || echo "Warning: 'npm run export' failed. Ensure next export is supported if you want to serve static files from backend."
      else
        echo "No 'export' script found. If you want to serve frontend statically from backend, add an 'export' script (next export)."
      fi
    fi
  )
else
  echo "npm not found; skipping frontend build."
fi

echo "Build completed successfully!"
