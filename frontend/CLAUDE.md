# frontend/ — Next.js UI

Next.js 13+ App Router, single page (`app/page.tsx`) driving a three-phase flow: `input` (repo URL) →
`processing` (indexing progress) → `dashboard` (tabs: Files / Dependencies / Complexity / Chat). Built
as a **static export** (`output: 'export'` in `next.config.js`) — there is no Next.js server at runtime;
`next build` produces `frontend/out`, which is either deployed as its own Render static site or served
directly by the FastAPI backend (see `backend/CLAUDE.md`). Because of static export, don't use features
that require a Node server (API routes, SSR data fetching, middleware) — everything must work as
client-side fetches from a prebuilt HTML/JS bundle.

## API URL resolution

There's no server-side env injection at request time — `NEXT_PUBLIC_API_URL` is baked in at **build**
time. `page.tsx`'s `sanitizeApiUrl()` validates it (rejects empty/`"undefined"`/`"null"`/malformed
values — these are exactly what a broken build-time env substitution produces) and, only when running on
`localhost`/`127.0.0.1`, falls back to `http://localhost:8000` for local dev convenience. In a deployed
build with `NEXT_PUBLIC_API_URL` unset, `apiMisconfigured` becomes true and a warning is logged — this
is intentional user-facing signal, not swallowed silently, so preserve it if you touch this logic.

`next.config.js` sets `trailingSlash: true` for static-host friendliness — keep any new routes/links
consistent with that.

## Components

`components/ui/` holds small primitives (`button.tsx`, `input.tsx`) styled with Tailwind +
`class-variance-authority`/`tailwind-merge`. There's no component library beyond this — new UI pieces
should follow the same minimal cva-based pattern rather than introducing a new one.

## Commands

```bash
cd frontend
npm install
npm run dev       # http://localhost:3000, expects backend on :8000
npm run build     # static export to frontend/out
```

No test suite or linter is wired up for the frontend currently.
