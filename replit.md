# PDF Assistant

A legal-grade PDF editing assistant. Upload up to 14+ PDFs, then chat with AI in plain English to merge, split, delete pages, rotate, highlight, and combine them into a single final document — all processed entirely on the server. No document content ever leaves your environment.

## Run & Operate

- `bash artifacts/api-server/start.sh` — run the PDF Assistant (port 8080, served at `/`)
- The managed workflow `artifacts/api-server: API Server` runs this automatically

## Stack

- **Backend**: FastAPI (Python 3.11) + PyMuPDF (`fitz`) for all PDF operations
- **Frontend**: Single-page app — HTML5, Tailwind CSS (CDN), Vanilla JS (no build step)
- **AI**: Replit AI Integration (OpenAI-compatible) — deterministic command parser only
- **Session storage**: `/tmp/pdf_sessions/<uuid>/` — auto-purged after 2 hours idle

## Where things live

- `artifacts/api-server/app/main.py` — FastAPI routes (upload, operate, chat, download, undo, reset)
- `artifacts/api-server/app/pdf_ops.py` — all PyMuPDF operations (merge, delete, split, rotate, highlight, reorder, undo)
- `artifacts/api-server/app/ai_parser.py` — AI command parser with hard security boundary
- `artifacts/api-server/app/session_mgr.py` — session isolation, filename sanitization, expiry cleanup
- `artifacts/api-server/app/static/index.html` — full SPA (UI + vanilla JS)

## Security Model

- **Hard AI boundary**: AI receives only `{filename, page_count}` metadata + the instruction text. No PDF bytes, no extracted text, no rendered images ever reach the LLM. Server-side assertion enforces this before every AI call.
- **Path traversal guard**: all filenames are sanitized via `sanitize_filename()` before any filesystem operation; validated against session dir with `.resolve()` comparison.
- **Session isolation**: each user gets a UUID-keyed directory under `/tmp/pdf_sessions/`. Files deleted on explicit reset or 2-hour idle timeout.
- **AI schema validation**: the AI's JSON output is validated against a strict action schema; unknown action types or malformed JSON are rejected with a user-facing error before any execution.

## Architecture decisions

- FastAPI + PyMuPDF chosen over Node/pdf-lib: fitz is C-backed, handles complex legal PDFs reliably, and covers thumbnails, annotations, and all manipulation in one native dependency.
- No React/Vite/codegen: vanilla JS SPA with no build step keeps the stack lean and auditable.
- Single service serves both the SPA and API to eliminate cross-origin complexity.
- Thumbnail cache (`_thumb_<stem>_p<n>.png`) is invalidated on every write operation to keep previews accurate.
- One-step undo: every destructive op saves `<name>_bak.pdf` before executing; undo swaps it back.

## User preferences

_Populate as needed._

## Gotchas

- Do not run `pnpm run dev` — the app is Python/uvicorn, not Node. Use the workflow or `bash artifacts/api-server/start.sh`.
- `SESSION_SECRET` env var must be set for session signing (already configured as a Replit secret).
- `AI_INTEGRATIONS_OPENAI_BASE_URL` and `AI_INTEGRATIONS_OPENAI_API_KEY` are auto-set by Replit AI Integration — do not set manually.
- fitz warns about deprecated `import fitz` — use `import pymupdf` in future; functional either way.
- Thumbnail generation for very large PDFs (100+ pages) is lazy-loaded per page to avoid blocking startup.
