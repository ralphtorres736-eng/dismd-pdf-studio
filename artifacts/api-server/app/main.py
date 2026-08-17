"""
PDF Assistant — FastAPI Application
Serves the SPA and all PDF operation API endpoints.
"""
import io
import os
import shutil
import uuid
import zipfile
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .session_mgr import (
    destroy_session, get_session_dir, list_session_files,
    sanitize_filename, cleanup_expired_sessions,
)
from .pdf_ops import (
    add_page_numbers, apply_sticker, check_has_text, delete_pages,
    get_all_page_counts, get_page_count, get_thumbnail, highlight_page,
    merge_pdfs, ocr_pdf, redact_pdf, reorder_pages, rotate_pages,
    split_pdf, undo_last_op,
)
from .ai_parser import parse_instruction

SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-me")
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB per file

# In-memory OCR job tracker: {job_id: {status, current_page, total_pages, output, error, session_dir}}
_ocr_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Lifespan — background cleanup task
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(600)  # run every 10 minutes
            await asyncio.to_thread(cleanup_expired_sessions)

    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(title="PDF Assistant", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="pdfassist_sid",
    max_age=7200,
    https_only=False,
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _get_sid(request: Request) -> str:
    """Return existing session ID or create a new one."""
    sid = request.session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        request.session["sid"] = sid
    return sid


def _session_dir(request: Request) -> Path:
    return get_session_dir(_get_sid(request))


def _file_metadata(session_dir: Path) -> list[dict]:
    """Build the file list with page counts for the current session."""
    files = list_session_files(session_dir)
    counts = get_all_page_counts(session_dir)
    return [
        {"name": f["name"], "pages": counts.get(f["name"], 0), "size_bytes": f["size_bytes"]}
        for f in files
    ]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    session_dir = _session_dir(request)
    saved = []
    errors = []

    for upload in files:
        raw_name = upload.filename or "document.pdf"
        name = sanitize_filename(raw_name)
        dest = session_dir / name

        # Resolve conflicts with a numeric suffix
        stem = Path(name).stem
        counter = 1
        while dest.exists():
            name = f"{stem}_{counter}.pdf"
            dest = session_dir / name
            counter += 1

        content = await upload.read()
        if len(content) > MAX_UPLOAD_SIZE:
            errors.append({"name": raw_name, "error": "File exceeds 500 MB limit."})
            continue
        if not content.startswith(b"%PDF"):
            errors.append({"name": raw_name, "error": "Not a valid PDF file."})
            continue

        dest.write_bytes(content)
        saved.append(name)

    return JSONResponse({
        "saved": saved,
        "errors": errors,
        "files": _file_metadata(session_dir),
    })


# ---------------------------------------------------------------------------
# File list
# ---------------------------------------------------------------------------

@app.get("/api/files")
async def list_files(request: Request):
    session_dir = _session_dir(request)
    return JSONResponse({"files": _file_metadata(session_dir)})


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------

@app.get("/api/thumbnail/{filename}/{page}")
async def get_page_thumbnail(request: Request, filename: str, page: int):
    session_dir = _session_dir(request)
    try:
        png = await asyncio.to_thread(get_thumbnail, session_dir, filename, page)
    except FileNotFoundError:
        raise HTTPException(404, f"File '{filename}' not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Thumbnail render failed: {exc}")

    return Response(content=png, media_type="image/png")


# ---------------------------------------------------------------------------
# Direct operations (manual UI controls)
# ---------------------------------------------------------------------------

@app.post("/api/operate")
async def apply_operation(request: Request):
    """
    Execute a single validated operation from the manual UI.
    Body: {"action": {...}} — same action object schema as used by the AI parser.
    """
    body = await request.json()
    action = body.get("action")
    if not action or "type" not in action:
        raise HTTPException(400, "Request body must contain an 'action' object with a 'type' field.")

    session_dir = _session_dir(request)

    try:
        result = await asyncio.to_thread(_execute_action, session_dir, action)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Operation failed: {exc}")

    return JSONResponse({"result": result, "files": _file_metadata(session_dir)})


# ---------------------------------------------------------------------------
# AI chat — parse instruction then execute
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_instruction(request: Request):
    """
    Accept a plain-English instruction, parse it via AI (structural metadata only),
    execute the resulting action plan, and return the AI summary + updated file list.
    """
    body = await request.json()
    instruction = (body.get("instruction") or "").strip()
    if not instruction:
        raise HTTPException(400, "Instruction must not be empty.")

    session_dir = _session_dir(request)

    # Build metadata — ONLY names and page counts; never text content
    files = _file_metadata(session_dir)
    if not files:
        raise HTTPException(400, "No files uploaded yet. Please upload PDFs first.")

    doc_metadata = [{"name": f["name"], "pages": f["pages"]} for f in files]

    # Parse instruction via AI (security boundary enforced inside parse_instruction)
    try:
        plan = await asyncio.to_thread(parse_instruction, instruction, doc_metadata)
    except AssertionError as exc:
        raise HTTPException(500, f"Security boundary violation: {exc}")
    except ValueError as exc:
        raise HTTPException(422, f"AI parsing error: {exc}")
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))

    # Execute each action in the plan
    execution_results = []
    errors = []
    for action in plan["actions"]:
        try:
            res = await asyncio.to_thread(_execute_action, session_dir, action)
            execution_results.append({"action": action["type"], "result": res})
        except Exception as exc:
            errors.append({"action": action.get("type"), "error": str(exc)})

    return JSONResponse({
        "summary": plan["summary"],
        "actions_executed": execution_results,
        "errors": errors,
        "files": _file_metadata(session_dir),
    })


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

@app.post("/api/undo/{filename}")
async def undo_operation(request: Request, filename: str):
    session_dir = _session_dir(request)
    try:
        restored = await asyncio.to_thread(undo_last_op, session_dir, filename)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc))

    if not restored:
        raise HTTPException(404, f"No backup found for '{filename}'.")

    return JSONResponse({"restored": filename, "files": _file_metadata(session_dir)})


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@app.get("/api/download/{filename}")
async def download_file(request: Request, filename: str):
    session_dir = _session_dir(request)
    safe_name = sanitize_filename(filename)
    path = session_dir / safe_name
    if not path.exists():
        raise HTTPException(404, f"File '{safe_name}' not found.")
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=safe_name,
    )


@app.get("/api/download-all")
async def download_all(request: Request):
    session_dir = _session_dir(request)
    files = list_session_files(session_dir)
    if not files:
        raise HTTPException(404, "No files in session.")

    if len(files) == 1:
        # Single file — return directly
        name = files[0]["name"]
        return FileResponse(
            path=str(session_dir / name),
            media_type="application/pdf",
            filename=name,
        )

    # Multiple files — zip them
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(str(session_dir / f["name"]), arcname=f["name"])
    zip_buffer.seek(0)

    return StreamingResponse(
        content=iter([zip_buffer.read()]),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=pdf_documents.zip"},
    )


# ---------------------------------------------------------------------------
# OCR — background job with progress polling
# ---------------------------------------------------------------------------

@app.post("/api/ocr/{filename}")
async def start_ocr(request: Request, filename: str):
    """
    Start an OCR job for the given file. Returns immediately with a job_id.
    Poll /api/ocr-status/{job_id} for progress.
    """
    session_dir = _session_dir(request)

    # Validate file exists before spawning the job
    from .pdf_ops import _validate_file
    try:
        _validate_file(session_dir, filename)
    except FileNotFoundError:
        raise HTTPException(404, f"File '{filename}' not found.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    has_text = await asyncio.to_thread(check_has_text, session_dir, filename)
    total_pages = await asyncio.to_thread(
        lambda: __import__('fitz').open(str(session_dir / filename)).page_count
    )

    job_id = str(uuid.uuid4())
    _ocr_jobs[job_id] = {
        "status": "running",
        "current_page": 0,
        "total_pages": total_pages,
        "output": None,
        "error": None,
        "session_dir": session_dir,
    }

    def _progress(current: int, total: int):
        if job_id in _ocr_jobs:
            _ocr_jobs[job_id]["current_page"] = current

    async def _run_ocr():
        try:
            result = await asyncio.to_thread(ocr_pdf, session_dir, filename, _progress)
            if job_id in _ocr_jobs:
                _ocr_jobs[job_id].update({"status": "done", "output": result, "current_page": total_pages})
        except Exception as exc:
            if job_id in _ocr_jobs:
                _ocr_jobs[job_id].update({"status": "error", "error": str(exc)})

    asyncio.create_task(_run_ocr())

    return JSONResponse({
        "job_id": job_id,
        "has_text": has_text,
        "total_pages": total_pages,
    })


@app.get("/api/ocr-status/{job_id}")
async def ocr_status(request: Request, job_id: str):
    """Return the current progress of an OCR job."""
    job = _ocr_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "OCR job not found.")

    response: dict = {
        "status": job["status"],
        "current_page": job["current_page"],
        "total_pages": job["total_pages"],
        "output": job["output"],
        "error": job["error"],
    }
    if job["status"] == "done":
        response["files"] = _file_metadata(job["session_dir"])
    return JSONResponse(response)


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------

@app.post("/api/reset")
async def reset_session(request: Request):
    sid = _get_sid(request)
    await asyncio.to_thread(destroy_session, sid)
    # Issue a new session ID
    new_sid = str(uuid.uuid4())
    request.session["sid"] = new_sid
    return JSONResponse({"message": "Session cleared.", "files": []})


# ---------------------------------------------------------------------------
# Internal: action executor
# ---------------------------------------------------------------------------

def _execute_action(session_dir: Path, action: dict) -> str:
    """
    Dispatch a single action dict to the appropriate pdf_ops function.
    Returns a human-readable result string.
    Raises ValueError / FileNotFoundError on bad inputs.
    """
    atype = action.get("type", "").upper()

    if atype == "MERGE":
        files = action.get("files", [])
        output = sanitize_filename(action.get("output", "merged.pdf"))
        cover_page = action.get("cover_page")  # None when not requested
        if len(files) < 1:
            raise ValueError("MERGE requires at least one file.")
        out = merge_pdfs(session_dir, files, output, cover_page=cover_page)
        cover_note = " (with cover page)" if cover_page else ""
        return f"Merged {len(files)} file(s){cover_note} → {out}"

    elif atype == "DELETE_PAGES":
        fname = action.get("file", "")
        pages = [int(p) for p in action.get("pages", [])]
        if not fname:
            raise ValueError("DELETE_PAGES requires 'file'.")
        delete_pages(session_dir, fname, pages)
        return f"Deleted pages {pages} from {fname}"

    elif atype == "SPLIT":
        fname = action.get("file", "")
        start = int(action.get("start_page", 1))
        end = int(action.get("end_page", 1))
        output = sanitize_filename(action.get("output", "split.pdf"))
        split_pdf(session_dir, fname, start, end, output)
        return f"Split pages {start}–{end} of {fname} → {output}"

    elif atype == "ROTATE":
        fname = action.get("file", "")
        pages = [int(p) for p in action.get("pages", [])]
        degrees = int(action.get("degrees", 90))
        rotate_pages(session_dir, fname, pages, degrees)
        label = "all pages" if not pages else f"pages {pages}"
        return f"Rotated {label} of {fname} by {degrees}°"

    elif atype == "REORDER":
        fname = action.get("file", "")
        new_order = [int(p) for p in action.get("new_order", [])]
        reorder_pages(session_dir, fname, new_order)
        return f"Reordered pages of {fname}"

    elif atype == "HIGHLIGHT":
        fname = action.get("file", "")
        page = int(action.get("page", 1))
        rect = [float(x) for x in action.get("rect", [50, 50, 500, 80])]
        color = [float(c) for c in action.get("color", [1.0, 0.9, 0.0])]
        label = str(action.get("label", ""))
        highlight_page(session_dir, fname, page, rect, color, label)
        return f"Added highlight on page {page} of {fname}"

    elif atype == "RENAME":
        fname = action.get("file", "")
        new_name = sanitize_filename(action.get("new_name", fname))
        src = session_dir / fname
        dst = session_dir / new_name
        if not src.exists():
            raise FileNotFoundError(f"File '{fname}' not found.")
        shutil.move(str(src), str(dst))
        # Move backup too if it exists
        src_bak = session_dir / (Path(fname).stem + "_bak.pdf")
        if src_bak.exists():
            shutil.move(str(src_bak), str(session_dir / (Path(new_name).stem + "_bak.pdf")))
        return f"Renamed {fname} → {new_name}"

    elif atype == "REDACT":
        fname = action.get("file", "")
        terms = [str(t) for t in action.get("terms", []) if str(t).strip()]
        if not fname:
            raise ValueError("REDACT requires 'file'.")
        if not terms:
            raise ValueError("REDACT requires at least one non-empty term.")
        output, hits = redact_pdf(session_dir, fname, terms)
        return f"Redacted {hits} occurrence(s) of {len(terms)} term(s) in {fname} → {output}"

    elif atype == "ADD_STICKER":
        fname = action.get("file", "")
        if not fname:
            raise ValueError("ADD_STICKER requires 'file'.")
        page_numbers = [int(p) for p in action.get("page_numbers", [1])] or [1]
        sticker_config = {
            "category": str(action.get("category", "legal_exhibit")),
            "preset":   str(action.get("preset", "EXHIBIT")),
            "position": str(action.get("position", "bottom-right")),
            "rotation": float(action.get("rotation", 0)),
            "custom_text": str(action.get("custom_text", "")) if action.get("custom_text") else "",
        }
        _, applied_pages = apply_sticker(session_dir, fname, page_numbers, sticker_config)
        pages_str = ", ".join(str(p) for p in applied_pages) if applied_pages else "none"
        return (
            f"Applied {sticker_config['preset']} {sticker_config['category']} sticker "
            f"to page(s) {pages_str} of {fname}"
        )

    elif atype == "ADD_PAGE_NUMBERS":
        fname = action.get("file", "")
        if not fname:
            raise ValueError("ADD_PAGE_NUMBERS requires 'file'.")
        pn_config = {
            "format":          str(action.get("format", "Page {n} of {total}")),
            "position":        str(action.get("position", "bottom-center")),
            "start_page":      int(action.get("start_page", 1)),
            "skip_first_page": bool(action.get("skip_first_page", False)),
            "font_size":       int(action.get("font_size", 10)),
        }
        _, stamped = add_page_numbers(session_dir, fname, pn_config)
        pos_label = pn_config["position"]
        fmt_label = pn_config["format"]
        return (
            f"Added page numbers to {stamped} page(s) of {fname} "
            f"({fmt_label!r} · {pos_label})"
        )

    else:
        raise ValueError(f"Unknown action type: '{atype}'")


# ---------------------------------------------------------------------------
# Serve SPA — must be last so API routes take priority
# ---------------------------------------------------------------------------

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return Response("App not found.", status_code=404)
    return FileResponse(str(index), media_type="text/html")
