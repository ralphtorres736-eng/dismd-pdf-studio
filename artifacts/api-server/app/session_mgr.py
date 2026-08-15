"""Session management: per-session temp directories, expiry cleanup."""
import os
import time
import shutil
import threading
from pathlib import Path

SESSIONS_BASE = Path("/tmp/pdf_sessions")
SESSION_TTL = 7200  # 2 hours in seconds
_lock = threading.Lock()

SESSIONS_BASE.mkdir(parents=True, exist_ok=True)


def get_session_dir(session_id: str) -> Path:
    """Return (and create if needed) the directory for a session."""
    d = SESSIONS_BASE / session_id
    d.mkdir(parents=True, exist_ok=True)
    _touch_session(session_id)
    return d


def _touch_session(session_id: str):
    """Update last-access timestamp for a session."""
    ts_file = SESSIONS_BASE / f"{session_id}.ts"
    ts_file.write_text(str(time.time()))


def get_last_access(session_id: str) -> float:
    ts_file = SESSIONS_BASE / f"{session_id}.ts"
    if ts_file.exists():
        try:
            return float(ts_file.read_text())
        except ValueError:
            pass
    return time.time()


def destroy_session(session_id: str):
    """Delete all session files immediately."""
    session_dir = SESSIONS_BASE / session_id
    ts_file = SESSIONS_BASE / f"{session_id}.ts"
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    if ts_file.exists():
        ts_file.unlink(missing_ok=True)


def cleanup_expired_sessions():
    """Remove sessions that have been idle longer than SESSION_TTL."""
    now = time.time()
    with _lock:
        for ts_file in SESSIONS_BASE.glob("*.ts"):
            session_id = ts_file.stem
            try:
                last = float(ts_file.read_text())
                if now - last > SESSION_TTL:
                    destroy_session(session_id)
            except Exception:
                pass


def sanitize_filename(name: str) -> str:
    """Strip path components and dangerous characters from a filename."""
    # Take only the basename
    name = os.path.basename(name)
    # Allow only safe characters
    safe = "".join(c for c in name if c.isalnum() or c in "._- ")
    safe = safe.strip(". ")
    if not safe:
        safe = "document.pdf"
    # Enforce .pdf extension
    if not safe.lower().endswith(".pdf"):
        safe = safe + ".pdf"
    return safe


def list_session_files(session_dir: Path) -> list[dict]:
    """Return a list of {name, size_bytes} for PDF files in the session dir."""
    files = []
    for p in sorted(session_dir.glob("*.pdf")):
        # Skip backup files
        if p.stem.endswith("_bak"):
            continue
        files.append({"name": p.name, "size_bytes": p.stat().st_size})
    return files
