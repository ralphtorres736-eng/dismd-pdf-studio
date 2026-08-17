"""
PDF operations via PyMuPDF (fitz).

All page numbers in the public API are 1-based.
Internally fitz uses 0-based indexes; this module handles the conversion.
All operations save a .bak backup before modifying, enabling one-step undo.
"""
import shutil
from pathlib import Path
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backup(path: Path):
    """Copy <name>.pdf → <name>_bak.pdf before a destructive operation."""
    bak = path.parent / (path.stem + "_bak.pdf")
    shutil.copy2(str(path), str(bak))


def _invalidate_thumb_cache(session_dir: Path, filename: str):
    """Delete cached thumbnails for a file so they are re-rendered."""
    stem = Path(filename).stem
    for f in session_dir.glob(f"_thumb_{stem}_p*.png"):
        f.unlink(missing_ok=True)


def _validate_file(session_dir: Path, filename: str) -> Path:
    """
    Return the absolute path of a PDF that is known to be inside session_dir.
    Raises ValueError if the filename is outside the session directory (path traversal guard).
    """
    safe = (session_dir / Path(filename).name).resolve()
    if not str(safe).startswith(str(session_dir.resolve())):
        raise ValueError(f"Filename '{filename}' escapes session directory.")
    if not safe.exists():
        raise FileNotFoundError(f"File '{filename}' not found in session.")
    return safe


def _safe_save(doc: fitz.Document, path: Path) -> None:
    """
    Save a modified Document back to *path* via a sibling temp file then atomic replace.

    PyMuPDF ≥ 1.24 raises "save to original must be incremental" when the destination
    string matches doc.name (the path from which the file was opened).  Writing to a
    different temp path sidesteps the restriction without needing incremental mode.
    """
    tmp = path.parent / (path.stem + "_~tmp_save.pdf")
    try:
        doc.save(str(tmp), garbage=4, deflate=True)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Read-only operations
# ---------------------------------------------------------------------------

def get_page_count(session_dir: Path, filename: str) -> int:
    path = _validate_file(session_dir, filename)
    doc = fitz.open(str(path))
    count = doc.page_count
    doc.close()
    return count


def get_thumbnail(session_dir: Path, filename: str, page_num: int, width: int = 220) -> bytes:
    """
    Render a single page as PNG bytes.
    page_num is 1-based; cached on disk after first render.
    """
    path = _validate_file(session_dir, filename)
    stem = Path(filename).stem
    cache_path = session_dir / f"_thumb_{stem}_p{page_num}.png"
    if cache_path.exists():
        return cache_path.read_bytes()

    doc = fitz.open(str(path))
    if page_num < 1 or page_num > doc.page_count:
        doc.close()
        raise ValueError(f"Page {page_num} out of range (1–{doc.page_count}).")
    page = doc[page_num - 1]
    scale = width / page.rect.width
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    doc.close()
    cache_path.write_bytes(png_bytes)
    return png_bytes


def get_all_page_counts(session_dir: Path) -> dict[str, int]:
    """Return {filename: page_count} for all PDFs in the session."""
    result = {}
    for p in sorted(session_dir.glob("*.pdf")):
        if p.stem.endswith("_bak"):
            continue
        try:
            doc = fitz.open(str(p))
            result[p.name] = doc.page_count
            doc.close()
        except Exception:
            result[p.name] = 0
    return result


# ---------------------------------------------------------------------------
# Mutating operations (all create a .bak before modifying)
# ---------------------------------------------------------------------------

def generate_cover_page(cover_page: dict) -> "fitz.Document":
    """
    Generate a single-page cover page PDF from structured data.
    Returns an open fitz.Document — caller must close it.

    cover_page keys (all optional):
      title    — bold centered heading (e.g. "EXHIBIT A")
      subtitle — centered sub-heading (e.g. case name / judge)
      body     — list[str] of left-aligned lines (parties, cause numbers, etc.)
    """
    PAGE_W, PAGE_H = 612, 792   # US Letter in PDF points
    MARGIN = 72                  # 1-inch margins

    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    title    = (cover_page.get("title") or "").strip()
    subtitle = (cover_page.get("subtitle") or "").strip()
    body     = [str(ln).strip() for ln in (cover_page.get("body") or []) if str(ln).strip()]

    y = 210  # vertical cursor — start below top margin

    # ── TITLE  (24 pt bold, centered) ──────────────────────────────────────
    if title:
        rc = fitz.Rect(MARGIN, y, PAGE_W - MARGIN, y + 60)   # tall rect — never clips
        page.insert_textbox(
            rc, title,
            fontname="hebo",          # Helvetica-Bold (PyMuPDF base-14 alias)
            fontsize=24,
            align=1,                  # TEXT_ALIGN_CENTER
            color=(0, 0, 0),
        )
        y += 52

    # ── SUBTITLE  (13 pt, centered, may wrap) ───────────────────────────────
    if subtitle:
        rc = fitz.Rect(MARGIN, y, PAGE_W - MARGIN, y + 72)   # extra height for wrapping
        page.insert_textbox(
            rc, subtitle,
            fontname="helv",          # Helvetica (PyMuPDF base-14 alias)
            fontsize=13,
            align=1,                  # TEXT_ALIGN_CENTER
            color=(0.2, 0.2, 0.2),
        )
        y += 72

    # ── HORIZONTAL RULE ────────────────────────────────────────────────────
    y += 10
    page.draw_line(
        fitz.Point(MARGIN, y), fitz.Point(PAGE_W - MARGIN, y),
        color=(0.55, 0.55, 0.55), width=0.75,
    )
    y += 24

    # ── BODY LINES  (11 pt, left-aligned) ──────────────────────────────────
    for line in body:
        if y > PAGE_H - MARGIN:
            break                     # safety — never overflow the page
        rc = fitz.Rect(MARGIN, y, PAGE_W - MARGIN, y + 28)   # 28pt per line at 11pt font
        page.insert_textbox(
            rc, line,
            fontname="helv",          # Helvetica (PyMuPDF base-14 alias)
            fontsize=11,
            align=0,                  # TEXT_ALIGN_LEFT
            color=(0, 0, 0),
        )
        y += 20

    return doc


def merge_pdfs(
    session_dir: Path,
    filenames: list[str],
    output_name: str,
    cover_page: dict | None = None,
) -> str:
    """
    Merge PDFs in the given order into output_name.
    If cover_page is provided, a generated cover page is prepended as page 1.
    Returns the output filename.
    """
    from .session_mgr import sanitize_filename
    output_name = sanitize_filename(output_name)
    out_path = session_dir / output_name

    result = fitz.open()

    # Prepend generated cover page if requested
    if cover_page:
        cover_doc = generate_cover_page(cover_page)
        result.insert_pdf(cover_doc)
        cover_doc.close()

    for fname in filenames:
        src_path = _validate_file(session_dir, fname)
        src = fitz.open(str(src_path))
        result.insert_pdf(src)
        src.close()

    result.save(str(out_path))
    result.close()
    _invalidate_thumb_cache(session_dir, output_name)
    return output_name


def delete_pages(session_dir: Path, filename: str, pages: list[int]) -> str:
    """
    Delete specified 1-based page numbers from the PDF.
    Modifies the file in-place (with backup).
    """
    path = _validate_file(session_dir, filename)
    _backup(path)
    _invalidate_thumb_cache(session_dir, filename)

    doc = fitz.open(str(path))
    total = doc.page_count
    # Convert to 0-based, validate, deduplicate, sort descending to avoid index shift
    zero_based = sorted(
        set(p - 1 for p in pages if 1 <= p <= total),
        reverse=True
    )
    if not zero_based:
        doc.close()
        raise ValueError("No valid pages specified for deletion.")
    if len(zero_based) >= total:
        doc.close()
        raise ValueError("Cannot delete all pages from a document.")

    for idx in zero_based:
        doc.delete_page(idx)

    _safe_save(doc, path)
    doc.close()
    return filename


def split_pdf(session_dir: Path, filename: str, start_page: int, end_page: int, output_name: str) -> str:
    """
    Extract pages start_page..end_page (1-based, inclusive) into output_name.
    The source file is NOT modified.
    """
    from .session_mgr import sanitize_filename
    src_path = _validate_file(session_dir, filename)
    output_name = sanitize_filename(output_name)
    out_path = session_dir / output_name

    src = fitz.open(str(src_path))
    total = src.page_count
    s = max(1, start_page) - 1   # 0-based
    e = min(total, end_page) - 1  # 0-based inclusive

    result = fitz.open()
    result.insert_pdf(src, from_page=s, to_page=e)
    result.save(str(out_path))
    result.close()
    src.close()
    _invalidate_thumb_cache(session_dir, output_name)
    return output_name


def rotate_pages(session_dir: Path, filename: str, pages: list[int], degrees: int) -> str:
    """
    Rotate specified 1-based pages by degrees (must be 90, 180, or 270).
    Modifies in-place with backup.
    """
    if degrees not in (90, 180, 270):
        raise ValueError("Rotation must be 90, 180, or 270 degrees.")
    path = _validate_file(session_dir, filename)
    _backup(path)
    _invalidate_thumb_cache(session_dir, filename)

    doc = fitz.open(str(path))
    total = doc.page_count
    target = set(p - 1 for p in pages if 1 <= p <= total)
    # "all" sentinel: empty list means all pages
    if not target:
        target = set(range(total))

    for idx in target:
        page = doc[idx]
        page.set_rotation((page.rotation + degrees) % 360)

    _safe_save(doc, path)
    doc.close()
    return filename


def reorder_pages(session_dir: Path, filename: str, new_order: list[int]) -> str:
    """
    Reorder pages according to new_order (1-based page numbers in desired order).
    Modifies in-place with backup.
    """
    path = _validate_file(session_dir, filename)

    # Validate before touching the backup — a failed validation must not
    # overwrite the existing backup with a bad or partially-processed state.
    doc = fitz.open(str(path))
    total = doc.page_count
    zero_based = [p - 1 for p in new_order]
    if sorted(zero_based) != list(range(total)):
        doc.close()
        raise ValueError(
            f"new_order must be a valid permutation of all {total} page numbers "
            f"(1–{total}) with no duplicates or out-of-range values."
        )

    _backup(path)
    _invalidate_thumb_cache(session_dir, filename)

    doc.select(zero_based)
    # Save to a temp file then replace — direct in-place save raises
    # "save to original must be incremental" after doc.select() in PyMuPDF ≥1.24.
    tmp_path = path.parent / (path.stem + "_reorder_tmp.pdf")
    doc.save(str(tmp_path), garbage=4, deflate=True)
    doc.close()
    shutil.move(str(tmp_path), str(path))
    return filename


def highlight_page(
    session_dir: Path,
    filename: str,
    page_num: int,
    rect: list[float],    # [x0, y0, x1, y1] in PDF user-space points
    color: list[float],   # [r, g, b] each 0–1
    label: str = "",
) -> str:
    """
    Add a highlight annotation rectangle on the specified 1-based page.
    rect is [x0, y0, x1, y1] in PDF points from top-left.
    Modifies in-place with backup.
    """
    path = _validate_file(session_dir, filename)
    _backup(path)
    _invalidate_thumb_cache(session_dir, filename)

    doc = fitz.open(str(path))
    if page_num < 1 or page_num > doc.page_count:
        doc.close()
        raise ValueError(f"Page {page_num} out of range.")

    page = doc[page_num - 1]
    annot_rect = fitz.Rect(rect[0], rect[1], rect[2], rect[3])

    # Add a highlight annotation
    annot = page.add_rect_annot(annot_rect)
    r, g, b = (float(c) for c in color[:3])
    annot.set_colors(stroke=None, fill=(r, g, b))
    annot.set_opacity(0.35)
    if label:
        annot.set_info(content=label)
    annot.update()

    _safe_save(doc, path)
    doc.close()
    return filename


def redact_pdf(session_dir: Path, filename: str, terms: list[str]) -> tuple[str, int]:
    """
    Permanently redact all occurrences of each search term in the PDF.
    Applies opaque black fill rectangles and burns them into the page content
    using PyMuPDF's redaction API (content is unrecoverable after save).

    Creates a NEW file named <stem>-redacted.pdf — the source is NEVER modified.
    Returns (output_filename, total_match_count).
    """
    from .session_mgr import sanitize_filename

    src_path = _validate_file(session_dir, filename)
    stem = Path(filename).stem
    output_name = sanitize_filename(f"{stem}-redacted.pdf")
    out_path = session_dir / output_name

    clean_terms = [t.strip() for t in terms if t.strip()]
    if not clean_terms:
        raise ValueError("At least one non-empty search term is required.")

    # Collision-safe output name: <stem>-redacted.pdf, then <stem>-redacted_1.pdf, etc.
    candidate = output_name
    counter = 1
    while (session_dir / candidate).exists():
        candidate = sanitize_filename(f"{stem}-redacted_{counter}.pdf")
        counter += 1
    output_name = candidate
    out_path = session_dir / output_name

    doc = fitz.open(str(src_path))
    total_hits = 0

    for page in doc:
        for term in clean_terms:
            rects = page.search_for(term, quads=False)
            for rect in rects:
                page.add_redact_annot(rect, fill=(0, 0, 0))
                total_hits += 1
        # PDF_REDACT_IMAGE_PIXELS blacks out the pixel region within each
        # redaction rectangle in any underlying image layer, preventing
        # recovery of sensitive content from image-backed (e.g. scanned) pages.
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)

    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    _invalidate_thumb_cache(session_dir, output_name)
    return output_name, total_hits


def check_has_text(session_dir: Path, filename: str) -> bool:
    """
    Return True if the PDF appears to already contain an extractable text layer.
    Checks the first few pages to avoid scanning the whole document.
    """
    path = _validate_file(session_dir, filename)
    doc = fitz.open(str(path))
    found = False
    for i in range(min(3, doc.page_count)):
        if doc[i].get_text("text").strip():
            found = True
            break
    doc.close()
    return found


def ocr_pdf(
    session_dir: Path,
    filename: str,
    progress_callback=None,  # callable(current_page: int, total_pages: int) | None
) -> str:
    """
    Run Tesseract OCR on every page of the PDF and produce a new searchable PDF.

    Each page is rendered at 300 DPI to a PIL Image, passed to pytesseract
    (which returns a single-page searchable PDF with an invisible text layer),
    then all pages are stitched together with PyMuPDF.

    Creates a NEW file <stem>-ocr.pdf — source is NEVER modified.
    Returns the output filename.
    """
    import io as _io
    from PIL import Image
    import pytesseract
    from .session_mgr import sanitize_filename

    src_path = _validate_file(session_dir, filename)
    stem = Path(filename).stem

    # Collision-safe output name
    output_name = sanitize_filename(f"{stem}-ocr.pdf")
    counter = 1
    while (session_dir / output_name).exists():
        output_name = sanitize_filename(f"{stem}-ocr_{counter}.pdf")
        counter += 1

    src = fitz.open(str(src_path))
    total = src.page_count

    output_doc = fitz.open()

    for i in range(total):
        if progress_callback:
            progress_callback(i + 1, total)

        page = src[i]
        # 300 DPI for solid OCR quality (72 is PDF native, so scale = 300/72 ≈ 4.17)
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Pixmap → PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # PIL Image → single-page searchable PDF bytes (invisible text layer over image)
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension="pdf")

        # Merge the single-page OCR PDF into our output document
        ocr_page_doc = fitz.open("pdf", _io.BytesIO(pdf_bytes))
        output_doc.insert_pdf(ocr_page_doc)
        ocr_page_doc.close()

    src.close()

    out_path = session_dir / output_name
    output_doc.save(str(out_path), garbage=4, deflate=True)
    output_doc.close()
    _invalidate_thumb_cache(session_dir, output_name)

    # Write an undo-by-deletion marker so undo_last_op can remove this file.
    # OCR never modifies the source, so "undo" means deleting what was just created.
    (session_dir / (Path(output_name).stem + ".undo_delete")).write_bytes(b"")

    return output_name


def undo_last_op(session_dir: Path, filename: str) -> bool:
    """
    Undo the last operation on a file.  Two modes:

    1. Restore-from-backup (mutating ops like delete/rotate/redact-in-place):
       Copies <stem>_bak.pdf back over the file and removes the backup.

    2. Undo-by-deletion (creating ops like OCR that produce a new file):
       If a <stem>.undo_delete marker exists, delete the generated file
       and the marker.  This is the correct undo for operations that never
       modify the original — "remove what was just created."

    Returns True if either undo succeeded.
    """
    path = _validate_file(session_dir, filename)
    bak = session_dir / (Path(filename).stem + "_bak.pdf")
    delete_marker = session_dir / (Path(filename).stem + ".undo_delete")

    if bak.exists():
        # Standard restore-from-backup path
        shutil.copy2(str(bak), str(path))
        bak.unlink(missing_ok=True)
        delete_marker.unlink(missing_ok=True)  # safety clean-up
        _invalidate_thumb_cache(session_dir, filename)
        return True

    if delete_marker.exists():
        # Undo by deleting the newly created file
        delete_marker.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        _invalidate_thumb_cache(session_dir, filename)
        return True

    return False


# ---------------------------------------------------------------------------
# Sticker / Stamp Overlay Engine
# ---------------------------------------------------------------------------

# ── Preset colour tables (all RGB floats 0-1) ─────────────────────────────

_EXHIBIT_PRESETS: dict[str, dict] = {
    "PLAINTIFF":  {"fill": (0.992, 0.878, 0.278), "tc": (0.0, 0.0, 0.0), "header": "PLAINTIFF'S\nEXHIBIT"},
    "DEFENDANT":  {"fill": (0.220, 0.741, 0.973), "tc": (0.0, 0.0, 0.0), "header": "DEFENDANT'S\nEXHIBIT"},
    "PETITIONER": {"fill": (0.984, 0.573, 0.188), "tc": (0.0, 0.0, 0.0), "header": "PETITIONER'S\nEXHIBIT"},
    "RESPONDENT": {"fill": (0.176, 0.831, 0.749), "tc": (0.0, 0.0, 0.0), "header": "RESPONDENT'S\nEXHIBIT"},
    "EXHIBIT":    {"fill": (1.000, 1.000, 1.000), "tc": (0.0, 0.0, 0.0), "header": "EXHIBIT"},
}

_NOVELTY_PRESETS: dict[str, dict] = {
    "BORN TO ARGUE":        {"fill": (0.118, 0.227, 0.373), "tc": (0.965, 0.624, 0.043)},
    "LIVE LAUGH LAWSUIT":   {"fill": (0.078, 0.325, 0.173), "tc": (1.0, 1.0, 1.0)},
    "BILLING YOU FOR THIS": {"fill": (0.471, 0.208, 0.059), "tc": (0.996, 0.953, 0.773)},
    "OBJECTION":            {"fill": (0.498, 0.114, 0.114), "tc": (1.0, 1.0, 1.0)},
    "JUSTICE SERVED":       {"fill": (0.118, 0.251, 0.694), "tc": (1.0, 1.0, 1.0)},
}


# ── Geometry helpers ──────────────────────────────────────────────────────

def _sticker_rect(page_rect: fitz.Rect, position: str, w: float, h: float) -> fitz.Rect:
    """Return a Rect of size w×h placed at position on the page (20 pt margin)."""
    margin = 20.0
    pw, ph = page_rect.width, page_rect.height
    pos = position.lower()
    if pos == "top-left":
        return fitz.Rect(margin, margin, margin + w, margin + h)
    elif pos == "top-right":
        return fitz.Rect(pw - margin - w, margin, pw - margin, margin + h)
    elif pos == "bottom-left":
        return fitz.Rect(margin, ph - margin - h, margin + w, ph - margin)
    elif pos == "center":
        return fitz.Rect((pw - w) / 2, (ph - h) / 2, (pw + w) / 2, (ph + h) / 2)
    else:  # default: bottom-right
        return fitz.Rect(pw - margin - w, ph - margin - h, pw - margin, ph - margin)


def _inner(rect: fitz.Rect, pad: float) -> fitz.Rect:
    """Shrink a Rect by pad on all sides."""
    return fitz.Rect(rect.x0 + pad, rect.y0 + pad, rect.x1 - pad, rect.y1 - pad)


# ── Per-category drawing helpers ──────────────────────────────────────────

def _draw_legal_exhibit(page: fitz.Page, position: str, preset_key: str, custom_text: str) -> None:
    """Draw a coloured exhibit tab (100 × 72 pt)."""
    key = preset_key.upper()
    preset = _EXHIBIT_PRESETS.get(key, _EXHIBIT_PRESETS["EXHIBIT"])
    fill = preset["fill"]
    tc   = preset["tc"]
    header = preset["header"]

    rect = _sticker_rect(page.rect, position, 100.0, 72.0)
    mid_y = rect.y0 + 42.0        # separator line y-coordinate

    # Filled background + black border
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(fill=fill, color=(0.0, 0.0, 0.0), width=1.2)
    shape.commit()

    # Separator line
    shape2 = page.new_shape()
    shape2.draw_line(fitz.Point(rect.x0 + 4, mid_y), fitz.Point(rect.x1 - 4, mid_y))
    shape2.finish(color=(0.0, 0.0, 0.0), width=0.7)
    shape2.commit()

    # Header text (bold, e.g. "PLAINTIFF'S\nEXHIBIT")
    header_rect = fitz.Rect(rect.x0 + 2, rect.y0 + 4, rect.x1 - 2, mid_y - 2)
    page.insert_textbox(header_rect, header, fontname="hebo", fontsize=8.5,
                        color=tc, align=1)  # align=1 → centred

    # Custom label below separator
    if custom_text:
        label_rect = fitz.Rect(rect.x0 + 2, mid_y + 3, rect.x1 - 2, rect.y1 - 3)
        page.insert_textbox(label_rect, custom_text, fontname="helv", fontsize=8,
                            color=tc, align=1)


def _draw_status_stamp(page: fitz.Page, position: str, preset_key: str) -> None:
    """Draw CONFIDENTIAL, URGENT, or DRAFT status stamps."""
    key = preset_key.upper()

    if key == "DRAFT":
        # Large diagonal-ish watermark: draw "DRAFT" vertically centred, rotated 90°
        pw, ph = page.rect.width, page.rect.height
        # rotate=90 in insert_text means the text runs bottom-to-top in PDF coordinates,
        # which visually reads left-to-right when the page is rotated — we use 0 here
        # for a clean centred horizontal watermark that spans the page width.
        draft_rect = fitz.Rect(pw * 0.1, ph * 0.38, pw * 0.90, ph * 0.62)
        # Draw a very light grey filled rect behind the text so it reads as a band
        shape = page.new_shape()
        shape.draw_rect(draft_rect)
        shape.finish(fill=(0.90, 0.90, 0.90), color=None, width=0)
        shape.commit()
        page.insert_textbox(draft_rect, "DRAFT",
                            fontname="hebo", fontsize=80,
                            color=(0.612, 0.639, 0.686), align=1)
        return

    if key == "URGENT":
        # Filled red banner
        rect = _sticker_rect(page.rect, position, 210.0, 38.0)
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(fill=(0.863, 0.149, 0.149), color=(0.863, 0.149, 0.149), width=0)
        shape.commit()
        page.insert_textbox(_inner(rect, 3), "URGENT / TIME SENSITIVE",
                            fontname="hebo", fontsize=13,
                            color=(1.0, 1.0, 1.0), align=1)
        return

    # Default: CONFIDENTIAL — thick red border frame, red text
    rect = _sticker_rect(page.rect, position, 210.0, 42.0)
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(fill=None, color=(0.863, 0.149, 0.149), width=3.0)
    shape.commit()
    page.insert_textbox(_inner(rect, 4), "CONFIDENTIAL",
                        fontname="hebo", fontsize=16,
                        color=(0.863, 0.149, 0.149), align=1)


def _draw_novelty(page: fitz.Page, position: str, preset_key: str,
                  rotation: float, custom_text: str) -> None:
    """Draw a novelty badge, with optional slight rotation via morph."""
    key = preset_key.upper()
    preset = _NOVELTY_PRESETS.get(key, _NOVELTY_PRESETS["OBJECTION"])
    fill = preset["fill"]
    tc   = preset["tc"]
    display = custom_text if custom_text else key  # fallback to preset name

    rect = _sticker_rect(page.rect, position, 168.0, 52.0)
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    center = fitz.Point(cx, cy)

    use_rotation = abs(rotation) > 0.1
    morph = (center, fitz.Matrix(rotation)) if use_rotation else None

    # Filled background + thick border
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(fill=fill, color=tc, width=3.0, morph=morph)
    shape.commit()

    # Inner highlight line (top accent bar)
    bar = fitz.Rect(rect.x0 + 5, rect.y0 + 5, rect.x1 - 5, rect.y0 + 8)
    shape2 = page.new_shape()
    shape2.draw_rect(bar)
    shape2.finish(fill=tc, color=None, width=0, morph=morph)
    shape2.commit()

    if use_rotation:
        # insert_text supports morph for arbitrary rotation
        fsize = 11
        # Place origin at left-centre so morph rotates it into the badge centre
        text_pt = fitz.Point(rect.x0 + 8, cy + fsize * 0.35)
        page.insert_text(text_pt, display, fontname="hebo",
                         fontsize=fsize, color=tc, morph=morph)
    else:
        page.insert_textbox(_inner(rect, 8), display,
                            fontname="hebo", fontsize=11,
                            color=tc, align=1)


# ── Public entry point ────────────────────────────────────────────────────

def apply_sticker(
    session_dir: Path,
    filename: str,
    page_numbers: list[int],
    sticker_config: dict,
) -> tuple[str, list[int]]:
    """
    Apply a vector sticker/stamp to the specified pages of a PDF.

    sticker_config keys:
        category    : "legal_exhibit" | "status_stamp" | "novelty"
        preset      : preset name string (see _EXHIBIT_PRESETS / _NOVELTY_PRESETS)
        position    : "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center"
        rotation    : float degrees, -15 to +15 (used for novelty only)
        custom_text : optional extra label (legal_exhibit label or novelty override)

    Returns (filename, list_of_pages_actually_modified).
    Raises FileNotFoundError / ValueError on bad input.
    """
    if not page_numbers:
        page_numbers = [1]

    path = _validate_file(session_dir, filename)
    _backup(path)

    doc = fitz.open(str(path))
    total_pages = doc.page_count

    category    = sticker_config.get("category", "legal_exhibit").lower()
    preset      = sticker_config.get("preset", "EXHIBIT").upper()
    position    = sticker_config.get("position", "bottom-right").lower()
    rotation    = float(sticker_config.get("rotation", 0))
    custom_text = sticker_config.get("custom_text", "") or ""

    # Clamp rotation to safe range
    rotation = max(-15.0, min(15.0, rotation))

    applied: list[int] = []
    for pnum in page_numbers:
        if pnum < 1 or pnum > total_pages:
            continue  # silently skip out-of-range pages

        page = doc[pnum - 1]

        if category == "legal_exhibit":
            _draw_legal_exhibit(page, position, preset, custom_text)
        elif category == "status_stamp":
            _draw_status_stamp(page, position, preset)
        else:  # novelty
            _draw_novelty(page, position, preset, rotation, custom_text)

        applied.append(pnum)

    if applied:
        _safe_save(doc, path)
        # Invalidate thumbnails for every modified page
        for pnum in applied:
            stem = Path(filename).stem
            thumb = session_dir / f"_thumb_{stem}_p{pnum}.png"
            thumb.unlink(missing_ok=True)

    doc.close()
    return filename, applied


# ── Page Numbering ────────────────────────────────────────────────────────────

def add_page_numbers(
    session_dir: Path,
    filename: str,
    config: dict,
) -> tuple[str, int]:
    """
    Stamp page numbers onto every page of a PDF using PyMuPDF insert_text.

    config keys:
        format          : str  — template with {n} and {total}. Default "Page {n} of {total}"
        position        : "bottom-center" | "bottom-right" | "top-right". Default "bottom-center"
        start_page      : int  — logical number for the first page. Default 1
        skip_first_page : bool — omit number on physical page 1. Default False
        font_size       : int  — point size. Default 10

    Returns (filename, pages_stamped).
    """
    path = _validate_file(session_dir, filename)
    _backup(path)

    fmt         = str(config.get("format", "Page {n} of {total}"))
    position    = str(config.get("position", "bottom-center")).lower()
    start_page  = int(config.get("start_page", 1))
    skip_first  = bool(config.get("skip_first_page", False))
    font_size   = int(config.get("font_size", 10))

    # Clamp font size to a sane range
    font_size = max(6, min(24, font_size))

    doc = fitz.open(str(path))
    total = doc.page_count
    stamped = 0

    for idx in range(total):
        if skip_first and idx == 0:
            continue

        page = doc[idx]
        rect = page.rect
        w, h = rect.width, rect.height

        logical_n = start_page + idx

        label = fmt.replace("{n}", str(logical_n)).replace("{total}", str(total))

        # Estimate text width (≈ font_size * 0.5 per char is conservative)
        approx_w = len(label) * font_size * 0.52

        margin = 18  # points from edge

        if position == "bottom-center":
            x = (w - approx_w) / 2
            y = h - margin
        elif position == "bottom-right":
            x = w - approx_w - margin
            y = h - margin
        elif position == "top-right":
            x = w - approx_w - margin
            y = margin + font_size
        else:
            # fallback: bottom-center
            x = (w - approx_w) / 2
            y = h - margin

        page.insert_text(
            fitz.Point(x, y),
            label,
            fontname="helv",
            fontsize=font_size,
            color=(0.0, 0.0, 0.0),
        )
        stamped += 1

    if stamped:
        _safe_save(doc, path)
        _invalidate_thumb_cache(session_dir, filename)

    doc.close()
    return filename, stamped
