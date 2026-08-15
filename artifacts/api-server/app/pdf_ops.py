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

def merge_pdfs(session_dir: Path, filenames: list[str], output_name: str) -> str:
    """
    Merge PDFs in the given order into output_name.
    Returns the output filename.
    """
    from .session_mgr import sanitize_filename
    output_name = sanitize_filename(output_name)
    out_path = session_dir / output_name

    result = fitz.open()
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

    doc.save(str(path), incremental=False, garbage=4, deflate=True)
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

    doc.save(str(path), incremental=False, garbage=4, deflate=True)
    doc.close()
    return filename


def reorder_pages(session_dir: Path, filename: str, new_order: list[int]) -> str:
    """
    Reorder pages according to new_order (1-based page numbers in desired order).
    Modifies in-place with backup.
    """
    path = _validate_file(session_dir, filename)
    _backup(path)
    _invalidate_thumb_cache(session_dir, filename)

    doc = fitz.open(str(path))
    total = doc.page_count
    zero_based = [p - 1 for p in new_order if 1 <= p <= total]
    if len(zero_based) != total:
        doc.close()
        raise ValueError(f"new_order must contain all {total} page numbers exactly once.")

    doc.select(zero_based)
    doc.save(str(path), incremental=False, garbage=4, deflate=True)
    doc.close()
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

    doc.save(str(path), incremental=False, garbage=4, deflate=True)
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
