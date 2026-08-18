@'
# DISMD AI Secure PDF Studio

A privacy-first, zero-cloud-leak legal document engineering platform built in Python using **FastAPI** and **PyMuPDF (`fitz`)**. Designed for high-volume court filing prep, multi-volume exhibit compilation, vector stamp generation, and local PDF manipulation.

---

## Key Features

- **Zero-Cloud-Leak Security Boundary:** Processes files entirely in local ephemeral VM memory (`/tmp/pdf_sessions/`) with zero disk persistence or external LLM data transmission.
- **Deterministic Action Parser:** Translates natural-language chat inputs into strict, validated JSON action payloads (`MERGE`, `ADD_STICKER`, `REDACT`, `TRIM`, `SPLIT`) to eliminate model hallucination.
- **High-Volume Exhibit Processing:** Tested and verified on complex real-world filings, including 16-file master exhibit packages (`EX_A` through `EX_Q`) with a 500 MB RAM stability threshold.
- **Native Vector Overlay Engine:** Generates court-compliant digital exhibit stickers and stamps directly into PDF coordinate spaces using native PyMuPDF vector drawing (`draw_rect`, `insert_textbox`).
- **Page Manipulation & OCR:** Supports dynamic page reordering, page trimming, splitting, cover page generation, and local optical character recognition.

---

## Technical Stack

- **Backend:** Python 3.11+, FastAPI, PyMuPDF (`fitz`), Uvicorn
- **Frontend:** Single-Page Application (Vanilla JS / HTML5 / Tailwind CSS)
- **Memory & Storage:** Ephemeral local VM workspace (`/tmp/pdf_sessions/`), Client-side Blob URL downloads

---

## Quick Start (Local Staging)

### 1. Clone Repository & Install Dependencies
```bash
git clone [https://github.com/ralphtorres736-eng/dismd-pdf-studio.git](https://github.com/ralphtorres736-eng/dismd-pdf-studio.git)
cd dismd-pdf-studio
pip install -r requirements.txt
