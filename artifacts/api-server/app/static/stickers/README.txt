DISMD AI — Sticker Assets Directory
=====================================
This directory is reserved for optional future PNG/SVG sticker overlay assets.

Current stickers (legal exhibit tabs, status stamps, novelty badges) are rendered
entirely as vector graphics via PyMuPDF — no files in this directory are required
for the sticker engine to function.

If you add named PNG files here (e.g. "objection.png"), you can extend
pdf_ops.py's _draw_novelty() to composite them as raster overlays.
