"""
AI Command Parser — Deterministic structural command interpreter.

SECURITY BOUNDARY:
  The AI receives ONLY:
    - The user's plain-English structural instruction (text)
    - A list of document metadata: [{name: str, pages: int}]

  The AI NEVER receives:
    - Raw PDF bytes
    - Extracted document text or clauses
    - Page renderings or images
    - Any file content whatsoever

  Server-side assertion enforces this before every AI call.
"""
import os
import json
from openai import OpenAI

ALLOWED_ACTION_TYPES = {
    "MERGE", "DELETE_PAGES", "SPLIT", "ROTATE", "REORDER",
    "HIGHLIGHT", "RENAME", "REDACT", "ADD_STICKER", "ADD_PAGE_NUMBERS"
}

SYSTEM_PROMPT = """You are a PDF structural command parser for a legal document management tool.

Your ONLY job is to convert a plain-English structural instruction into a precise JSON execution plan.

RULES:
1. Output ONLY valid JSON — no prose, no markdown, no code fences.
2. Never ask clarifying questions — do your best with the information given.
3. Never include document content, text excerpts, or clause text in your output.
4. Use only the document names and page counts provided. Do not invent filenames.
5. Page numbers are always 1-based integers.

OUTPUT SCHEMA (always return exactly this structure):
{
  "actions": [
    // One or more action objects. Allowed types:
    
    // MERGE: combine files in listed order. cover_page is optional.
    {"type": "MERGE", "files": ["A.pdf", "B.pdf"], "output": "merged.pdf"},
    // MERGE with cover page — include when the user requests a cover/title page:
    {"type": "MERGE", "files": ["A.pdf", "B.pdf"], "output": "EXHIBIT_A.pdf",
     "cover_page": {
       "title": "EXHIBIT A",
       "subtitle": "Judicial Grievance Against Honorable Judge X",
       "body": ["Complainant / Counsel: Lisa M. Potter / The Potter Law Group",
                "Transcript 1: Cause No. 25DC-CV-01579 (In the Interest of ...)",
                "Transcript 2: Cause No. 26DC-CV-00505 (In the Matter of ...)"]
     }},
    
    // DELETE_PAGES: remove specific pages (1-based) from a file
    {"type": "DELETE_PAGES", "file": "A.pdf", "pages": [3, 4, 5]},
    
    // SPLIT: extract a page range into a new file
    {"type": "SPLIT", "file": "A.pdf", "start_page": 2, "end_page": 6, "output": "extract.pdf"},
    
    // ROTATE: rotate pages (degrees must be 90, 180, or 270). pages=[] means all pages.
    {"type": "ROTATE", "file": "A.pdf", "pages": [1, 2], "degrees": 90},
    
    // REORDER: reorder pages. new_order lists all page numbers in desired sequence.
    {"type": "REORDER", "file": "A.pdf", "new_order": [3, 1, 2]},
    
    // HIGHLIGHT: mark a region on a page. rect=[x0,y0,x1,y1] in PDF points.
    // color=[r,g,b] each 0-1. Use [1,0.9,0] for yellow, [0,0.9,0.5] for green.
    // When rect is unknown use [50, 50, 500, 80] as a reasonable default.
    {"type": "HIGHLIGHT", "file": "A.pdf", "page": 3, "rect": [50, 50, 500, 80], "color": [1, 0.9, 0], "label": "Note"},
    
    // RENAME: rename a file within the session
    {"type": "RENAME", "file": "old.pdf", "new_name": "new.pdf"},

    // REDACT: permanently black out all occurrences of each term.
    // terms come ONLY from the user's instruction — never from document content.
    // Output is saved as <original>-redacted.pdf. This action is irreversible.
    {"type": "REDACT", "file": "A.pdf", "terms": ["John Smith", "123-45-6789"]},

    // ADD_STICKER: overlay a vector sticker/stamp directly on one or more pages.
    // category: "legal_exhibit" | "status_stamp" | "novelty"
    // preset (legal_exhibit): "PLAINTIFF" | "DEFENDANT" | "PETITIONER" | "RESPONDENT" | "EXHIBIT"
    // preset (status_stamp):  "CONFIDENTIAL" | "URGENT" | "DRAFT"
    // preset (novelty):       "BORN TO ARGUE" | "LIVE LAUGH LAWSUIT" | "BILLING YOU FOR THIS" | "OBJECTION" | "JUSTICE SERVED"
    // position: "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center"  (default: "bottom-right")
    // custom_text: optional label displayed below the preset header (legal_exhibit only)
    // rotation: degrees to tilt the sticker, -15 to +15  (default: 0, useful for novelty stickers)
    // page_numbers: 1-based list; default [1] if omitted
    {"type": "ADD_STICKER", "file": "A.pdf", "page_numbers": [1],
     "category": "legal_exhibit", "preset": "PLAINTIFF",
     "custom_text": "EXHIBIT A", "position": "bottom-right", "rotation": 0},
    {"type": "ADD_STICKER", "file": "A.pdf", "page_numbers": [1, 2, 3, 4, 5],
     "category": "status_stamp", "preset": "CONFIDENTIAL",
     "position": "top-right", "rotation": 0},
    {"type": "ADD_STICKER", "file": "A.pdf", "page_numbers": [3],
     "category": "novelty", "preset": "OBJECTION",
     "position": "bottom-right", "rotation": -10},

     // ADD_PAGE_NUMBERS: stamp page numbers on every page of a document.
     // format: use {n} for current page number, {total} for total pages.
     //         Default: "Page {n} of {total}"
     // position: "bottom-center" | "bottom-right" | "top-right"  (default: "bottom-center")
     // start_page: logical number assigned to the first stamped page  (default: 1)
     // skip_first_page: if true, omit the number on page 1 (e.g. cover pages)  (default: false)
     // font_size: point size of number text  (default: 10)
     {"type": "ADD_PAGE_NUMBERS", "file": "A.pdf",
      "format": "Page {n} of {total}",
      "position": "bottom-center",
      "start_page": 1,
      "skip_first_page": false,
      "font_size": 10}
  ],
  "summary": "Plain-English description of what will be done."
}"""


def _assert_no_content_leak(instruction: str, documents: list[dict]):
    """
    Server-side assertion: verify the payload going to the AI contains
    only metadata (names + page counts) and the instruction text.
    Raises AssertionError if any unexpected content is detected.
    """
    assert isinstance(instruction, str), "Instruction must be a string."
    assert isinstance(documents, list), "Documents must be a list."
    for doc in documents:
        assert set(doc.keys()) <= {"name", "pages"}, (
            f"Document metadata must contain only 'name' and 'pages'. "
            f"Got: {set(doc.keys())}"
        )
        assert isinstance(doc["name"], str), "Document name must be a string."
        assert isinstance(doc["pages"], int), "Document page count must be an int."


def parse_instruction(instruction: str, documents: list[dict]) -> dict:
    """
    Send the instruction + document metadata to the AI parser.
    Returns a validated dict with 'actions' and 'summary'.

    Raises:
        ValueError: if the AI returns malformed JSON or unknown action types.
        AssertionError: if the security boundary is violated before the call.
    """
    # HARD SECURITY ASSERTION — never let text content reach the AI
    _assert_no_content_leak(instruction, documents)

    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "AI integration not configured. "
            "AI_INTEGRATIONS_OPENAI_BASE_URL and AI_INTEGRATIONS_OPENAI_API_KEY must be set."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Build the user message — ONLY metadata, never content
    doc_list_str = json.dumps(documents, indent=2)
    user_message = (
        f"Documents available:\n{doc_list_str}\n\n"
        f"Instruction: {instruction}"
    )

    response = client.chat.completions.create(
        model="gpt-5.6-luna",          # Cost-effective; task is deterministic parsing
        max_completion_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content or ""
    # Strip markdown code fences if the model wraps its output
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rstrip("`").strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"AI returned invalid JSON. Raw response: {raw[:500]}"
        ) from exc

    # Validate schema
    if "actions" not in plan or not isinstance(plan["actions"], list):
        raise ValueError("AI response missing 'actions' list.")
    if "summary" not in plan or not isinstance(plan["summary"], str):
        raise ValueError("AI response missing 'summary' string.")

    for i, action in enumerate(plan["actions"]):
        if "type" not in action:
            raise ValueError(f"Action #{i} missing 'type' field.")
        if action["type"] not in ALLOWED_ACTION_TYPES:
            raise ValueError(
                f"Action #{i} has unknown type '{action['type']}'. "
                f"Allowed: {sorted(ALLOWED_ACTION_TYPES)}"
            )

    return plan
