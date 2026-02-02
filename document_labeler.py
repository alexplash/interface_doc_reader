from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import OpenAI

load_dotenv()


# -----------------------------
# Utilities
# -----------------------------

def safe_filename(name: str, max_len: int = 200) -> str:
    """
    Sanitize a string so it's safe-ish to store/use (NOT used for filenames anymore).
    Numbers are preserved.
    """
    name = (name or "").strip()
    name = re.sub(r"[\x00-\x1F]+", " ", name)  # remove control chars
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def extract_first_json_object(text: str) -> Optional[dict]:
    """
    Try to find and parse the first JSON object in a model response.
    """
    if not text:
        return None

    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# -----------------------------
# Agent
# -----------------------------

@dataclass
class DocumentLabelerAgent:
    pdf_path: str = "sample.pdf"
    out_dir: str = "labeled_files"
    metadata_filename: str = "metadata.json"

    model: str = "gpt-4.1-mini"  # vision-capable model
    dpi: int = 200
    title_band_ratio: float = 0.22  # top ~22% of page

    def __post_init__(self) -> None:
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Could not find PDF: {self.pdf_path}")

        os.makedirs(self.out_dir, exist_ok=True)

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.doc = fitz.open(self.pdf_path)

        # Holds mapping "0001" -> {"titles": [...], ...}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def close(self) -> None:
        try:
            self.doc.close()
        except Exception:
            pass

    def _page_id(self, page_index: int) -> str:
        # page_index is 0-based; id is 1-based with zero-padding
        return f"{page_index + 1:04d}"

    def _render_title_band_png_bytes(self, page_index: int) -> bytes:
        """
        Render just the top band of the page at requested DPI.
        """
        page = self.doc[page_index]
        rect = page.rect
        band_h = rect.height * float(self.title_band_ratio)
        clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + band_h)

        pix = page.get_pixmap(dpi=self.dpi, clip=clip)
        return pix.tobytes("png")

    def _ask_model_for_titles(self, png_bytes: bytes) -> List[str]:
        """
        Return ALL titles found in the header band.
        Expected output JSON: {"titles": ["...", "..."]}.
        """
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        prompt = (
            "You are extracting ALL document titles that appear in the TOP area of this page image.\n"
            "Sometimes there are multiple titles.\n\n"
            "Return ONLY valid JSON with exactly one key:\n"
            '  {"titles": ["<title1>", "<title2>", "..."]}\n\n'
            "Rules:\n"
            "- Extract every title/heading in the header area that matches the title formatting.\n"
            "- Preserve the order from top-to-bottom.\n"
            "- Include dates and small header text IF they are part of a title line.\n"
            "- If there is only one title, return a list with one item.\n"
            "- If there are no titles, return an empty list: {\"titles\": []}\n"
            "- No extra keys. No commentary. JSON only.\n"
        )

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )

        raw = (resp.output_text or "").strip()
        obj = extract_first_json_object(raw)

        # Fallback: treat output as a single title if JSON failed
        if not obj or "titles" not in obj:
            maybe = safe_filename(raw)
            return [maybe] if maybe else []

        titles = obj.get("titles", [])
        if not isinstance(titles, list):
            return []

        # Clean titles (keep numbers; just normalize whitespace/control chars)
        cleaned: List[str] = []
        for t in titles:
            if isinstance(t, str):
                s = safe_filename(t)
                if s:
                    cleaned.append(s)

        return cleaned

    def _save_single_page_pdf_numeric(self, page_index: int) -> str:
        """
        Save just one page into a new PDF named 0001.pdf, 0002.pdf, ...
        """
        page_id = self._page_id(page_index)
        out_path = os.path.join(self.out_dir, f"{page_id}.pdf")

        sub = fitz.open(self.pdf_path)
        sub.select([page_index])
        sub.save(out_path)
        sub.close()

        return out_path

    def _write_metadata_json(self) -> str:
        path = os.path.join(self.out_dir, self.metadata_filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        return path

    def run(self, start_page: int = 0, end_page: Optional[int] = None) -> None:
        total = self.doc.page_count
        if end_page is None or end_page > total:
            end_page = total

        for i in range(start_page, end_page):
            page_id = self._page_id(i)

            try:
                png_bytes = self._render_title_band_png_bytes(i)
                titles = self._ask_model_for_titles(png_bytes)

                pdf_path = self._save_single_page_pdf_numeric(i)

                # Store metadata
                self.metadata[page_id] = {
                    "titles": titles,
                    "page_index": i,
                    "source_pdf": os.path.basename(self.pdf_path),
                    "output_pdf": os.path.basename(pdf_path),
                }

                print(f"[OK] Page {i+1}/{total} -> {pdf_path} | titles={len(titles)}")

            except Exception as e:
                pdf_path = self._save_single_page_pdf_numeric(i)
                self.metadata[page_id] = {
                    "titles": [],
                    "page_index": i,
                    "source_pdf": os.path.basename(self.pdf_path),
                    "output_pdf": os.path.basename(pdf_path),
                    "error": str(e),
                }
                print(f"[ERR] Page {i+1}/{total}: {e} -> saved {pdf_path}")

        meta_path = self._write_metadata_json()
        print(f"[DONE] Wrote metadata: {meta_path}")
