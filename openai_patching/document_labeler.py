
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
import shutil
from pathlib import Path
import io
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


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


def extract_first_json_object_from_model(text: str) -> Optional[dict]:
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
    


@dataclass
class DocumentLabeler:
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

        # OCR engine (init once)
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

        # Holds mapping "0001" -> {"titles": [...], ...}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def close(self) -> None:
        try:
            self.doc.close()
        except Exception:
            pass

    def page_id(self, page_index: int) -> str:
        # page_index is 0-based; id is 1-based with zero-padding
        return f"{page_index + 1:04d}"

    def render_title_band_png_bytes(self, page_index: int) -> bytes:
        """
        Render just the top band of the page at requested DPI.
        """
        page = self.doc[page_index]
        rect = page.rect
        band_h = rect.height * float(self.title_band_ratio)
        clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + band_h)

        pix = page.get_pixmap(dpi=self.dpi, clip=clip)
        return pix.tobytes("png")

    def ask_model_for_titles(self, png_bytes: bytes) -> List[str]:
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
        obj = extract_first_json_object_from_model(raw)

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

    def save_single_page_pdf_numeric(self, page_index: int) -> str:
        """
        Save just one page into a new PDF named 0001.pdf, 0002.pdf, ...
        """
        page_id = self.page_id(page_index)
        out_path = os.path.join(self.out_dir, f"{page_id}.pdf")

        sub = fitz.open(self.pdf_path)
        sub.select([page_index])
        sub.save(out_path)
        sub.close()

        return out_path

    def write_metadata_json(self) -> str:
        path = os.path.join(self.out_dir, self.metadata_filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        return path
    
    def clear_out_dir(self) -> None:
        """
        Delete all contents of self.out_dir (files + subfolders), but keep the directory itself.
        Includes a couple sanity checks to reduce foot-guns.
        """
        out_path = Path(self.out_dir).resolve()

        # Basic safety checks
        if str(out_path) in ("/", ""):
            raise ValueError(f"Refusing to clear dangerous out_dir: {out_path}")

        out_path.mkdir(parents=True, exist_ok=True)

        for child in out_path.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except Exception as e:
                raise RuntimeError(f"Failed clearing {child}: {e}") from e
    
    def extract_pdf_text_items(self, page_index: int) -> List[Dict[str, Any]]:
        """
        Extract *all* text spans from the PDF text layer with bounding boxes.
        This is Tier-1: no classification yet, just capture everything reliably.

        Returns:
            [
            {
                "raw": "F-715A",
                "text": "F-715A",
                "bbox": [x0, y0, x1, y1],
                "font_size": 7.5,
                "font": "Helvetica",
                "flags": 4
            },
            ...
            ]
        """
        page = self.doc[page_index]

        # "dict" preserves coordinates and span-level granularity
        data = page.get_text("dict")

        items = set()

        blocks = data.get("blocks", [])
        for b in blocks:
            if b.get("type") != 0:
                # type 0 = text; other types can be images/graphics
                continue

            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    raw = span.get("text", "")
                    if not raw:
                        continue

                    # Normalize whitespace/control chars but keep content (numbers etc.)
                    cleaned = safe_filename(raw)
                    if not cleaned:
                        continue

                    bbox = span.get("bbox", None)
                    if not bbox or len(bbox) != 4:
                        continue

                    items.add(cleaned)

        return list(items)
    
    def render_full_page_png_bytes(self, page_index: int, max_side_px: int = 3800) -> bytes:
        page = self.doc[page_index]
        rect = page.rect  # points; 72 points = 1 inch

        max_side_points = max(rect.width, rect.height)
        dpi = int(max_side_px * 72 / max_side_points)
        dpi = max(120, min(400, dpi))  # clamp to sane range

        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")

    def norm_text(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def ocr_png_bytes(self, png_bytes: bytes) -> List[Dict[str, Any]]:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img)

        result = self.ocr.ocr(arr)
        d = result[0]
        texts = d.get("rec_texts", []) or []
        scores = d.get("rec_scores", []) or []
        polys = d.get("rec_polys") or d.get("dt_polys") or []

        items = set()

        for poly, text, score in zip(polys, texts, scores):
            text_clean = safe_filename(text)
            if not text_clean:
                continue

            # poly is often a numpy array shape (4,2)
            if hasattr(poly, "tolist"):
                poly = poly.tolist()

            # poly: [[x,y],[x,y],[x,y],[x,y]] (or more points)
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

            conf_score = float(score) if score is not None else None
            if conf_score is not None:
                if len(text_clean) <= 2 and conf_score < 0.9:
                    continue
                if conf_score < 0.8:
                    continue
                else:
                    norm_item = self.norm_text(text_clean)
                    items.add(norm_item)

        return list(items)


    def run(self, start_page: int = 0, end_page: Optional[int] = None) -> None:
        self.clear_out_dir()

        total = self.doc.page_count
        if end_page is None or end_page > total:
            end_page = total

        for i in range(start_page, end_page):
            page_id = self.page_id(i)

            titles: List[str] = []
            text_items: List[str] = []
            ocr_items: List[str] = []

            title_error: Optional[str] = None
            text_items_error: Optional[str] = None
            ocr_error: Optional[str] = None
            save_error: Optional[str] = None

            # ---- Titles (OpenAI vision) ----
            try:
                png_bytes = self.render_title_band_png_bytes(i)
                titles = self.ask_model_for_titles(png_bytes)
            except Exception as e:
                title_error = str(e)
                print(f"TITLE ERROR: {title_error}")

            # ---- Text Items (PDF text layer) ----
            try:
                text_items = self.extract_pdf_text_items(i)
            except Exception as e:
                text_items_error = str(e)
                print(f"TEXT ITEMS ERROR: {text_items_error}")
                
            # ---- OCR (always) ----
            try:
                full_png = self.render_full_page_png_bytes(i)
                ocr_items = self.ocr_png_bytes(full_png)
            except Exception as e:
                ocr_error = str(e)
                print(f"OCR ERROR: {ocr_error}")   
                
            # ---- Always try saving the single-page PDF ----
            pdf_path = ""
            try:
                pdf_path = self.save_single_page_pdf_numeric(i)
            except Exception as e:
                save_error = str(e)

            # Store metadata (always)
            record: Dict[str, Any] = {
                "titles": titles,
                "text_items": text_items,
                "ocr_items": ocr_items,
                "page_index": i,
                "output_pdf": os.path.basename(pdf_path) if pdf_path else "",
                "stats": {
                    "titles": len(titles),
                    "text_items": len(text_items),
                    "ocr_items": len(ocr_items),
                },
            }

            if title_error:
                record["title_error"] = title_error
            if text_items_error:
                record["text_items_error"] = text_items_error
            if ocr_error:
                record["ocr_error"] = ocr_error
            if save_error:
                record["save_error"] = save_error

            self.metadata[page_id] = record

            # Logging
            status_parts = []
            status_parts.append("titles=OK" if not title_error else "titles=ERR")
            status_parts.append("text_items=OK" if not text_items_error else "items=ERR")
            status_parts.append("ocr=OK" if not ocr_error else "ocr=ERR")
            status_parts.append("save=OK" if not save_error else "save=ERR")

            print(
                f"[{'/'.join(status_parts)}] "
                f"Page {i+1}/{total} -> {pdf_path or '(not saved)'} "
                f"| titles={len(titles)} | text_items={len(text_items)} | ocr_items={len(ocr_items)}"
            )

        meta_path = self.write_metadata_json()
        print(f"[DONE] Wrote metadata: {meta_path}")

