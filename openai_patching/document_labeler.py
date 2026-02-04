
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


def norm_str(name: str, max_len: int = 200) -> str:
    name = (name or "").strip()
    name = re.sub(r"[\x00-\x1F]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def extract_first_json_object_from_model(text: str) -> Optional[dict]:
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

    model: str = "gpt-4.1-mini"
    dpi: int = 200
    title_band_ratio: float = 0.22

    def __init__(self) -> None:
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Could not find PDF: {self.pdf_path}")

        os.makedirs(self.out_dir, exist_ok=True)

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.doc = fitz.open(self.pdf_path)

        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

        self.metadata: Dict[str, Dict[str, Any]] = {}

    def close(self) -> None:
        try:
            self.doc.close()
        except Exception:
            pass

    def page_id(self, page_index: int) -> str:
        return f"{page_index + 1:04d}"

    def render_title_band_png_bytes(self, page_index: int) -> bytes:
        page = self.doc[page_index]
        rect = page.rect
        band_h = rect.height * float(self.title_band_ratio)
        clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + band_h)

        pix = page.get_pixmap(dpi=self.dpi, clip=clip)
        return pix.tobytes("png")

    def ask_model_for_titles(self, png_bytes: bytes) -> List[str]:
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

        if not obj or "titles" not in obj:
            maybe = norm_str(raw)
            return [maybe] if maybe else []

        titles = obj.get("titles", [])
        if not isinstance(titles, list):
            return []

        cleaned: List[str] = []
        for t in titles:
            if isinstance(t, str):
                s = norm_str(t)
                if s:
                    cleaned.append(s)

        return cleaned

    def save_single_page_pdf_numeric(self, page_index: int) -> str:
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
        out_path = Path(self.out_dir).resolve()

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
    
    def extract_pdf_text_items(self, page_index: int) -> List[str]:
        page = self.doc[page_index]

        data = page.get_text("dict")

        items = set()

        blocks = data.get("blocks", [])
        for b in blocks:
            if b.get("type") != 0:
                continue

            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    raw = span.get("text", "")
                    if not raw:
                        continue

                    cleaned = norm_str(raw)
                    if not cleaned:
                        continue

                    bbox = span.get("bbox", None)
                    if not bbox or len(bbox) != 4:
                        continue

                    items.add(cleaned)

        return list(items)
    
    def render_full_page_png_bytes(self, page_index: int, max_side_px: int = 3800) -> bytes:
        page = self.doc[page_index]
        rect = page.rect

        max_side_points = max(rect.width, rect.height)
        dpi = int(max_side_px * 72 / max_side_points)
        dpi = max(120, min(400, dpi))  

        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")


    def ocr_png_bytes(self, png_bytes: bytes) -> List[str]:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img)

        result = self.ocr.ocr(arr)

        items: List[Dict[str, Any]] = []
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            for line in result[0]:
                if not (isinstance(line, (list, tuple)) and len(line) >= 2):
                    continue
                box = line[0]
                rec = line[1]

                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    text, conf = rec[0], rec[1]
                else:
                    continue

                text_clean = norm_str(text)
                if not text_clean:
                    continue

                if hasattr(box, "tolist"):
                    box = box.tolist()

                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

                conf_score = float(conf) if conf is not None else None
                if conf_score is not None:
                    if len(text_clean) <= 2 and conf_score < 0.9:
                        continue
                    if conf_score < 0.8:
                        continue
                    else:
                        items.add(text_clean)

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

            try:
                png_bytes = self.render_title_band_png_bytes(i)
                titles = self.ask_model_for_titles(png_bytes)
            except Exception as e:
                title_error = str(e)
                print(f"TITLE ERROR: {title_error}")

            try:
                text_items = self.extract_pdf_text_items(i)
            except Exception as e:
                text_items_error = str(e)
                print(f"TEXT ITEMS ERROR: {text_items_error}")
                
            try:
                full_png = self.render_full_page_png_bytes(i)
                ocr_items = self.ocr_png_bytes(full_png)
            except Exception as e:
                ocr_error = str(e)
                print(f"OCR ERROR: {ocr_error}")   
                
            pdf_path = ""
            try:
                pdf_path = self.save_single_page_pdf_numeric(i)
            except Exception as e:
                save_error = str(e)

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
        

if __name__ == "__main__":
    
    document_labeler = DocumentLabeler()
    
    try:
        document_labeler.run()
    finally:
        document_labeler.close()

