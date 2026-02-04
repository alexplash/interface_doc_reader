
from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Any, Tuple
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np


class DocumentRanker:
    
    def __init__(
        self,
        metadata: Dict[str, Dict[str, Any]],
        model: str = "gpt-5",
        max_items_per_field: int = 120,
        embed_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        embed_threshold: float = 0.70,
    ):
        self.metadata = metadata
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("KEY"))
        self.model = model
        self.max_items_per_field = int(max_items_per_field)

        self.embed_threshold = float(embed_threshold)
        self.embedder = SentenceTransformer(embed_model_name, device="cpu")

        self.system = (
            "You judge whether a single P&ID page is DIRECTLY relevant to query items.\n"
            "Return ONLY JSON: {\"why\": \"...\", \"relevant\": true/false}.\n"
            "Say relevant=true ONLY if the page clearly involves the equipment/service tags in the query.\n"
            "If unsure, return relevant=false."
        )


    def _clip_list(self, xs: List[str], n: int) -> List[str]:
        xs = [x for x in (xs or []) if isinstance(x, str) and x.strip()]
        return xs[:n]

    def _page_items(self, rec: Dict[str, Any]) -> List[str]:
        titles = rec.get("titles", [])
        text_items = rec.get("text_items", [])
        ocr_items = rec.get("ocr_items", [])
        
        return titles + text_items + ocr_items


    _ID_LIKE = re.compile(
        r"(?i)^(?=.*[a-z])(?=.*\d)[a-z0-9]+(?:[\s\-_:\/]+[a-z0-9]+)*$"
    )

    def _is_id_like(self, term: str) -> bool:
        t = (term or "").strip()
        if not t:
            return False
        return bool(self._ID_LIKE.match(t))

    def _split_query_terms(self, query_items: List[str]) -> Tuple[List[str], List[str]]:
        id_terms: List[str] = []
        name_terms: List[str] = []
        for q in (query_items or []):
            if not isinstance(q, str):
                continue
            q = q.strip()
            if not q:
                continue
            if self._is_id_like(q):
                id_terms.append(q)
            else:
                name_terms.append(q)
        return id_terms, name_terms


    def _norm_id(self, s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    def _extract_id_parts(self, term: str) -> Tuple[str, str, List[str]]:
        full = self._norm_id(term)
        if not full:
            return "", "", []

        m = re.match(r"^([a-z]{1,10})", full)
        letters = m.group(1) if m else ""

        nums = re.findall(r"\d+", full)

        return full, letters, nums

    def _page_matches_any_id(self, id_terms: List[str], items: List[str]) -> bool:
        if not id_terms:
            return False

        norm_items = [self._norm_id(x) for x in items if isinstance(x, str) and x.strip()]

        for term in id_terms:
            full, letters, nums = self._extract_id_parts(term)
            if not full:
                continue

            if any(full in it for it in norm_items):
                return True

            if letters and nums:
                letters_present = any(letters in it for it in norm_items)
                if not letters_present:
                    continue

                all_nums_present = all(
                    any(num in it for it in norm_items)
                    for num in nums
                )

                if all_nums_present:
                    return True

        return False

    def _page_matches_any_name(self, name_terms: List[str], items: List[str]) -> bool:
        if not name_terms:
            return False

        items = [x for x in items if isinstance(x, str) and x.strip()]
        if not items:
            return False

        q_vecs = self.embedder.encode(name_terms, convert_to_tensor=True, normalize_embeddings=True)
        item_vecs = self.embedder.encode(items, convert_to_tensor=True, normalize_embeddings=True)

        sims = util.cos_sim(q_vecs, item_vecs)
        best = float(sims.max().item())
        return best >= self.embed_threshold

    def _candidates(self, query_items: List[str]) -> List[str]:
        id_terms, name_terms = self._split_query_terms(query_items)
        print(f"ID TERMS: {id_terms}\n NAME TERMS: {name_terms}")

        candidates: List[str] = []
        for page_id, rec in self.metadata.items():
            items = self._page_items(rec)

            id_hit = self._page_matches_any_id(id_terms, items)
            name_hit = self._page_matches_any_name(name_terms, items) if name_terms else False

            if id_hit or name_hit:
                candidates.append(page_id)
        
        print(f"CANDIDATES: {candidates}")

        return candidates

    def _judge_page(self, query_items: List[str], page_id: str, rec: Dict[str, Any]) -> Dict[str, Any]:
        titles = self._clip_list(rec.get("titles", []) or [], 8)
        text_items = self._clip_list(rec.get("text_items", []) or [], self.max_items_per_field)
        ocr_items = self._clip_list(rec.get("ocr_items", []) or [], self.max_items_per_field)

        payload = {
            "query_items": query_items,
            "page_id": page_id,
            "titles": titles,
            "text_items_sample": text_items,
            "ocr_items_sample": ocr_items,
            "stats": rec.get("stats", {}),
        }

        user_prompt = (
            "Decide if this P&ID page is directly relevant to the query items.\n"
            "Be strict: if it’s only loosely related, return relevant=false.\n"
            "Return ONLY JSON.\n\n"
            f"PAGE:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = (resp.output_text or "").strip()

        try:
            obj = json.loads(raw)
            relevant = bool(obj.get("relevant", False))
            why = obj.get("why", "")
            return {"ok": True, "relevant": relevant, "why": str(why)[:400]}
        except Exception:
            return {"ok": True, "relevant": False, "why": "model_output_not_json"}


    def rank(self, query_items: List[str]) -> List[str]:
        if not query_items:
            return []

        cand_ids = self._candidates(query_items)

        relevant: List[str] = []
        for page_id in cand_ids:
            rec = self.metadata.get(page_id, {})
            verdict = self._judge_page(query_items, page_id, rec)
            if verdict.get("ok") and verdict.get("relevant"):
                relevant.append(page_id)

        return relevant
