
from __future__ import annotations
import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import OpenAI
from document_labeler import DocumentLabeler

load_dotenv()

if __name__ == "__main__":
    ans = input(
        "Would you like to label / relabel files from sample.pdf into labeled_files/? (y/n): "
    ).strip().lower()

    if ans not in {"y", "yes"}:
        print("Okay — exiting without labeling.")
        raise SystemExit(0)

    document_labeler = DocumentLabeler()
    try:
        document_labeler.run()
    finally:
        document_labeler.close()

