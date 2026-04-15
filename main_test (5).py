import asyncio
import concurrent.futures
import hashlib
import logging
import os
import re
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import fitz
import numpy as np
from azure.storage.blob import BlobServiceClient, ContentSettings
from docx import Document
from docx.shared import RGBColor, Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pdf2image import convert_from_bytes
from PIL import Image
from pytesseract import Output
import pytesseract
from sentence_transformers import SentenceTransformer, util
from azure.storage.blob import BlobServiceClient

def get_blob_url(filename: str):
    account_name = "prodgcctbc2024"
    container_name = "tulja2025/Answer_sheets"

    return f"https://{account_name}.blob.core.windows.net/{container_name}/{filename}"

# =============================================================================
# CONFIG & LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    force=True,
    encoding="utf-8",
)
logger = logging.getLogger("htr-prototype")

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME            = os.getenv("AZURE_CONTAINER_NAME", "tulja2025")

if not AZURE_STORAGE_CONNECTION_STRING or not AZURE_CONTAINER_NAME:
    raise RuntimeError(
        "Missing AZURE_STORAGE_CONNECTION_STRING or AZURE_CONTAINER_NAME in environment/.env"
    )

blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client    = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

# Reports folder
REPORTS_DIR = "Reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

COSINE_MODEL    = "all-MiniLM-L6-v2"
VYAKYARTH_MODEL = "krutrim-ai-labs/Vyakyarth"

LLAMA_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LLAMA_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
llm: Optional[object] = None

_LENS_SEMAPHORE: Optional[asyncio.Semaphore] = None
embedder_english: Optional[SentenceTransformer] = None
embedder_indic:   Optional[SentenceTransformer] = None

_cpu_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(4, (os.cpu_count() or 4)),
    thread_name_prefix="cpu",
)
_io_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="io",
)

_feedback_cache:  dict = {}
_embedding_cache: dict = {}

app = FastAPI(title="HTR OCR + Evaluation Report")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  
os.environ["TQDM_DISABLE"] = "1"
os.environ["TQDM_ASCII"] = "true"

# =============================================================================
# MARKS CONFIGURATION  —  4-level + optional support
# =============================================================================
MARKS_CONFIG: dict = {
    "ENGLISH-2026-EN": {

        "1": {
            "A": {
                "type": "attempt_any",
                "pick": 4,
                "children": {
                    "1": {"type": "mandatory", "marks": 2,
                          "label": "Complete words (i–iv)"},
                    "2": {"type": "mandatory", "marks": 2,
                          "label": "Alphabetical order (i–ii)"},
                    "3": {"type": "mandatory", "marks": 2,
                          "label": "Punctuate (i–ii)"},
                    "4": {"type": "mandatory", "marks": 2,
                          "label": "Make four words from 'Championship'"},
                    "5": {"type": "mandatory", "marks": 2,
                          "label": "Write related words (web diagram)"},
                    "6": {"type": "mandatory", "marks": 2,
                          "label": "Word-chain of Adjectives"},
                },
            },
            "B": {
                "type": "mandatory",
                "total_marks": 2,
                "children": {
                    "1": {
                        "type": "optional_group",
                        "marks": 1,
                        "options": ["a", "b"],
                        "children": {
                            "a": {"type": "mandatory", "marks": 1,
                                  "label": "Sentence using 'at the edge of'"},
                            "b": {"type": "mandatory", "marks": 1,
                                  "label": "Add clause to 'I know.'"},
                        },
                    },
                    "2": {
                        "type": "optional_group",
                        "marks": 1,
                        "options": ["a", "b"],
                        "children": {
                            "a": {"type": "mandatory", "marks": 1,
                                  "label": "Prefix/suffix for responsible, continue"},
                            "b": {"type": "mandatory", "marks": 1,
                                  "label": "Sentence using responsible/continue"},
                        },
                    },
                },
            },
        },

        "2": {
            "A": {
                "type": "mandatory",
                "children": {
                    "A1": {"type": "mandatory", "marks": 2, "label": "Complete sentences (i–iv)"},
                    "A2": {"type": "mandatory", "marks": 2, "label": "Complete web – hibiscus flower"},
                    "A3": {"type": "mandatory", "marks": 2, "label": "Describing words for nouns (i–iv)"},
                    "A4": {"type": "mandatory", "marks": 2, "label": "Do as directed (tense + gerund)"},
                    "A5": {"type": "mandatory", "marks": 2, "label": "Personal response – Nature is best teacher"},
                },
            },
            "B": {
                "type": "mandatory",
                "children": {
                    "B1": {"type": "mandatory", "marks": 2, "label": "True or False – langur passage (i–iv)"},
                    "B2": {"type": "mandatory", "marks": 2, "label": "Give reasons – narrator rushed to veranda"},
                    "B3": {"type": "mandatory", "marks": 2, "label": "Antonyms from extract (i–iv)"},
                    "B4": {"type": "mandatory", "marks": 2, "label": "Do as directed (not only–but also + passive)"},
                    "B5": {"type": "mandatory", "marks": 2, "label": "Personal response – injured animal"},
                },
            },
        },

        "3": {
            "A": {
                "type": "mandatory",
                "children": {
                    "A1": {"type": "mandatory", "marks": 2, "label": "True or False – O Captain (i–iv)"},
                    "A2": {"type": "mandatory", "marks": 2, "label": "Complete web – Condition of captain"},
                    "A3": {"type": "mandatory", "marks": 1, "label": "Rhyming words (bed/kill/fun/bread)"},
                },
            },
            "B": {
                "type": "mandatory",
                "children": {
                    "appreciation": {
                        "type": "mandatory", "marks": 5,
                        "label": "Poem appreciation – The Twins (title/poet/rhyme/figure/theme)",
                        "children": {
                            "title":  {"type": "mandatory", "marks": 0.5, "label": "Title"},
                            "poet":   {"type": "mandatory", "marks": 0.5, "label": "Name of poet"},
                            "rhyme":  {"type": "mandatory", "marks": 1,   "label": "Rhyme scheme"},
                            "figure": {"type": "mandatory", "marks": 1,   "label": "Figure of speech"},
                            "theme":  {"type": "mandatory", "marks": 2,   "label": "Theme/Central idea"},
                        },
                    },
                },
            },
        },

        "4": {
            "A": {
                "type": "mandatory",
                "children": {
                    "A1": {"type": "mandatory", "marks": 2, "label": "Complete sentences – Swapnil Kusale (i–iv)"},
                    "A2": {"type": "mandatory", "marks": 2, "label": "Rearrange events in order (i–iv)"},
                    "A3": {"type": "mandatory", "marks": 2, "label": "Match synonyms (i–iv)"},
                    "A4": {"type": "mandatory", "marks": 2, "label": "Do as directed (Wh-question + question tag)"},
                    "A5": {"type": "mandatory", "marks": 2, "label": "Personal response – favourite game"},
                },
            },
            "B": {
                "type": "mandatory",
                "children": {
                    "summary": {"type": "mandatory", "marks": 5,
                                "label": "Summary writing with suitable title"},
                },
            },
        },

        "5": {
            "A": {
                "type": "optional_group",
                "marks": 5,
                "options": ["A1", "A2"],
                "children": {
                    "A1": {"type": "mandatory", "marks": 5,
                           "label": "Informal Letter – appeal friend to visit exhibition"},
                    "A2": {"type": "mandatory", "marks": 5,
                           "label": "Formal Letter – thank Youth Club president"},
                },
            },
            "B": {
                "type": "optional_group",
                "marks": 5,
                "options": ["B1", "B2"],
                "children": {
                    "B1": {
                        "type": "mandatory", "marks": 5,
                        "label": "Dialogue writing",
                        "children": {
                            "a": {"type": "mandatory", "marks": 1,
                                  "label": "Rearrange jumbled dialogue"},
                            "b": {"type": "mandatory", "marks": 1,
                                  "label": "Complete dialogue about favourite actor"},
                            "c": {"type": "mandatory", "marks": 3,
                                  "label": "Write dialogue – Importance of Computer Education"},
                        },
                    },
                    "B2": {"type": "mandatory", "marks": 5,
                           "label": "Drafting a speech – Books are our real friends"},
                },
            },
        },

        "6": {
            "A": {
                "type": "optional_group",
                "marks": 5,
                "options": ["A1", "A2"],
                "children": {
                    "A1": {"type": "mandatory", "marks": 5,
                           "label": "Non-verbal to verbal – Effective/Ineffective communication table"},
                    "A2": {"type": "mandatory", "marks": 5,
                           "label": "Verbal to non-verbal – Energy tree diagram"},
                },
            },
            "B": {
                "type": "optional_group",
                "marks": 5,
                "options": ["B1", "B2"],
                "children": {
                    "B1": {"type": "mandatory", "marks": 5,
                           "label": "News Report – Nav Bharat School Science Day"},
                    "B2": {"type": "mandatory", "marks": 5,
                           "label": "Story development – 'In the last summer vacation…'"},
                },
            },
        },

        "7": {
            "": {
                "type": "mandatory",
                "children": {
                    "a": {
                        "type": "attempt_any",
                        "pick": 4,
                        "total_marks": 2,
                        "children": {
                            "1": {"type": "mandatory", "marks": 0.5, "label": "Translate: Space"},
                            "2": {"type": "mandatory", "marks": 0.5, "label": "Translate: Promise"},
                            "3": {"type": "mandatory", "marks": 0.5, "label": "Translate: Skill"},
                            "4": {"type": "mandatory", "marks": 0.5, "label": "Translate: Pretend"},
                            "5": {"type": "mandatory", "marks": 0.5, "label": "Translate: Boundary"},
                            "6": {"type": "mandatory", "marks": 0.5, "label": "Translate: Natural"},
                        },
                    },
                    "b": {
                        "type": "attempt_any",
                        "pick": 2,
                        "total_marks": 2,
                        "children": {
                            "1": {"type": "mandatory", "marks": 1, "label": "Translate: Take care of health"},
                            "2": {"type": "mandatory", "marks": 1, "label": "Translate: Wash hands before eating"},
                            "3": {"type": "mandatory", "marks": 1, "label": "Translate: Avoid readymade food"},
                            "4": {"type": "mandatory", "marks": 1, "label": "Translate: Eat green vegetables"},
                        },
                    },
                    "c": {
                        "type": "attempt_any",
                        "pick": 1,
                        "total_marks": 1,
                        "children": {
                            "1": {"type": "mandatory", "marks": 1, "label": "Translate: An empty vessel…"},
                            "2": {"type": "mandatory", "marks": 1, "label": "Translate: Truth is always bitter"},
                        },
                    },
                },
            },
        },
    },
    # =============================================================================
    # HINDI-2026-HI
    # =============================================================================
    "HINDI-2026-HI": {
        "1": {
            "type": "mandatory",
            "children": {
                "A": {"type": "mandatory", "marks": 8, "label": "पठित गद्यांश - गोवा"},
                "B": {"type": "mandatory", "marks": 8, "label": "पठित गद्यांश - काका कालेलकर पत्र"},
                "C": {"type": "mandatory", "marks": 4, "label": "अपठित गद्यांश - विज्ञापन"},
            },
        },

        "2": {
            "type": "mandatory",
            "children": {
                "A": {"type": "mandatory", "marks": 6, "label": "पठित पद्यांश - अपनी गंध नहीं बेचूँगा"},
                "B": {"type": "mandatory", "marks": 6, "label": "पठित पद्यांश - मीरा बाई"},
            },
        },

        "3": {
            "type": "mandatory",
            "children": {
                "A": {"type": "mandatory", "marks": 4, "label": "पठित गद्यांश - सिरचन"},
                "B": {"type": "mandatory", "marks": 4, "label": "पठित पद्यांश - हम उस धरती के लड़के हैं"},
            },
        },

        "4": {
            "type": "mandatory",
            "total_marks": 14,
            "label": "व्याकरण",
        },

        "5": {
            "type": "mandatory",
            "children": {
                "A": {
                    "type": "optional_group",
                    "marks": 5,
                    "options": ["1", "2"],
                    "children": {
                        "1": {"type": "mandatory", "marks": 5, "label": "वधाई पत्र"},
                        "2": {"type": "mandatory", "marks": 5, "label": "शिकायत पत्र"},
                    },
                },
                "B": {"type": "mandatory", "marks": 4, "label": "गद्य आकलन - प्रश्न निर्मित"},
                "C": {
                    "type": "optional_group",
                    "marks": 5,
                    "options": ["1", "2"],
                    "children": {
                        "1": {"type": "mandatory", "marks": 5, "label": "वृत्तांत लेखन"},
                        "2": {"type": "mandatory", "marks": 5, "label": "कहानी लेखन"},
                    },
                },
                "D": {"type": "mandatory", "marks": 5, "label": "विज्ञापन लेखन"},
                "E": {
                    "type": "optional_group",
                    "marks": 7,
                    "options": ["1", "2", "3"],
                    "children": {
                        "1": {"type": "mandatory", "marks": 7, "label": "निबंध - जल है तो कल है"},
                        "2": {"type": "mandatory", "marks": 7, "label": "निबंध - आजादी का अमृत महोत्सव"},
                        "3": {"type": "mandatory", "marks": 7, "label": "निबंध - पेड़ की आत्मकथा"},
                    },
                },
            },
        },
    },
}

# ── Convenience helpers ───────────────────────────────────────────────────────

def extract_text_from_typed_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from the Model Answer PDF (typed/printed text).
    Uses PyMuPDF (fitz) which is fast, accurate, and needs no image conversion
    for digitally-typeset PDFs.  Falls back to Tesseract only if fitz returns
    very little text (i.e. the PDF is scanned).
    """
    try:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        logger.info(f"[Model PDF] fitz extracted {len(text)} chars over {len(doc)} pages.")
        if len(text.strip()) > 100:
            return text
    except Exception as e:
        logger.error(f"PyMuPDF failed on model PDF: {e}")
        text = ""

    # Fallback: scanned model PDF — use Tesseract
    logger.warning("[Model PDF] fitz returned little text; falling back to Tesseract.")
    try:
        images = convert_from_bytes(
            pdf_bytes, dpi=200, poppler_path=POPPLER_PATH, thread_count=4
        )
        pages = []
        for img in images:
            page_img = np.array(img.convert("RGB"))[:, :, ::-1]
            gray     = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
            pages.append(
                pytesseract.image_to_string(gray, config="--psm 6 --oem 3")
            )
        return "\n\n".join(pages).strip()
    except Exception as e:
        logger.error(f"Tesseract fallback failed on model PDF: {e}")
        raise


def clean_and_structure_text(text: str) -> str:
    """Clean OCR output and preserve line structure"""
    if not text:
        return ""
   
    lines = text.split('\n')
    cleaned = []
   
    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append("")
            continue
           
        line = (line.replace("â€¦", "...")
                    .replace("Û±Û·", "→")
                    .replace("ã‚·", "→")
                    .replace("Q.1A)", "Q.1 A)")
                    .replace("Sec t", "Sect")
                    .replace("karre Nagar", "Karve Nagar")
                    .replace("dogign", "doing")
                    .replace("intresting", "interesting"))
       
        cleaned.append(line)
   
    return "\n".join(cleaned)

def fix_broken_words(text: str) -> str:
    lines = text.split("\n")
    fixed_lines = []

    for i in range(len(lines)):
        line = lines[i].strip()

        # If line is too short → likely broken word
        if i > 0 and len(line) <= 3 and line.isalpha():
            fixed_lines[-1] += line  # join to previous line
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_dialogues(text: str) -> str:
    text = re.sub(
        r"([A-Za-z]\s*:\s*[^:\n]+)\s+([A-Za-z]\s*:)",
        r"\1\n\2",
        text
    )
    return text

def preserve_line_structure(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for i, line in enumerate(lines):
        line = line.strip()

        if not line:
            cleaned.append("")
            continue

        # Rule 1: Keep dialogues separate
        if re.match(r"^[A-Za-z]\s*[:\-]", line):
            cleaned.append(line)
            continue

        # Rule 2: Keep numbered/bullets separate
        if re.match(r"^(\(?\d+\)|\d+\.|\(?[ivx]+\))", line, re.I):
            cleaned.append(line)
            continue

        # Rule 3: Merge only broken small lines
        if cleaned and len(line) < 4:
            cleaned[-1] += " " + line
        else:
            cleaned.append(line)

    return "\n".join(cleaned)

def fix_dialogues(text: str) -> str:
    text = re.sub(
        r"([A-Za-z]\s*:\s*[^:\n]+)\s+([A-Za-z]\s*:)",
        r"\1\n\2",
        text
    )
    return text

def _iter_leaves(node: dict, path: tuple = ()) -> list[tuple]:
    """
    Recursively yield (path, leaf_node) for every scoreable leaf.
    """
    if "marks" in node and "children" not in node:
        return [(path, node)]
    results = []
    children = node.get("children", {})
    for key, child in children.items():
        results.extend(_iter_leaves(child, path + (key,)))
    return results


def get_marks_config(paper_code: str) -> dict:
    config = MARKS_CONFIG.get(paper_code.upper().strip())
    if config is None:
        logger.warning(f"No marks config for '{paper_code}'.")
        return {}
    return config

def get_paper_total_marks(paper_code: str) -> float:
    config = get_marks_config(paper_code)
    if not config:
        return 0.0

    total = 0.0
    for q_node in config.values():
        q_total = _sum_marks(q_node)
        total += q_total

    return round(total, 1)


def _sum_marks(node: dict) -> float:
    """Robust version that correctly calculates total marks for both English & Hindi papers"""
    if not isinstance(node, dict):
        return 0.0

    if "marks" in node and "children" not in node:
        return float(node["marks"])

    total = 0.0
    node_type = node.get("type", "mandatory")

    if "type" not in node and "marks" not in node and "children" not in node:
        for child in node.values():
            total += _sum_marks(child)
        return total

    children = node.get("children", {})

    if node_type == "attempt_any":
        pick = node.get("pick", 1)
        if children:
            per_mark = next(iter(children.values())).get("marks", 0)
            total += pick * per_mark

    elif node_type == "optional_group":
        total += float(node.get("marks", 0))

    elif "total_marks" in node:
        total += float(node["total_marks"])

    else:
        for child in children.values():
            total += _sum_marks(child)

    return total

def list_available_papers() -> list[str]:
    return sorted(MARKS_CONFIG.keys())


# =============================================================================
# STARTUP — load all ML models
# =============================================================================
@app.on_event("startup")
async def startup_tasks():
    global _LENS_SEMAPHORE, embedder_english, embedder_indic, llm
    _LENS_SEMAPHORE = asyncio.Semaphore(12)
    loop = asyncio.get_event_loop()

    async def _load_english_embedder():
        global embedder_english
        logger.info(f"Loading English embedder '{COSINE_MODEL}'…")
        embedder_english = await loop.run_in_executor(
            _cpu_pool,
            lambda: SentenceTransformer(
                COSINE_MODEL,
                local_files_only=False,
            )
        )
        logger.info("[OK] English embedder ready.")

    async def _load_indic_embedder():
        global embedder_indic
        logger.info(f"Loading Vyakyarth model '{VYAKYARTH_MODEL}'…")
        embedder_indic = await loop.run_in_executor(
            _cpu_pool, lambda: SentenceTransformer(VYAKYARTH_MODEL)
        )
        logger.info("[OK] Vyakyarth embedder ready.")

    async def _load_llama():
        global llm
        logger.info(f"Loading {LLAMA_FILE}…")
        def _load():
            try:
                from llama_cpp import Llama
                from huggingface_hub import hf_hub_download
                mp = hf_hub_download(repo_id=LLAMA_REPO, filename=LLAMA_FILE,
                                     local_dir="models", resume_download=True)
                return Llama(model_path=mp, n_ctx=1536, n_batch=1024,
                             n_threads=os.cpu_count() or 8, n_gpu_laayers=0, verbose=False)
            except Exception as e:
                logger.error(f"Llama load failed: {e}")
                return None
        llm = await loop.run_in_executor(_cpu_pool, _load)
        if llm:
            logger.info("[OK] Llama loaded.")

    await asyncio.gather(_load_english_embedder(), _load_indic_embedder(), _load_llama())
    logger.info("[OK] All startup tasks complete.")


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================
def normalize_devanagari_digits(text: str) -> str:
    return text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))

_HINDI_SUB_MAP = {"अ": "a", "ब": "b", "क": "c", "ड": "d", "इ": "e", "फ": "f"}

def _normalize_sub_letter(text: str) -> str:
    for m, e in _HINDI_SUB_MAP.items():
        text = text.replace(m, e)
    return text

def detect_script(text: str) -> str:
    sample = text[:500]
    deva  = sum(1 for ch in sample if "\u0900" <= ch <= "\u097F")
    latin = sum(1 for ch in sample if ch.isalpha() and ch.isascii())
    total = deva + latin
    if total == 0: return "unknown"
    if deva  / total >= 0.60: return "devanagari"
    if latin / total >= 0.60: return "latin"
    return "mixed"

def label_language(text: str) -> str:
    return {
        "devanagari": "Hindi / Marathi (Devanagari)",
        "latin":      "English (Latin script)",
        "mixed":      "Mixed script",
    }.get(detect_script(text), "Unknown")

def check_language_compatibility(ref: str, ext: str) -> list[str]:
    warnings = []
    r, e = detect_script(ref), detect_script(ext)
    if r == "latin"      and e == "devanagari": warnings.append("⚠️ Reference English but answer Devanagari.")
    if r == "devanagari" and e == "latin":       warnings.append("⚠️ Reference Devanagari but answer English.")
    return warnings

def route_comparison(ref: str, ext: str) -> str:
    return "vyakyarth" if (detect_script(ref) == "devanagari" and detect_script(ext) == "devanagari") else "english"


# =============================================================================
# QUESTION MARKER DETECTION  — supports 4-level addressing
# =============================================================================
_Q_PATTERN = re.compile(
    r"""
    (?:
        Q(?:ue(?:stion)?)?[\.\s]*
        (?P<en_q>\d+)
        [\.\s]*
        (?:\((?P<en_sub>[A-Fa-f])\)|(?P<en_sub2>[A-Fa-f]))?
        [\.\s]*
        (?:\((?P<en_ss>[A-Za-z0-9]+)\)|(?P<en_ss2>[A-Za-z0-9]+))?
        [\.\s]*
        (?:\((?P<en_item>[ivxlcdmIVXLCDM]+|[a-f]|\d)\))?
    )
    |
    (?:
        (?:प्र(?:श्न)?|उत्तर)[\.\s:]*
        (?P<mr_q>\d+)
        \s*
        (?P<mr_sub>[a-f]?)
        \s*
        (?P<mr_ss>[A-Za-z0-9]*)
        \s*
        (?:\((?P<mr_item>[ivxIVX]+|[a-f]|\d)\))?
    )
    |
    (?:
        ^[\(\[]?(?P<act_sub>[AB])(?P<act_ss>\d+)[\)\]]?[\.:\s]
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE | re.UNICODE,
)


def detect_question_markers(page_text: str, page_num: int) -> list[dict]:
    text = normalize_devanagari_digits(page_text)
    text = _normalize_sub_letter(text)
    results, seen = [], set()

    for m in _Q_PATTERN.finditer(text):
        g = m.groupdict()

        q_raw   = g.get("en_q") or g.get("mr_q")
        sub_raw = (g.get("en_sub") or g.get("en_sub2") or
                   g.get("mr_sub") or g.get("act_sub") or "").upper().strip()
        ss_raw  = (g.get("en_ss") or g.get("en_ss2") or
                   g.get("mr_ss") or g.get("act_ss") or "").strip()
        it_raw  = (g.get("en_item") or g.get("mr_item") or "").strip()

        if not q_raw and g.get("act_sub"):
            q_raw  = "0"
            sub_raw = g["act_sub"].upper()
            ss_raw  = g["act_ss"]

        if not q_raw:
            continue

        key = (q_raw, sub_raw, ss_raw, it_raw)
        if key not in seen:
            seen.add(key)
            results.append({
                "q_num":   q_raw.strip(),
                "sub_q":   sub_raw,
                "sub_sub": ss_raw,
                "item":    it_raw,
                "pos":     m.start(),
                "page":    page_num,
            })
    return results


# =============================================================================
# ANSWER STITCHING  — 4-level dict
# =============================================================================
def stitch_student_answers(page_texts: dict) -> dict:
    """
    Returns:
        {q_num: {sub_q: {sub_sub: {item: text}}}}
    Empty string keys mean the level is absent.
    """
    all_markers = []
    for pn in sorted(page_texts.keys()):
        all_markers.extend(detect_question_markers(page_texts[pn], pn))

    if not all_markers:
        full = "\n".join(page_texts[p] for p in sorted(page_texts))
        return {"0": {"": {"": {"": full}}}}

    raw: dict[tuple, str] = {}

    for i, marker in enumerate(all_markers):
        q, sq, ss, it = marker["q_num"], marker["sub_q"], marker["sub_sub"], marker["item"]
        page, pos = marker["page"], marker["pos"]

        if i + 1 < len(all_markers) and all_markers[i + 1]["page"] == page:
            segment = page_texts[page][pos: all_markers[i + 1]["pos"]]
        else:
            segment = page_texts[page][pos:]
            last_page = all_markers[i + 1]["page"] if i + 1 < len(all_markers) else max(page_texts)
            for mid in range(page + 1, last_page):
                segment += "\n" + page_texts.get(mid, "")
            if i + 1 < len(all_markers):
                nxt = all_markers[i + 1]
                segment += "\n" + page_texts[nxt["page"]][: nxt["pos"]]

        key = (q, sq, ss, it)
        raw[key] = raw.get(key, "") + "\n" + segment

    result: dict = {}
    for (q, sq, ss, it), text in raw.items():
        result.setdefault(q, {}).setdefault(sq, {}).setdefault(ss, {})[it] = text.strip()
    return result


# =============================================================================
# MODEL ANSWER EXTRACTION  — also 4-level
# =============================================================================
def parse_model_answers(text: str) -> dict:
    """
    Parse the model answer PDF into the same 4-level dict structure as student answers.
    """
    text = normalize_devanagari_digits(text)
    text = _normalize_sub_letter(text)
    markers = []
    seen    = set()

    for m in _Q_PATTERN.finditer(text):
        g = m.groupdict()
        q_raw   = g.get("en_q") or g.get("mr_q")
        sub_raw = (g.get("en_sub") or g.get("en_sub2") or g.get("mr_sub") or g.get("act_sub") or "").upper()
        ss_raw  = (g.get("en_ss") or g.get("en_ss2") or g.get("mr_ss") or g.get("act_ss") or "").strip()
        it_raw  = (g.get("en_item") or g.get("mr_item") or "").strip()
        if not q_raw: continue
        key = (q_raw.strip(), sub_raw, ss_raw, it_raw)
        if key not in seen:
            seen.add(key)
            markers.append({"q_num": q_raw.strip(), "sub_q": sub_raw,
                            "sub_sub": ss_raw, "item": it_raw, "pos": m.start()})

    result: dict = {}
    for i, mk in enumerate(markers):
        q, sq, ss, it = mk["q_num"], mk["sub_q"], mk["sub_sub"], mk["item"]
        start = mk["pos"]
        end   = markers[i + 1]["pos"] if i + 1 < len(markers) else len(text)
        result.setdefault(q, {}).setdefault(sq, {}).setdefault(ss, {})[it] = text[start:end].strip()

    logger.info(f"[Model PDF] Parsed {len(result)} questions.")
    return result


# =============================================================================
# OPTIONAL QUESTION RESOLUTION
# =============================================================================
def resolve_optional(config_node: dict, student_q: dict) -> list[str]:
    attempted = []
    for opt in config_node.get("options", []):
        sub_answers = student_q.get(opt, {})
        if isinstance(sub_answers, dict):
            if any(t.strip() for v in sub_answers.values()
                   for t in (v.values() if isinstance(v, dict) else [v])):
                attempted.append(opt)
        elif isinstance(sub_answers, str) and sub_answers.strip():
            attempted.append(opt)
    return attempted


def pick_best_optional(
    config_node: dict,
    attempted_opts: list[str],
    student_answers: dict,
    model_answers: dict,
) -> Optional[str]:
    if not attempted_opts:
        return None
    if len(attempted_opts) == 1:
        return attempted_opts[0]

    best_opt, best_score = None, -1.0
    for opt in attempted_opts:
        s_text = _flatten_text(student_answers.get(opt, {}))
        m_text = _flatten_text(model_answers.get(opt, {}))
        if not m_text:
            continue
        score = compare_texts(s_text, m_text)["composite_score"]
        if score > best_score:
            best_score, best_opt = score, opt
    return best_opt or attempted_opts[0]


def _flatten_text(node) -> str:
    """Flatten any nested dict / str into a single string."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return " ".join(_flatten_text(v) for v in node.values())
    return ""


# =============================================================================
# ATTEMPT_ANY RESOLUTION
# =============================================================================
def resolve_attempt_any(config_node: dict, student_sub: dict, model_sub: dict) -> list[str]:
    pick      = config_node.get("pick", len(config_node.get("children", {})))
    attempted = {k: _flatten_text(v) for k, v in student_sub.items() if _flatten_text(v).strip()}

    if len(attempted) <= pick:
        return list(attempted.keys())

    scored = []
    for k, s_text in attempted.items():
        m_text = _flatten_text(model_sub.get(k, ""))
        score  = compare_texts(s_text, m_text)["composite_score"] if m_text else 0.0
        scored.append((score, k))
    scored.sort(reverse=True)
    return [k for _, k in scored[:pick]]


# =============================================================================
# BLOB HELPERS
# =============================================================================
def _upload_blob_sync(blob_name: str, content: bytes, content_type: str = "application/octet-stream") -> str:
    bc = container_client.get_blob_client(blob_name)
    bc.upload_blob(content, overwrite=True, content_settings=ContentSettings(content_type=content_type))
    return bc.url

async def upload_blob_async(blob_name, content, content_type="application/octet-stream"):
    return await asyncio.get_event_loop().run_in_executor(_io_pool, _upload_blob_sync, blob_name, content, content_type)

def get_blob_url(blob_name: str) -> str:
    return container_client.get_blob_client(blob_name).url

def _download_blob_bytes_sync(blob_name: str) -> Optional[bytes]:
    try:
        bc = container_client.get_blob_client(blob_name)
        return bc.download_blob().readall() if bc.exists() else None
    except Exception:
        return None

async def download_blob_bytes_async(blob_name: str) -> Optional[bytes]:
    return await asyncio.get_event_loop().run_in_executor(_io_pool, _download_blob_bytes_sync, blob_name)


# =============================================================================
# EMBEDDING CACHE
# =============================================================================
def _get_cached_embedding(text: str):
    return _embedding_cache.get(hashlib.md5(text.encode()).hexdigest())

def _cache_embedding(text: str, emb):
    key = hashlib.md5(text.encode()).hexdigest()
    _embedding_cache[key] = emb
    if len(_embedding_cache) > 500:
        del _embedding_cache[next(iter(_embedding_cache))]


# =============================================================================
# SIMILARITY ENGINES
# =============================================================================
def _cosine_sim(model, a: str, b: str) -> float:
    e1 = _get_cached_embedding(a)
    e2 = _get_cached_embedding(b)
    if e1 is None:
        e1 = model.encode(a, convert_to_tensor=True, show_progress_bar=False); _cache_embedding(a, e1)
    if e2 is None:
        e2 = model.encode(b, convert_to_tensor=True, show_progress_bar=False); _cache_embedding(b, e2)
    return round(float(util.cos_sim(e1, e2)[0][0]) * 100, 1)

def cosine_score_english(a, b):
    global embedder_english
    if embedder_english is None: embedder_english = SentenceTransformer(COSINE_MODEL)
    return _cosine_sim(embedder_english, a, b)

def cosine_score_vyakyarth(a, b):
    global embedder_indic
    if embedder_indic is None: embedder_indic = SentenceTransformer(VYAKYARTH_MODEL)
    return _cosine_sim(embedder_indic, a, b)

def get_vyakyarth_verdict(score: float) -> tuple:
    if   score >= 90: return ("उत्कृष्ट – मूलतः समान", "A+")
    elif score >= 75: return ("अच्छी समानता",           "A")
    elif score >= 55: return ("मध्यम समानता",            "B")
    elif score >= 30: return ("कम समानता",               "C")
    else:             return ("बहुत कम समानता",           "D")

def compare_texts(student: str, reference: str) -> dict:
    if "[Lens OCR failed" in student or not student.strip():
        return {"cosine_pct": 0.0, "word_overlap_pct": 0.0, "length_ratio_pct": 0.0,
                "composite_score": 0.0, "engine": "none",
                "ref_language": label_language(reference), "student_language": "N/A",
                "language_warnings": [], "vyakyarth_verdict": None, "vyakyarth_grade": None}

    engine  = route_comparison(reference, student)
    warns   = check_language_compatibility(reference, student)
    cs = max(0.0, cosine_score_vyakyarth(student, reference) if engine == "vyakyarth"
             else cosine_score_english(student, reference))

    sw      = set(re.findall(r"\w+", student.lower()))
    rw      = set(re.findall(r"\w+", reference.lower()))
    jaccard = round((len(sw & rw) / len(sw | rw) * 100) if (sw | rw) else 0.0, 1)
    lr_pct  = round(min(len(student) / max(len(reference), 1), 1.0) * 100, 1)
    composite = round(0.70 * cs + 0.20 * jaccard + 0.10 * lr_pct, 1)

    vy_v = vy_g = None
    if engine == "vyakyarth":
        vy_v, vy_g = get_vyakyarth_verdict(cs)

    return {"cosine_pct": cs, "word_overlap_pct": jaccard, "length_ratio_pct": lr_pct,
            "composite_score": composite, "engine": engine,
            "ref_language": label_language(reference), "student_language": label_language(student),
            "language_warnings": warns, "vyakyarth_verdict": vy_v, "vyakyarth_grade": vy_g}


# =============================================================================
# MARKS CALCULATION
# =============================================================================
def calculate_marks(composite: float, max_marks: float) -> float:
    if   composite >= 85: ratio = 1.0
    elif composite >= 70: ratio = 0.8
    elif composite >= 55: ratio = 0.6
    elif composite >= 35: ratio = 0.4
    else:                 ratio = 0.0
    return round(ratio * max_marks, 1)


# =============================================================================
# DESKEW & GOOGLE LENS OCR
# =============================================================================
def _crop_answer_sheet(img: Image.Image) -> Image.Image:
    """
    Crop printed header/footer zones from Maharashtra board answer sheets
    before sending to OCR to avoid extracting printed text as handwriting.
    """
    w, h = img.size
    top_crop    = int(h * 0.04)
    bottom_crop = int(h * 0.03)
    return img.crop((0, top_crop, w, h - bottom_crop))


def deskew(img: Image.Image) -> Image.Image:
    """
    1. Auto-rotate via Tesseract OSD.
    2. CLAHE contrast on LAB L-channel.
    3. Unsharp-mask sharpening.
    4. Upscale to ≥ 1500 px wide.
    """
    try:
        arr  = np.array(img.convert("RGB"))[:, :, ::-1].copy()
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        try:
            osd    = pytesseract.image_to_osd(
                gray, output_type=Output.DICT,
                config="--psm 0 -c min_characters_to_try=5"
            )
            rotate = int(osd.get("rotate", 0))
            rotmap = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                      270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            if rotate in rotmap:
                arr  = cv2.rotate(arr,  rotmap[rotate])
                gray = cv2.rotate(gray, rotmap[rotate])
        except Exception:
            pass

        lab     = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        arr     = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=2.0)
        arr     = cv2.addWeighted(arr, 1.5, blurred, -0.5, 0)

        h, w = arr.shape[:2]
        if w < 1500:
            scale = 1500 / w
            arr   = cv2.resize(arr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    except Exception:
        return img

from chrome_lens_py import LensAPI
lens_api = LensAPI()

POPPLER_PATH = r"D:\OCR_Prototype\poppler-25.12.0\Library\bin"

def _sanitize_ocr_text(raw: str) -> str:
    """
    Clean Google Lens OCR output for handwritten Indian school answer sheets.
    """
    if not raw:
        return raw

    # Step 1: selective mojibake fix
    _MOJI_RE = re.compile(r"[ÃÂ][^\x00-\x7F]")
    if _MOJI_RE.search(raw):
        try:
            candidate    = raw.encode("latin-1").decode("utf-8")
            orig_non_asc = sum(1 for c in raw       if ord(c) > 127)
            cand_non_asc = sum(1 for c in candidate if ord(c) > 127)
            fixed = candidate if cand_non_asc <= orig_non_asc else raw
        except (UnicodeEncodeError, UnicodeDecodeError):
            fixed = raw
    else:
        fixed = raw

    # Step 2: strip junk codepoints
    fixed = fixed.replace("\ufffd", "").replace("\x00", "")
    fixed = re.sub(r"[\ue000-\uf8ff]", "", fixed)

    # Step 3: strip watermark lines
    fixed = re.sub(r"(?im)^.*scann?ed\s+by\s+\w[\w\s]*scanner.*$", "", fixed)

    # Step 4: strip header/footer grid lines
    _Q_NUM_RE = re.compile(r"\bQ\.?\s*\d", re.I)

    def _is_header_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if re.fullmatch(r"\s*\d{1,2}\s*", stripped):
            return True
        alpha = sum(1 for ch in stripped if ch.isalpha())
        total = len(stripped)
        if total > 4 and alpha / total < 0.25:
            if not _Q_NUM_RE.search(stripped):
                return True
        if re.search(r"\bQ\.?\s*No\.?\b", stripped, re.I) and alpha < 8:
            return True
        return False

    lines = fixed.splitlines()
    start = 0
    for i, ln in enumerate(lines[:4]):
        if _is_header_line(ln):
            start = i + 1
    end = len(lines)
    for i, ln in enumerate(reversed(lines[-3:])):
        if _is_header_line(ln):
            end = len(lines) - i - 1
    lines = lines[start:end]
    fixed = "\n".join(lines)

    # Step 5: normalise arrows
    fixed = re.sub(r"[→➔➜➝⟶⇒▶►]", "", fixed)
    fixed = re.sub(r"[←↑↓↔⇐⇑⇓]",  "",  fixed)
    # Remove standalone junk symbols
    fixed = re.sub(r"(?m)^\s*[>\-|]+\s*$", "", fixed)

    # Step 6: circled / enclosed numerals and roman symbols
    for i in range(20):
        fixed = fixed.replace(chr(0x2460 + i), f"({i + 1})")
    for i in range(20):
        fixed = fixed.replace(chr(0x2474 + i), f"({i + 1})")
    for i in range(10):
        fixed = fixed.replace(chr(0x2776 + i), f"({i + 1})")
    _roman = ["i","ii","iii","iv","v","vi","vii","viii","ix","x","xi","xii"]
    for i, r in enumerate(_roman):
        fixed = fixed.replace(chr(0x2160 + i), f"({r})")
        fixed = fixed.replace(chr(0x2170 + i), f"({r})")

    # Step 7: decide page script
    deva  = sum(1 for ch in fixed if "\u0900" <= ch <= "\u097F")
    latin = sum(1 for ch in fixed if ch.isalpha() and ch.isascii())
    is_devanagari = deva > latin

    # Step 8: drop hallucinated non-ASCII tokens — LINE BY LINE
    if not is_devanagari:
        def _keep_token(tok: str) -> bool:
            alpha = [ch for ch in tok if ch.isalpha()]
            if len(alpha) <= 3:
                return True
            non_asc = sum(1 for ch in alpha if not ch.isascii())
            return (non_asc / len(alpha)) < 0.70

        cleaned = []
        for ln in fixed.splitlines():
            cleaned.append(" ".join(t for t in ln.split() if _keep_token(t)))
        fixed = "\n".join(cleaned)

    # Step 9: collapse excessive blank lines
    fixed = re.sub(r"(\n{3,})", "\n\n", fixed).strip()
    return fixed


async def extract_text_with_lens(image_bytes: bytes, filename: str = "image.jpg") -> str:
    """
    Extract handwritten text from an image using Google Lens OCR.
    This is the sole OCR engine for student handwritten answer sheets.
    """
    global _LENS_SEMAPHORE
    if _LENS_SEMAPHORE is None: _LENS_SEMAPHORE = asyncio.Semaphore(12)
    tmp_dir  = Path("temp_ocr"); tmp_dir.mkdir(exist_ok=True)
    tmp_path = None
    try:
        async with _LENS_SEMAPHORE:
            with tempfile.NamedTemporaryFile(suffix=".png", dir=tmp_dir, delete=False) as f:
                f.write(image_bytes); tmp_path = f.name
            result = await lens_api.process_image(tmp_path)
            raw = (result or {}).get("ocr_text", "") or ""
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            text = _sanitize_ocr_text(raw.strip())
            text = fix_dialogues(text)
            text = preserve_line_structure(text)
            text = clean_and_structure_text(text)
            text = fix_broken_words(text)
            
            if text:
                logger.debug(f"[Lens OCR] {filename}: {len(text)} chars after sanitisation")
            return text or "[No text detected by Google Lens OCR]"
    except Exception as e:
        logger.exception(f"Lens OCR failed for {filename}")
        return f"[Lens OCR failed: {type(e).__name__}]"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except Exception: pass

async def ocr_pdf_to_page_texts(student_pdf_bytes: bytes, base_name: str) -> dict:
    """
    Convert every page of the student PDF to a text string using Google Lens OCR.
    Returns {page_number: text_string} (1-indexed).
    """
    logger.info(f"[OCR PDF] Starting: {base_name}")
    loop     = asyncio.get_running_loop()
    pdf_folder = f"PDF_files/{base_name}/"

    try:
        images = await loop.run_in_executor(
            _cpu_pool,
            lambda: convert_from_bytes(
                student_pdf_bytes, dpi=300,
                poppler_path=POPPLER_PATH, thread_count=6
            )
        )
    except Exception as e:
        raise RuntimeError(f"PDF → image conversion failed: {e}") from e

    page_sem = asyncio.Semaphore(12)

    async def _process_page(i: int, page: Image.Image) -> tuple[int, str]:
        async with page_sem:
            pn       = f"{i:02d}"
            deskewed = await loop.run_in_executor(_cpu_pool, deskew, page)
            cropped  = await loop.run_in_executor(_cpu_pool, _crop_answer_sheet, deskewed)

            buf = BytesIO()
            cropped.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            asyncio.create_task(
                upload_blob_async(f"{pdf_folder}{pn}.png", img_bytes, "image/png")
            )

            # Google Lens OCR — sole engine for handwritten text
            text = await extract_text_with_lens(img_bytes, f"page_{pn}.png")
            logger.info(
                f"[OCR page {pn}] {len(text)} chars | preview: {text[:120]!r}"
            )
            return i, text

    results = await asyncio.gather(
        *[_process_page(i, pg) for i, pg in enumerate(images, 1)]
    )
    results = sorted(results, key=lambda x: x[0])

    asyncio.create_task(
        upload_blob_async(
            f"{pdf_folder}{base_name}.pdf", student_pdf_bytes, "application/pdf"
        )
    )
    return {i: t for i, t in results}


# =============================================================================
# FEEDBACK GENERATION
# =============================================================================
_GRADER_SYSTEM = (
    "You are a fair, helpful, and lenient teacher grading a student's handwritten answer "
    "extracted via OCR. Ignore minor spelling/OCR errors. Focus on conceptual understanding."
)

_UNIFIED_PROMPT = """Grade this student answer against the reference.

Question: {label}
Reference language: {ref_language}  |  Student language: {student_language}  |  Engine: {engine}

REFERENCE:
{reference}

STUDENT ANSWER:
{student}

Semantic Similarity: {cosine_pct}%  |  Keyword Overlap: {kw_pct}%  |  Composite: {comp_score}/100
Max Marks: {max_marks}  |  Awarded: {awarded_marks}
{vyakyarth_info}
**Strengths:**
- [1-2 strengths]

**Issues / Errors:**
- [factual mistakes or "No major errors."]

**Missing Content:**
- [missing key points or "Most key points covered."]

**Suggestions:**
- [1 practical tip]

**Overall:** {awarded_marks}/{max_marks} — [one sentence]
"""

def _rule_based_feedback(metrics: dict, awarded: float, max_m: float, label: str = "") -> str:
    v = ""
    if metrics.get("engine") == "vyakyarth" and metrics.get("vyakyarth_verdict"):
        v = f"\n- Verdict: {metrics['vyakyarth_verdict']} (Grade: {metrics['vyakyarth_grade']})"
    return (f"**Strengths:**\n- Core facts identified{v}\n\n"
            f"**Issues / Errors:**\n- Minor gaps in details\n\n"
            f"**Missing Content:**\n- Some supporting points\n\n"
            f"**Suggestions:**\n- Add more specific details\n\n"
            f"**Overall:** {awarded}/{max_m}")

async def _generate_feedback_llama(student: str, reference: str, metrics: dict,
                                    awarded: float, max_m: float, label: str = "") -> Optional[str]:
    if llm is None or not student.strip() or not reference.strip(): return None
    key = hashlib.md5(f"{student[:300]}{reference[:300]}{max_m}".encode()).hexdigest()
    if key in _feedback_cache: return _feedback_cache[key]
    if metrics.get("cosine_pct", 0) < 15: return None
    vy = (f"- Vyakyarth: {metrics['vyakyarth_verdict']} ({metrics['vyakyarth_grade']})\n"
          if metrics.get("engine") == "vyakyarth" and metrics.get("vyakyarth_verdict") else "")
    prompt = _UNIFIED_PROMPT.format(
        label=label, ref_language=metrics.get("ref_language",""),
        student_language=metrics.get("student_language",""),
        engine=metrics.get("engine","english"),
        reference=reference[:350], student=student[:350],
        cosine_pct=round(metrics.get("cosine_pct",0),1),
        kw_pct=round(metrics.get("word_overlap_pct",0),1),
        comp_score=round(metrics.get("composite_score",0),1),
        max_marks=max_m, awarded_marks=awarded, vyakyarth_info=vy,
    )
    full = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{_GRADER_SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    try:
        out = await asyncio.get_event_loop().run_in_executor(_cpu_pool, lambda: llm(
            full, max_tokens=250, temperature=0.15, top_p=0.88,
            repeat_penalty=1.1, top_k=40, stop=["<|eot_id|>","</s>"], echo=False))
        result = re.sub(r"<\|.*?\|>", "", out["choices"][0]["text"]).strip()
        if len(result) < 50: return None
        _feedback_cache[key] = result
        return result
    except Exception as e:
        logger.error(f"Llama feedback failed: {e}"); return None

async def generate_feedback(student: str, reference: str, metrics: dict,
                             awarded: float = 0.0, max_m: float = 0, label: str = "") -> str:
    if "[Lens OCR failed" in student or not student.strip():
        return f"**Overall:** 0/{max_m} – OCR failed."
    rb = _rule_based_feedback(metrics, awarded, max_m, label)
    if llm is None: return rb
    try:
        fb = await _generate_feedback_llama(student, reference, metrics, awarded, max_m, label)
        if fb: return fb
    except Exception as e:
        logger.warning(f"Llama fallback: {e}")
    return rb


# =============================================================================
# CORE EVALUATION  — walks the 4-level config tree
# =============================================================================
async def evaluate_all_questions(
    student_answers: dict,
    model_answers:   dict,
    marks_config:    dict,
) -> dict:
    question_results = []
    total_scored     = 0.0
    total_possible   = 0.0

    async def _eval_node(node: dict, path: tuple, s_node: dict, m_node: dict):
        nonlocal total_scored, total_possible
        node_type = node.get("type", "mandatory")
        label     = node.get("label", " > ".join(str(p) for p in path))

        # ── LEAF: score directly ───────────────────────────────────────────
        if "marks" in node and "children" not in node:
            max_m = node["marks"]
            total_possible += max_m
            s_text = _flatten_text(s_node)
            m_text = _flatten_text(m_node)
            metrics      = compare_texts(s_text, m_text)
            awarded      = calculate_marks(metrics["composite_score"], max_m) if m_text else 0.0
            structural_penalty = 0.0
            structural_notes   = []
            q_type = node.get("question_type")
           
            if q_type == "letter" and s_text.strip():
                if not re.search(r"(dear|respected)\s+\w+", s_text, re.I):
                    structural_penalty += 1.0
                    structural_notes.append("Missing salutation (Dear / Respected ...)")
                if not re.search(r"(yours (sincerely|faithfully|truly)|warm regards|regards)", s_text, re.I):
                    structural_penalty += 0.5
                    structural_notes.append("Missing closing (Yours sincerely / Regards)")

            elif q_type == "dialogue" and s_text.strip():
                speaker_lines = re.findall(r"^\w[\w\s]*\s*:", s_text, re.M)
                if len(speaker_lines) < 4:
                    structural_penalty += 1.0
                    structural_notes.append(f"Too few speaker turns detected ({len(speaker_lines)}); expected 4+")

            elif q_type == "speech" and s_text.strip():
                if not re.search(r"(respected|dear|ladies and gentlemen|honourable)", s_text, re.I):
                    structural_penalty += 0.5
                    structural_notes.append("Missing formal address (Respected / Ladies and Gentlemen)")
                if not re.search(r"(thank you|jai hind|conclude|in conclusion)", s_text, re.I):
                    structural_penalty += 0.5
                    structural_notes.append("Missing conclusion / closing line")

            elif q_type == "summary" and s_text.strip():
                word_count = len(s_text.split())
                if word_count > max_m * 40:
                    structural_notes.append(f"Summary may be too long ({word_count} words)")
                if not re.search(r"\b(title|heading)\b", s_text, re.I):
                    structural_penalty += 0.5
                    structural_notes.append("No title found for summary")

            awarded = max(0.0, round(awarded - structural_penalty, 1))
            total_scored += awarded
            feedback = await generate_feedback(s_text, m_text, metrics, awarded, max_m, label)
            question_results.append({
                "display_label":    " > ".join(f"Q{p}" if i == 0 else str(p) for i, p in enumerate(path)),
                "label":            label,
                "question_text":    m_text if m_text else "Model answer not available",
                "student_answer":   s_text if s_text.strip() else "[No answer written by the student]",
                "max_marks":        max_m,
                "awarded_marks":    awarded,
                "composite_score":  metrics["composite_score"],
                "semantic_similarity": metrics["cosine_pct"],
                "keyword_overlap":  metrics["word_overlap_pct"],
                "engine":           metrics["engine"],
                "feedback":         feedback,
                "status":           "evaluated" if m_text else "no_model_answer",
                "attempted":        bool(s_text.strip()),
                "language_warnings": metrics.get("language_warnings", []),
                "vyakyarth_verdict": metrics.get("vyakyarth_verdict"),
                "vyakyarth_grade":   metrics.get("vyakyarth_grade"),
            })
            return

        children = node.get("children", {})
        if not children:
            return

        # ── OPTIONAL GROUP ─────────────────────────────────────────────────
        if node_type == "optional_group":
            options   = node.get("options", list(children.keys()))
            attempted = [o for o in options if _flatten_text(s_node.get(o, {})).strip()]

            if not attempted:
                chosen = options[0]
                max_m  = node.get("marks", children[chosen].get("marks", 0))
                total_possible += max_m
                question_results.append({
                    "display_label": " > ".join(str(p) for p in path) + f" [{' OR '.join(options)}]",
                    "label": label,
                    "max_marks": max_m, "awarded_marks": 0.0,
                    "composite_score": 0.0, "semantic_similarity": 0.0, "keyword_overlap": 0.0,
                    "engine": "none", "feedback": f"**Overall:** 0/{max_m} – Not attempted.",
                    "status": "not_attempted", "attempted": False,
                    "language_warnings": [], "vyakyarth_verdict": None, "vyakyarth_grade": None,
                })
                return

            best = pick_best_optional(node, attempted, s_node, m_node)
            child_node = children[best]
            await _eval_node(child_node, path + (best,),
                             s_node.get(best, {}), m_node.get(best, {}))
            return

        # ── ATTEMPT_ANY ────────────────────────────────────────────────────
        if node_type == "attempt_any":
            pick   = node.get("pick", len(children))
            chosen = resolve_attempt_any(node, s_node, m_node)
            scored_count = 0
            for k, child in children.items():
                if k in chosen and scored_count < pick:
                    await _eval_node(child, path + (k,),
                                     s_node.get(k, {}), m_node.get(k, {}))
                    scored_count += 1
                else:
                    max_m = child.get("marks", 0)
                    if k not in chosen:
                        total_possible += max_m
                        question_results.append({
                            "display_label": " > ".join(str(p) for p in path + (k,)),
                            "label": child.get("label", k),
                            "max_marks": max_m, "awarded_marks": 0.0,
                            "composite_score": 0.0, "semantic_similarity": 0.0,
                            "keyword_overlap": 0.0, "engine": "none",
                            "feedback": f"**Overall:** 0/{max_m} – Not selected.",
                            "status": "not_selected", "attempted": False,
                            "language_warnings": [], "vyakyarth_verdict": None, "vyakyarth_grade": None,
                        })
            return

        # ── MANDATORY: recurse into children ──────────────────────────────
        for k, child in children.items():
            await _eval_node(child, path + (k,),
                             s_node.get(k, s_node.get(k.lower(), {})),
                             m_node.get(k, m_node.get(k.lower(), {})))

    for q_num, q_node in marks_config.items():
        s_q = student_answers.get(q_num, {})
        m_q = model_answers.get(q_num, {})
        if "type" not in q_node and "marks" not in q_node and "children" not in q_node:
            synthetic = {"type": "mandatory", "children": q_node}
            await _eval_node(synthetic, (q_num,), s_q, m_q)
        else:
            await _eval_node(q_node, (q_num,), s_q, m_q)

    return {
        "questions":      question_results,
        "total_score":    round(total_scored, 2),
        "total_possible": round(total_possible, 2),
        "percentage":     round((total_scored / total_possible * 100) if total_possible else 0.0, 1),
    }

# =============================================================================
# HELPER: Make sure feedback is always 2-3+ lines
# =============================================================================
def _generate_detailed_feedback(q: dict, awarded: float, max_m: float) -> str:
    ratio = awarded / max_m if max_m > 0 else 0

    if ratio >= 0.80:
        return (f"The student has provided an excellent and well-explained answer. "
                f"Key concepts are clearly understood and presented logically. "
                f"The response is comprehensive and shows strong conceptual clarity.")

    elif ratio >= 0.60:
        return (f"The answer demonstrates a good understanding of the topic. "
                f"Most important points are covered, but a few areas could have been explained in more detail. "
                f"Overall, it is a solid response with room for improvement in depth.")

    elif ratio >= 0.30:
        return (f"The student has attempted the question and mentioned some relevant points. "
                f"However, the explanation is brief and lacks sufficient detail and examples. "
                f"More elaboration on key aspects would significantly improve the quality of the answer.")

    else:
        return (f"The answer is very brief or incomplete. "
                f"Only a few points are mentioned without proper explanation or understanding. "
                f"The student needs to focus more on the core concepts and write detailed answers.")
   
# =============================================================================
# DOCX REPORT
# =============================================================================
def create_questionwise_report(base_name: str, paper_code: str, eval_result: dict,
                               model_answers: dict = None, student_answers: dict = None) -> bytes:
    doc = Document()
   
    title = doc.add_heading("AI Answer Sheet Evaluation", 0)
    title.runs[0].font.color.rgb = RGBColor(255, 255, 255)

    doc.add_paragraph(f"Student ID : {base_name}          Paper Code : {paper_code}")
    doc.add_paragraph(f"Generated : {datetime.now().strftime('%d %B %Y at %I:%M %p')}")
    doc.add_paragraph()

    for q in eval_result.get("questions", []):
        display_label = q.get("display_label", q.get("label", ""))
        awarded = round(q.get("awarded_marks", 0), 1)
        max_m = round(q.get("max_marks", 0), 1)
       
        question_text = q.get("question_text", q.get("label", "Question text not available"))
        student_text = q.get("student_answer", "").strip()
        if not student_text:
            student_text = "[No answer written by the student]"

        doc.add_heading(f"Question: {display_label}", level=2)
        q_para = doc.add_paragraph(question_text)
        q_para.style = "Intense Quote"

        doc.add_heading("Student Answer", level=3)
        doc.add_paragraph(student_text)

        score_para = doc.add_paragraph()
        score_run = score_para.add_run(f"Score: {awarded}/{max_m}")
        score_run.bold = True
        score_run.font.size = Pt(12)
        score_run.font.color.rgb = RGBColor(0, 102, 204)

        doc.add_heading("AI Evaluation Feedback:", level=3)

        feedback = q.get("feedback", "").strip()
        if len(feedback.splitlines()) < 3 or len(feedback) < 120:
            feedback = _generate_detailed_feedback(q, awarded, max_m)

        fb_para = doc.add_paragraph(feedback)
        fb_para.paragraph_format.space_after = Pt(18)

        doc.add_paragraph("_" * 80)
        doc.add_paragraph()

    doc.add_page_break()
    doc.add_heading("Final Summary", level=1)
    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = "Table Grid"
    r = tbl.rows[0].cells
    r[0].text = "Total Score"
    r[1].text = f"{eval_result.get('total_score', 0)} / {eval_result.get('total_possible', 0)}  ({eval_result.get('percentage', 0)}%)"

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()

# =============================================================================
# DOWNLOAD REPORT ENDPOINT
# =============================================================================
@app.get("/download-report/{filename}")
async def download_report(filename: str):
    file_path = os.path.join(REPORTS_DIR, filename)
    
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"detail": "File not found"})

    if filename.endswith(".docx"):
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        media_type = "text/plain"

    # Clean & correct way (no yellow line)
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )  # type: ignore

# =============================================================================
# HELPERS
# =============================================================================
def _sanitize(name: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f/\\:]", "_", name).strip("_")


# =============================================================================
# CORE PIPELINE
# =============================================================================
async def _run_full_evaluation(
    base_name: str,
    student_pdf_bytes: bytes,
    model_pdf_bytes: Optional[bytes],
    paper_code: str,
) -> dict:
   
    loop = asyncio.get_running_loop()

    marks_config = get_marks_config(paper_code)
    if not marks_config:
        return {
            "status": "error",
            "message": f"No marks config for '{paper_code}'. Available: {list_available_papers()}"
        }

    if not model_pdf_bytes:
        model_pdf_bytes = await download_blob_bytes_async(f"Model_answers/{paper_code}.pdf")
        if not model_pdf_bytes:
            return {
                "status": "error",
                "message": f"No model answer PDF for '{paper_code}'."
            }

    try:
        model_raw = await loop.run_in_executor(
            _cpu_pool,
            extract_text_from_typed_pdf,
            model_pdf_bytes
        )

        model_ans = await loop.run_in_executor(
            _cpu_pool,
            parse_model_answers,
            model_raw
        )

        # OCR Student Answer Sheet using Google Lens
        page_texts = await ocr_pdf_to_page_texts(student_pdf_bytes, base_name)

        return {
            "status": "success",
            "base_name": base_name,
            "paper_code": paper_code,
            "model_answers": model_ans,
            "student_pages": page_texts,
            "message": "Evaluation completed successfully"
        }

    except Exception as e:
        logger.error(f"Full evaluation failed for {base_name}: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Evaluation failed: {str(e)}"
        }


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.post("/evaluate")
async def evaluate_paper(request: Request):
    try:
        form = await request.form()
        student_file = form.get("student_pdf")
        if not student_file or not hasattr(student_file, "filename"):
            return JSONResponse(status_code=400, content={"detail": "student_pdf required."})

        paper_code = (form.get("paper_code") or "").strip().upper()
        if not paper_code:
            return JSONResponse(status_code=400, content={
                "detail": "paper_code required.", "available": list_available_papers()})

        student_id = (form.get("student_id") or Path(student_file.filename).stem).strip()
        base_name = _sanitize(student_id)

        s_bytes = await student_file.read()
        if not s_bytes:
            return JSONResponse(status_code=400, content={"detail": "Empty student PDF."})

        m_bytes = None
        mf = form.get("model_pdf")
        if mf and hasattr(mf, "read"):
            m_bytes = await mf.read()

        # Run evaluation
        result = await _run_full_evaluation(base_name, s_bytes, m_bytes, paper_code)

        if result.get("status") != "success":
            return JSONResponse(status_code=400, content=result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_base = f"{base_name}_{paper_code}_{timestamp}"

        # 1. Save Student's Handwritten Extracted Text
        txt_path = os.path.join(REPORTS_DIR, f"{report_base}_student_handwritten.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"STUDENT HANDWRITTEN TEXT - {base_name} - {paper_code}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            for page, text in sorted(result["student_pages"].items()):
                f.write(f"──────────────────── PAGE {page} ────────────────────\n\n")
                f.write(text + "\n\n")

        # 2. Create DOCX Evaluation Report
        docx_bytes = create_questionwise_report(
            base_name=base_name,
            paper_code=paper_code,
            eval_result=result.get("evaluation", {}),
            model_answers=result.get("model_answers"),
            student_answers=result.get("student_stitched", {})
        )

        docx_path = os.path.join(REPORTS_DIR, f"{report_base}_evaluation_report.docx")
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)

        # Final Response with Download Links
        return JSONResponse(status_code=200, content={
            "status": "success",
            "base_name": base_name,
            "paper_code": paper_code,
            "message": "Evaluation completed successfully",
            "student_handwritten_text": {
                "download_url": f"/download-report/{os.path.basename(txt_path)}"
            },
            "evaluation_report": {
                "docx_download_url": f"/download-report/{os.path.basename(docx_path)}"
            },
            "student_pages_preview": {k: v[:250] + "..." for k, v in list(result["student_pages"].items())[:3]}
        })

    except Exception as e:
        import traceback
        logger.error(f"Evaluate failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e), "traceback": traceback.format_exc()}) 
        
@app.post("/upload-model-answer")
async def upload_model_answer(request: Request):
    try:
        form       = await request.form()
        paper_code = (form.get("paper_code") or "").strip().upper()
        model_file = form.get("model_pdf")
        if not paper_code: return JSONResponse(status_code=400, content={"detail": "paper_code required."})
        if not model_file or not hasattr(model_file, "filename"):
            return JSONResponse(status_code=400, content={"detail": "model_pdf required."})
        pdf_bytes = await model_file.read()
        if not pdf_bytes: return JSONResponse(status_code=400, content={"detail": "Empty model PDF."})
        url = await upload_blob_async(f"Model_answers/{paper_code}.pdf", pdf_bytes, "application/pdf")
        return JSONResponse(content={"status": "success", "paper_code": paper_code, "blob_url": url})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/papers")
async def list_papers():
    return {"available_papers": [
        {"paper_code": c, "total_marks": get_paper_total_marks(c), "num_questions": len(MARKS_CONFIG[c])}
        for c in list_available_papers()
    ]}


@app.get("/papers/{paper_code}")
async def get_paper_config(paper_code: str):
    config = get_marks_config(paper_code.upper())
    if not config:
        return JSONResponse(status_code=404, content={"detail": f"'{paper_code}' not found.",
                                                       "available": list_available_papers()})
    return {"paper_code": paper_code.upper(), "total_marks": get_paper_total_marks(paper_code), "config": config}


@app.get("/health")
async def health():
    return {"status": "healthy", "llama_loaded": llm is not None,
            "embedder_english": embedder_english is not None,
            "embedder_vyakyarth": embedder_indic is not None,
            "configured_papers": list_available_papers()}


@app.post("/test-debug")
async def debug_upload(request: Request):
    try:
        form  = await request.form()
        files = {k: getattr(v, "filename", str(v)) for k, v in form.items()}
        return {"fields": list(form.keys()), "files": files}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/get-report-url/{filename}")
async def get_report_url(filename: str):
    url = get_blob_url(filename)
    return {"url": url}


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)