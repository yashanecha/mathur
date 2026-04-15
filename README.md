# Onscreen_evaluation

# AI Answer Sheet Evaluator (HTR + OCR + LLM)

A FastAPI-based intelligent evaluation system for handwritten Maharashtra Board answer sheets using **Google Lens OCR**, semantic similarity (Sentence Transformers), and optional **Llama-3.2** feedback generation.

Supports **English (2026)** and **Hindi (2026)** board exam patterns with complex marking schemes including optional questions, attempt-any, and multi-level sub-questions.

---

## ✨ Features

- **Handwritten Text Recognition (HTR)** using Google Lens OCR (best accuracy for Indian handwriting)
- Advanced **pre-processing**: Deskew, CLAHE contrast enhancement, cropping of headers/footers
- **4-level question structure** support (Q → A/B → 1/2 → (i)/(ii)/(a)/(b))
- Automatic resolution of **Optional Groups** and **Attempt Any** questions
- Semantic evaluation using:
  - `all-MiniLM-L6-v2` (English)
  - `krutrim-ai-labs/Vyakyarth` (Hindi/Devanagari)
- Smart marks calculation with structural penalties (letter format, dialogue, speech, summary, etc.)
- Detailed AI-generated feedback using **Llama-3.2-3B-Instruct**
- Generates professional **DOCX evaluation reports**
- Azure Blob Storage integration for PDFs and images
- Clean separation of student handwritten text extraction

---

## 📋 Supported Papers

| Paper Code          | Language | Total Marks | Status |
|---------------------|----------|-------------|--------|
| `ENGLISH-2026-EN`   | English  | 80          | Fully Supported |
| `HINDI-2026-HI`     | Hindi    | 80          | Fully Supported |

---

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **OCR**: Google Lens + Tesseract (fallback)
- **PDF Processing**: PyMuPDF (fitz), pdf2image, Pillow
- **Computer Vision**: OpenCV
- **Embeddings**: Sentence-Transformers
- **LLM**: Llama-3.2-3B-Instruct (GGUF via llama-cpp-python)
- **Document Generation**: python-docx
- **Storage**: Azure Blob Storage
- **Async Support**: asyncio + ThreadPoolExecutor

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-answer-sheet-evaluator.git
cd ai-answer-sheet-evaluator
