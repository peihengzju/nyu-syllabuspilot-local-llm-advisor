# ingest_syllabi_optimized.py
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from config.paths import (
    INDEX_DIR,
    INDEX_PATH,
    TEXTS_PATH,
    E5_MODEL_NAME,
    INGEST_MANIFEST_PATH,
    INGEST_CACHE_DIR,
)

SYLLABUS_DIR = "docs/syllabus"

CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_CHARS = 200



# ========== Course Code Inference ==========

COURSE_CODE_PATTERN = re.compile(r"(ECE[_\-\s]?GY[_\-\s]?(\d{4}))", re.IGNORECASE)


def infer_course_from_filename(fname: str) -> str | None:
    """
    Infer course code from filename and normalize to 'ECE-GY 6143'.
    Examples:
      ECE_GY_6143 syllabus.pdf
      ece-gy6143_Syllabus_Fall2025.pdf
    """
    m = COURSE_CODE_PATTERN.search(fname)
    if not m:
        return None

    raw = m.group(1).upper()  # ECE_GY_6143 / ECE-GY6143 / ECE GY 6143
    raw = raw.replace("_", " ").replace("-", " ")
    parts = raw.split()  # e.g. ['ECE', 'GY', '6143'] / ['ECEGY6143']

    if len(parts) == 3 and parts[0] == "ECE" and parts[1] == "GY":
        # ECE GY 6143 -> ECE-GY 6143
        return f"{parts[0]}-{parts[1]} {parts[2]}"

    if len(parts) == 2 and parts[0].startswith("ECE") and "GY" in parts[0]:
        # Handle odd format such as ECEGY 6143.
        return f"ECE-GY {parts[1]}"

    # Fallback: ensure ECE-GY prefix is preserved.
    if " " in raw:
        head, tail = raw.rsplit(" ", 1)
        return f"{head.replace(' ', '-') } {tail}"

    return raw


# ========== Base Utilities ==========

def load_all_pdfs():
    texts: List[str] = []
    meta: List[Dict] = []

    for fname in os.listdir(SYLLABUS_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(SYLLABUS_DIR, fname)
        reader = PdfReader(path)

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if not page_text:
                continue
            texts.append(page_text)
            meta.append({"file": fname, "page": i})

    return texts, meta


def chunk_by_lines(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS
) -> List[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    coarse_chunks: List[str] = []
    cur = ""

    for ln in lines:
        candidate = (cur + "\n" + ln) if cur else ln
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                coarse_chunks.append(cur)
            cur = ln

    if cur:
        coarse_chunks.append(cur)

    final_chunks: List[str] = []
    for ch in coarse_chunks:
        if len(ch) <= max_chars:
            final_chunks.append(ch)
        else:
            start = 0
            while start < len(ch):
                part = ch[start:start + max_chars]
                if part:
                    final_chunks.append(part)
                start += max_chars - overlap_chars

    return final_chunks


def _join_window(lines: List[str], start: int, end: int) -> str:
    return "\n".join(ln for ln in lines[start:end] if ln.strip())


# ========== Grading Extraction ==========

def _is_grading_header(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return False

    if low.startswith("grading"):
        return True
    if low.startswith("grade distribution"):
        return True
    if low.startswith("evaluation"):
        return True
    if low.startswith("assessment"):
        return True
    if "grading" in low and ("evaluation" in low or "assessment" in low):
        return True

    return False


def extract_grading_sections(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    i = 0
    while i < n:
        line = lines[i]
        if _is_grading_header(line):
            start = i
            j = i + 1
            while j < n:
                next_line = lines[j]
                if _is_grading_header(next_line):
                    break
                stripped = next_line.strip()
                if stripped.isupper() and len(stripped.split()) <= 6:
                    break
                j += 1

            window = _join_window(lines, start, j)
            if len(window) >= 30 and window not in chunks:
                chunks.append(window)
            i = j
        else:
            i += 1

    return chunks


def extract_grading_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks: List[str] = []

    for ln in lines:
        low = ln.lower()
        if "%" in ln or "percent" in low or "grade" in low:
            if len(ln) < 10:
                continue
            if ln not in chunks:
                chunks.append(ln)

    return chunks


# ========== Structured Extraction: Description / Instructor / TA / Class Info ==========

def extract_course_description(text: str) -> List[str]:
    """
    Course name + code + description section.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "description:" in low or low.startswith("description "):
            start = max(0, i - 1)
            j = i + 1
            while j < n:
                nxt = lines[j].strip()
                if not nxt:
                    break
                if nxt.isupper() and len(nxt.split()) <= 6:
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 30 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_instructor_info(text: str) -> List[str]:
    """
    Professor / instructor / office-hours related info.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "professor" in low or "instructor" in low:
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["grader", "lecture", "class material", "prereq", "pre-requisites"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_graders_info(text: str) -> List[str]:
    """
    Grader list, emails, and related lines.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        if "grader" in ln.lower():
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["lecture", "office hours", "class material", "tentative schedule"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_lecture_info(text: str) -> List[str]:
    """
    Lecture details: time, location, zoom mode, attendance, laptop requirements.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        if "lecture:" in ln.lower():
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["grading", "grader", "class material", "tentative schedule"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_schedule(text: str) -> List[str]:
    """
    Tentative schedule / per-week topics。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "tentative schedule" in low or "schedule of classes" in low:
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["grading", "prereq", "pre-requisites", "class material"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 30 and window not in chunks:
                chunks.append(window)

    if not chunks:
        date_pattern = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
        tmp_lines = []
        for ln in lines:
            if date_pattern.search(ln):
                tmp_lines.append(ln)
        if tmp_lines:
            window = "\n".join(tmp_lines)
            chunks.append(window)

    return chunks


def extract_exam_info(text: str) -> List[str]:
    """
    Midterm / final exam timing and review info.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in ["midterm", "final exam", "final  exam", "exam review"]):
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            window = _join_window(lines, start, end)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_prerequisites(text: str) -> List[str]:
    """
    Pre-requisites / prerequisites section.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("pre-requisites") or low.startswith("prerequisites"):
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["class material", "online format", "schedule", "grading"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 30 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_materials(text: str) -> List[str]:
    """
    Class materials / textbook / GitHub / links.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "class material" in low or "textbook" in low:
            start = i
            j = i + 1
            while j < n:
                nxt = lines[j]
                low2 = nxt.lower()
                if any(k in low2 for k in ["online format", "tentative schedule", "grading"]):
                    break
                j += 1
            window = _join_window(lines, start, j)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)

    urls = []
    for ln in lines:
        if "http://" in ln or "https://" in ln or "github.com" in ln.lower():
            urls.append(ln.strip())
    if urls:
        window = "\n".join(urls)
        if window not in chunks:
            chunks.append(window)

    return chunks


def extract_project_info(text: str) -> List[str]:
    """
    Optional project and project-weight related info.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "project" in low:
            start = max(0, i - 1)
            end = min(len(lines), i + 3)
            window = _join_window(lines, start, end)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_homework_lab(text: str) -> List[str]:
    """
    Homework / lab related info.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "homework" in low or "lab" in low or "labs" in low:
            start = max(0, i - 1)
            end = min(len(lines), i + 3)
            window = _join_window(lines, start, end)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


def extract_online_format(text: str) -> List[str]:
    """
    Online format / zoom / pre-recorded / attendance related format info.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[str] = []

    for i, ln in enumerate(lines):
        low = ln.lower()
        if "online format" in low or "zoom" in low or "pre-recorded" in low or "attendance" in low:
            start = max(0, i - 1)
            end = min(len(lines), i + 4)
            window = _join_window(lines, start, end)
            if len(window) >= 20 and window not in chunks:
                chunks.append(window)
    return chunks


# ========== Main Flow ==========

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_cache_name(fname: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", fname)


def _cache_paths(fname: str) -> Dict[str, str]:
    base = _safe_cache_name(fname)
    return {
        "docs": os.path.join(INGEST_CACHE_DIR, f"{base}.docs.json"),
        "embs": os.path.join(INGEST_CACHE_DIR, f"{base}.emb.npy"),
    }


def _load_manifest() -> Dict:
    if not os.path.exists(INGEST_MANIFEST_PATH):
        return {"files": {}, "updated_at": None}
    with open(INGEST_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: Dict) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INGEST_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _append_chunks(
    docs: List[Dict],
    seen_texts: set,
    page_text: str,
    base_meta: Dict,
    extractor,
    chunk_type: str,
) -> None:
    for chunk in extractor(page_text):
        t = chunk.strip()
        if t and t not in seen_texts:
            seen_texts.add(t)
            docs.append({"text": t, "meta": {**base_meta, "type": chunk_type}})


def _extract_docs_for_pdf(path: str, fname: str) -> List[Dict]:
    docs: List[Dict] = []
    seen_texts = set()
    reader = PdfReader(path)
    course = infer_course_from_filename(fname)

    for i, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        base_meta = {"file": fname, "page": i, "course": course}
        _append_chunks(docs, seen_texts, page_text, base_meta, chunk_by_lines, "normal")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_grading_sections, "grading_section")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_grading_lines, "grading_line")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_course_description, "course_description")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_instructor_info, "instructor")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_graders_info, "grader")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_lecture_info, "lecture_info")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_schedule, "schedule")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_exam_info, "exam")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_prerequisites, "prerequisites")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_materials, "materials")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_project_info, "project")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_homework_lab, "homework_lab")
        _append_chunks(docs, seen_texts, page_text, base_meta, extract_online_format, "online_format")

    return docs


def main():
    print("Loading syllabus PDFs (incremental mode)...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(INGEST_CACHE_DIR, exist_ok=True)

    manifest = _load_manifest()
    prev_files = manifest.get("files", {})
    new_files: Dict[str, Dict] = {}

    pdf_files = sorted([f for f in os.listdir(SYLLABUS_DIR) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print("No PDFs found. Skipping.")
        return

    all_docs: List[Dict] = []
    all_embs: List[np.ndarray] = []
    changed_files = 0
    reused_files = 0
    total_chunks = 0

    model = SentenceTransformer(E5_MODEL_NAME)

    for fname in pdf_files:
        path = os.path.join(SYLLABUS_DIR, fname)
        sha = _sha256_file(path)
        cache_paths = _cache_paths(fname)
        old = prev_files.get(fname)

        use_cache = (
            old is not None
            and old.get("sha256") == sha
            and os.path.exists(cache_paths["docs"])
            and os.path.exists(cache_paths["embs"])
        )

        if use_cache:
            with open(cache_paths["docs"], "r", encoding="utf-8") as f:
                docs = json.load(f)
            embs = np.load(cache_paths["embs"])
            reused_files += 1
        else:
            docs = _extract_docs_for_pdf(path, fname)
            texts = [d["text"] for d in docs]
            if texts:
                embs = model.encode(
                    [f"passage: {t}" for t in texts],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype("float32")
            else:
                embs = np.zeros((0, 1), dtype="float32")
            with open(cache_paths["docs"], "w", encoding="utf-8") as f:
                json.dump(docs, f, ensure_ascii=False, indent=2)
            np.save(cache_paths["embs"], embs)
            changed_files += 1

        total_chunks += len(docs)
        all_docs.extend(docs)
        if embs.size > 0 and len(embs.shape) == 2 and embs.shape[0] > 0:
            all_embs.append(embs)

        new_files[fname] = {
            "sha256": sha,
            "chunks": len(docs),
            "cache_docs": os.path.relpath(cache_paths["docs"], INDEX_DIR),
            "cache_embs": os.path.relpath(cache_paths["embs"], INDEX_DIR),
            "updated_at": datetime.utcnow().isoformat(),
        }

    if all_embs:
        embeddings = np.vstack(all_embs).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(1)
        faiss.write_index(index, INDEX_PATH)

    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    report = {
        "updated_at": datetime.utcnow().isoformat(),
        "total_files": len(pdf_files),
        "changed_files": changed_files,
        "reused_files": reused_files,
        "total_chunks": total_chunks,
        "by_type": {},
    }
    for d in all_docs:
        t = d.get("meta", {}).get("type", "unknown")
        report["by_type"][t] = report["by_type"].get(t, 0) + 1

    manifest["files"] = new_files
    manifest["updated_at"] = report["updated_at"]
    manifest["last_report"] = report
    _save_manifest(manifest)

    print("Done:")
    print(f"  - Vector index -> {INDEX_PATH}")
    print(f"  - Text data -> {TEXTS_PATH}")
    print(f"  - Incremental manifest -> {INGEST_MANIFEST_PATH}")
    print("Summary:")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
