# rag_module.py
"""
Syllabus RAG module:

- Loads vector store index.faiss + texts.json
- Provides query rewrite, course routing, and chunk selection
- Public exports:
    - refine_question_with_qwen(question: str) -> str
    - classify_intent(question: str) -> Dict
    - retrieve_context(question: str, ...) -> List[str]
    - is_course_selection_question(question: str) -> bool
    - is_syllabus_question(question: str) -> bool
    - is_course_comparison_question(question: str) -> bool
    - LAST_RETRIEVAL_DEBUG: dict
"""

import os
import re
import json
from typing import List, Dict, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.course_catalog import load_course_keyword_map
from config.paths import INDEX_PATH, TEXTS_PATH, E5_MODEL_NAME, E5_LOCAL_ONLY
from qwen_client import call_qwen

# ================== Global Configuration ==================

TOP_K = 10
FAISS_RAW_K = 24
LEXICAL_RAW_K = 24
MAX_CONTEXT_CHARS = 3600

COURSE_FILE_HINTS: Dict[str, List[str]] = load_course_keyword_map()

LAST_RETRIEVAL_DEBUG: Dict = {}

# ================== Model & Vector Store Loading ==================

_emb_model = None
_emb_error = None
_emb_warned = False


def _load_embedding_model():
    global _emb_model, _emb_error
    if _emb_model is not None:
        return _emb_model
    if _emb_error is not None:
        return None
    kwargs = {}
    if E5_LOCAL_ONLY:
        kwargs["local_files_only"] = True
    try:
        try:
            _emb_model = SentenceTransformer(E5_MODEL_NAME, **kwargs)
        except TypeError:
            _emb_model = SentenceTransformer(E5_MODEL_NAME)
        return _emb_model
    except Exception as exc:
        _emb_error = exc
        return None


def _note_embedding_unavailable():
    global _emb_warned
    if _emb_warned or _emb_error is None:
        return
    print(f"[RAG] Embedding model unavailable: {_emb_error}. Falling back to lexical retrieval.")
    _emb_warned = True


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_DOC_TOKENS: Optional[List[Set[str]]] = None


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _ensure_doc_tokens() -> List[Set[str]]:
    global _DOC_TOKENS
    if _DOC_TOKENS is not None:
        return _DOC_TOKENS
    _DOC_TOKENS = [set(_tokenize(doc.get("text", ""))) for doc in DOCS]
    return _DOC_TOKENS


def _lexical_rank_docs(question: str, max_hits: int) -> List[Tuple[float, int]]:
    tokens = set(_tokenize(question))
    if not tokens:
        return []
    doc_tokens = _ensure_doc_tokens()
    scored = []
    for i, dt in enumerate(doc_tokens):
        if not dt:
            continue
        overlap = len(tokens & dt)
        if overlap == 0:
            continue
        score = overlap / (len(tokens) + 1)
        scored.append((score, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:max_hits]


faiss_index = None
if os.path.exists(INDEX_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
else:
    print(f"[RAG] Syllabus Faiss index not found at {INDEX_PATH}; embedding retrieval disabled.")


DOCS: List[Dict] = []
if os.path.exists(TEXTS_PATH):
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        DOCS = json.load(f)
    print(f"[RAG] Loaded {len(DOCS)} chunks from {TEXTS_PATH}")
else:
    print(f"[RAG] Syllabus texts.json not found at {TEXTS_PATH}; retrieval disabled.")


# ================== Qwen: Retrieval Query Rewrite ==================


def refine_question_with_qwen(question: str) -> str:
    system_prompt = (
        "You are a query rewriting assistant for a RAG system over NYU course syllabi.\n"
        "Your task: rewrite the student's question into a SHORT English search query or keyword list.\n"
        "Rules:\n"
        "1) Output ONLY the rewritten query, in English, on a single line.\n"
        "2) Do NOT explain, do NOT add any extra text.\n"
        "3) Preserve any course codes like 'ECE-GY 6143' exactly if they appear.\n"
        "4) Use 5-20 words that best capture the intent (topics, grading, exam, workload, etc.)."
    )

    user_prompt = (
        "Student question:\n"
        f"{question}\n\n"
        "Rewrite this as a concise English search query for retrieving relevant syllabus snippets."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        refined = call_qwen(
            messages,
            max_tokens=64,
            temperature=0.1,
            top_p=0.8,
        )
        refined = (refined or "").strip()
        if not refined:
            return question
        return refined
    except Exception:
        return question


# ================== Embedding & Course Code Utilities ==================


def embed_query(query: str) -> Optional[np.ndarray]:
    model = _load_embedding_model()
    if model is None:
        return None
    text = f"query: {query}"
    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    return emb.astype("float32").reshape(1, -1)


def _extract_course_codes(question: str) -> List[str]:
    pattern = re.compile(r"\b[A-Z]{2,4}-?GY\s*\d{3,4}\b")
    codes = pattern.findall(question)
    return [c.strip() for c in codes]


def _normalize_course_code(code: str) -> str:
    c = code.upper().replace("_", " ").replace("-", " ")
    parts = c.split()
    if len(parts) == 3:
        return f"{parts[0]}-{parts[1]} {parts[2]}"
    return code.strip()


def _course_from_explicit_code(question: str) -> Tuple[Optional[str], str]:
    codes = _extract_course_codes(question)
    if len(codes) == 1:
        return _normalize_course_code(codes[0]), "explicit"
    return None, "explicit"


def _course_from_keywords(question: str) -> Tuple[Optional[str], str]:
    text = question.lower()
    best_course = None
    best_score = 0

    for course, hints in COURSE_FILE_HINTS.items():
        score = 0
        for h in hints:
            if h.lower() in text:
                score += 1
        if score > best_score:
            best_score = score
            best_course = course

    if best_score >= 1:
        return best_course, "keywords"
    return None, "keywords"


def _vote_course_by_candidates(candidate_idxs: List[int]) -> Tuple[Optional[str], str]:
    course_scores: Dict[str, float] = {}
    for rank, doc_idx in enumerate(candidate_idxs):
        doc = DOCS[int(doc_idx)]
        course = doc.get("meta", {}).get("course")
        if not course:
            continue
        course_scores[course] = course_scores.get(course, 0.0) + (1.0 / (rank + 1))

    if not course_scores:
        return None, "candidate_vote"

    best_course, best_score = max(course_scores.items(), key=lambda kv: kv[1])
    total = sum(course_scores.values())
    if total <= 0:
        return None, "candidate_vote"
    if best_score / total < 0.30:
        return None, "candidate_vote"
    return best_course, "candidate_vote"


def route_course(question: str, candidate_idxs: List[int]) -> Tuple[Optional[str], str]:
    course, src = _course_from_explicit_code(question)
    if course:
        return course, src

    course, src = _course_from_keywords(question)
    if course:
        return course, src

    course, src = _vote_course_by_candidates(candidate_idxs)
    if course:
        return course, src

    return None, "unknown"


# ================== Intent Detection ==================


def _contains_any(text: str, keys: List[str]) -> bool:
    return any(k in text for k in keys)


def classify_intent(question: str) -> Dict[str, object]:
    q = question.lower()
    scores = {
        "comparison": 0,
        "selection": 0,
        "syllabus": 0,
        "chat": 0,
    }

    if _contains_any(q, [" vs ", "versus", "compare", "comparison"]):
        scores["comparison"] += 4

    if " or " in q and _contains_any(q, ["course", "class", "take", "choose"]):
        scores["comparison"] += 2

    codes = _extract_course_codes(question)
    if len(codes) >= 2:
        scores["comparison"] += 4
    elif len(codes) == 1:
        scores["syllabus"] += 2

    selection_signals = [
        "which course should i take",
        "which course should i choose",
        "which class should i take",
        "recommend a course",
        "course recommendation",
        "best course for me",
        "what course should i take",
        "which course",
        "what course",
        "recommend me",
        "suggest a course",
    ]
    if _contains_any(q, [
        *selection_signals,
    ]):
        scores["selection"] += 4

    # Goal/interest-driven prompts should usually route to recommendation,
    # unless they are clearly asking for a concrete syllabus fact.
    goal_signals = [
        "i want to learn",
        "i'm interested in",
        "interested in",
        "my goal is",
        "targeting",
        "fit for me",
        "good for me",
        "based on my background",
        "for someone",
    ]
    if _contains_any(q, goal_signals):
        scores["selection"] += 3

    # Catch broader "which ... should I take" phrasing.
    if re.search(r"\b(which|what)\s+(course|class)\b", q) and re.search(
        r"\b(should|can)\s+i\s+(take|choose)\b", q
    ):
        scores["selection"] += 3

    if _contains_any(q, [
        "ece-gy", "cs-gy", "syllabus", "grading", "exam", "midterm", "final",
        "homework", "project", "lab", "course content", "assignment", "prerequisite",
    ]):
        scores["syllabus"] += 3

    if _contains_any(q, [
        "hello", "hi", "how are you", "thank you", "thanks", "career", "plan", "goal",
        "resume", "background",
    ]):
        scores["chat"] += 2

    if scores["selection"] > 0 and scores["comparison"] > 0:
        scores["comparison"] += 1

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_intent, best_score = ordered[0]
    second_score = ordered[1][1]
    spread = best_score - second_score

    if best_score <= 0:
        return {
            "intent": "chat",
            "confidence": 0.45,
            "scores": scores,
            "reason": "no_rule_hit",
        }

    confidence = min(0.95, 0.45 + best_score * 0.08 + spread * 0.06)
    return {
        "intent": best_intent,
        "confidence": round(confidence, 3),
        "scores": scores,
        "reason": f"best={best_intent};spread={spread}",
    }


def detect_priority_types(question: str) -> Set[str]:
    q = question.lower()
    types: Set[str] = set()

    if any(k in q for k in ["grading", "grade", "weight", "percentage", "%", "score"]):
        types.update({"grading_section", "grading_line", "project", "homework_lab"})
    if any(k in q for k in ["quiz", "exam", "midterm", "final"]):
        types.update({"exam", "grading_section", "grading_line", "project"})

    if any(k in q for k in ["homework", "assignment", "problem set", "ps ", "ps.", "lab", "labs"]):
        types.update({"homework_lab", "grading_section", "grading_line"})
    if any(k in q for k in ["project"]):
        types.update({"project", "grading_section", "grading_line"})

    if any(k in q for k in ["attendance", "participation"]):
        types.update({"online_format", "lecture_info"})
    if any(k in q for k in ["remote", "zoom", "online", "pre-recorded"]):
        types.update({"online_format", "lecture_info"})

    if any(k in q for k in ["class time", "schedule", "course schedule", "weekly topics"]):
        types.update({"schedule", "lecture_info"})
    if "final exam" in q or "midterm" in q:
        types.add("schedule")

    if any(k in q for k in ["professor", "instructor", "office hour", "office hours"]):
        types.add("instructor")
    if any(k in q for k in ["ta", "grader"]):
        types.add("grader")

    if any(k in q for k in ["prereq", "pre-requisite", "prerequisite"]):
        types.add("prerequisites")

    if any(k in q for k in ["course content", "what does it cover", "cover", "introduce this course"]):
        types.update({"course_description", "materials", "schedule"})

    if any(k in q for k in ["textbook", "github", "slides", "lecture notes", "class material", "materials"]):
        types.update({"materials", "course_description"})

    return types


def is_course_comparison_question(question: str) -> bool:
    return classify_intent(question)["intent"] == "comparison"


def is_course_selection_question(question: str) -> bool:
    return classify_intent(question)["intent"] == "selection"


def is_syllabus_question(question: str) -> bool:
    intent = classify_intent(question)["intent"]
    return intent in {"syllabus", "comparison", "selection"}


def _looks_like_comparison(question: str, course_codes: List[str]) -> bool:
    q = question.lower()
    if " or " in q or "vs" in q or "versus" in q:
        return True
    if "compare" in q or "comparison" in q:
        return True
    return len(course_codes) >= 2


# ================== Chunk Scoring & Retrieval ==================


def _compute_boost(
    question: str,
    text: str,
    meta: Dict,
    base_rank: int,
    priority_types: Set[str],
    course_codes: List[str],
) -> int:
    q = question.lower()
    t = text.lower()
    meta_type = meta.get("type", "normal")

    boost = 0

    if any(k in q for k in ["quiz", "exam", "midterm", "final"]):
        if any(k in t for k in ["exam", "midterm", "final", "quiz", "test", "exams"]):
            boost += 3

    if any(k in q for k in ["grading", "grade", "weight", "percentage", "%", "score"]):
        if any(k in t for k in ["grading", "grade", "weight", "percentage", "%", "assessment", "evaluation"]):
            boost += 3

    if any(k in q for k in ["homework", "assignment", "project"]):
        if any(k in t for k in ["homework", "assignment", "project", "lab", "problem set"]):
            boost += 2

    if any(k in q for k in ["attendance", "participation"]):
        if any(k in t for k in ["attendance", "participation"]):
            boost += 2

    if any(k in q for k in ["class time", "schedule", "course schedule"]):
        if any(k in t for k in ["lecture", "schedule", "tuesday", "thursday", "monday", "room", "class time"]):
            boost += 2

    if priority_types:
        if meta_type in priority_types:
            boost += 6
        elif meta_type != "normal":
            boost -= 1

    for code in course_codes:
        if code.lower() in t:
            boost += 8

    # Keep a slight preference for higher fused ranks.
    boost += max(0, 3 - min(base_rank, 3))
    return boost


def lexical_grading_candidates(question: str, max_hits: int = 5) -> List[str]:
    q = question.lower()
    need_grading = any(k in q for k in ["grading", "grade", "percentage", "%", "exam", "midterm", "final", "score"])
    if not need_grading:
        return []

    course_codes = _extract_course_codes(question)
    hits = []

    for doc in DOCS:
        meta = doc.get("meta", {})
        t = doc.get("text", "")
        if meta.get("type") not in {"grading_section", "grading_line", "exam", "schedule"}:
            continue
        if "%" not in t and not any(k in t.lower() for k in ["midterm", "final", "exam", "quiz"]):
            continue

        low = t.lower()
        if course_codes:
            if not any(code.lower() in low for code in course_codes):
                fname = meta.get("file", "").lower()
                if not any(code.lower().replace(" ", "")[:7] in fname for code in course_codes):
                    continue

        hits.append((meta.get("file", "unknown"), meta.get("page", 0), t, meta.get("type", "fallback")))

    hits.sort(key=lambda x: (x[0], x[1]))
    results = []
    for fname, page, text, mtype in hits[:max_hits]:
        header = f"[{fname} | page {page + 1} | type={mtype}_fallback]"
        results.append(f"{header}\n{text}")
    return results


def _rrf_fuse(
    dense_idxs: List[int],
    lexical_idxs: List[int],
    k_rrf: int = 60,
) -> List[int]:
    fused: Dict[int, float] = {}

    for r, i in enumerate(dense_idxs):
        fused[i] = fused.get(i, 0.0) + (1.0 / (k_rrf + r + 1))

    for r, i in enumerate(lexical_idxs):
        fused[i] = fused.get(i, 0.0) + (1.0 / (k_rrf + r + 1))

    return [i for i, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]


def _dynamic_context_limits(question: str, default_top_k: int) -> Tuple[int, int]:
    q = question.lower()
    if any(k in q for k in ["grading", "grade", "%", "exam", "midterm", "final", "schedule", "class time"]):
        return max(default_top_k, 12), 4800
    return default_top_k, MAX_CONTEXT_CHARS


def retrieve_context(
    question: str,
    top_k: int = TOP_K,
    faiss_raw_k: int = FAISS_RAW_K,
    forced_course: Optional[str] = None,
    analysis_question: Optional[str] = None,
) -> List[str]:
    global LAST_RETRIEVAL_DEBUG
    aq = analysis_question or question
    course_codes = _extract_course_codes(aq)
    is_comparison = _looks_like_comparison(aq, course_codes)

    if not DOCS:
        LAST_RETRIEVAL_DEBUG.clear()
        LAST_RETRIEVAL_DEBUG.update({
            "mode": "none",
            "question": aq,
            "selected_course": None,
            "route_source": "none",
            "course_codes_in_question": course_codes,
            "is_comparison": is_comparison,
            "reason": "no_docs_loaded",
        })
        return []

    top_k, max_chars = _dynamic_context_limits(aq, top_k)

    dense_scores = np.zeros((1, 0), dtype="float32")
    dense_idx = np.zeros((1, 0), dtype="int64")
    mode = "hybrid"

    if faiss_index is not None:
        q_emb = embed_query(question)
        if q_emb is not None:
            dense_scores, dense_idx = faiss_index.search(q_emb, faiss_raw_k)
        else:
            _note_embedding_unavailable()

    lexical_ranked = _lexical_rank_docs(question, LEXICAL_RAW_K)
    lexical_idxs = [i for _, i in lexical_ranked]
    dense_idxs = [int(i) for i in dense_idx[0]] if dense_idx.size > 0 else []

    fused_idxs = _rrf_fuse(dense_idxs, lexical_idxs)
    if not fused_idxs:
        fused_idxs = lexical_idxs
        mode = "lexical"

    selected_course, route_source = route_course(aq, fused_idxs)

    if forced_course is not None:
        selected_course = _normalize_course_code(forced_course)
        route_source = "forced"

    priority_types = detect_priority_types(aq)

    candidates = []
    for rank, i in enumerate(fused_idxs[: max(faiss_raw_k, LEXICAL_RAW_K)]):
        doc = DOCS[int(i)]
        text = doc["text"]
        meta = doc.get("meta", {})
        fname = meta.get("file", "unknown")
        page = meta.get("page", 0) + 1

        header = f"[{fname} | page {page} | type={meta.get('type', 'normal')}]"
        full_text = f"{header}\n{text}"

        boost = _compute_boost(
            question=aq,
            text=text,
            meta=meta,
            base_rank=rank,
            priority_types=priority_types,
            course_codes=course_codes,
        )

        candidates.append({
            "rank": rank,
            "boost": boost,
            "text": full_text,
            "meta": meta,
        })

    candidates.sort(key=lambda x: (-x["boost"], x["rank"]))

    dense_top10 = []
    for r, (d, doc_idx) in enumerate(zip(dense_scores[0][:10], dense_idx[0][:10])):
        dense_top10.append({
            "rank": int(r),
            "doc_idx": int(doc_idx),
            "dist": float(d),
            "file": DOCS[int(doc_idx)].get("meta", {}).get("file"),
            "course": DOCS[int(doc_idx)].get("meta", {}).get("course"),
            "type": DOCS[int(doc_idx)].get("meta", {}).get("type"),
        })

    LAST_RETRIEVAL_DEBUG.clear()
    LAST_RETRIEVAL_DEBUG.update({
        "mode": mode,
        "question": aq,
        "selected_course": selected_course,
        "route_source": route_source,
        "course_codes_in_question": course_codes,
        "is_comparison": is_comparison,
        "dense_top10": dense_top10,
        "lexical_top10": [
            {
                "rank": i,
                "score": float(s),
                "doc_idx": int(idx),
                "file": DOCS[int(idx)].get("meta", {}).get("file"),
                "course": DOCS[int(idx)].get("meta", {}).get("course"),
                "type": DOCS[int(idx)].get("meta", {}).get("type"),
            }
            for i, (s, idx) in enumerate(lexical_ranked[:10])
        ],
        "fused_top20": [int(i) for i in fused_idxs[:20]],
        "candidates_top20": [
            {
                "rank": c["rank"],
                "boost": c["boost"],
                "file": c["meta"].get("file"),
                "course": c["meta"].get("course"),
                "type": c["meta"].get("type"),
            }
            for c in candidates[:20]
        ],
    })

    dominant_course: Optional[str] = None
    dominant_file: Optional[str] = None

    if not is_comparison and candidates:
        if selected_course is not None:
            dominant_course = selected_course
        else:
            file_counts: Dict[str, int] = {}
            for c in candidates[: min(len(candidates), faiss_raw_k)]:
                fname = c["meta"].get("file", "")
                file_counts[fname] = file_counts.get(fname, 0) + 1
            if file_counts:
                dominant_file = max(file_counts.items(), key=lambda kv: kv[1])[0]

    pieces: List[str] = []
    total_len = 0

    for c in candidates:
        if len(pieces) >= top_k:
            break

        meta = c["meta"]

        if dominant_course is not None:
            if _normalize_course_code(meta.get("course", "")) != dominant_course:
                continue
        elif dominant_file is not None:
            if meta.get("file") != dominant_file:
                continue

        p = c["text"]
        if total_len + len(p) > max_chars:
            continue

        pieces.append(p)
        total_len += len(p)

    fallback = lexical_grading_candidates(aq, max_hits=3)
    for fb in fallback:
        if len(pieces) >= top_k:
            break
        if fb in pieces:
            continue
        if total_len + len(fb) > max_chars:
            continue
        pieces.append(fb)
        total_len += len(fb)

    return pieces


__all__ = [
    "refine_question_with_qwen",
    "classify_intent",
    "retrieve_context",
    "is_course_selection_question",
    "is_syllabus_question",
    "is_course_comparison_question",
    "lexical_grading_candidates",
    "LAST_RETRIEVAL_DEBUG",
]
