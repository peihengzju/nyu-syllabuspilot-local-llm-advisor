# memory_module.py
"""
Conversation memory module:
- Stores memories in memory_store/memories.json
- Uses Faiss for vector retrieval (mem_index.faiss)
- Exposes: reset_memories / add_memory_from_turn / retrieve_memories / format_memories_block
"""

import os
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Set

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.paths import (
    MEMORY_DIR,
    MEMORY_INDEX_PATH,
    MEMORY_TEXTS_PATH,
    MEMORY_REINDEX_EVERY,
    E5_MODEL_NAME,
    E5_LOCAL_ONLY,
)
from qwen_client import call_qwen

# ========== Base Configuration ==========

MemorySlot = Literal["profile", "preference", "fact", "recent"]

# Memory slot types.
MEMORY_SLOTS = {
    "profile",        # Personal profile: name, school, major, background
    "preference",     # Long-term preferences
    "fact",           # Important facts
    "recent",         # Short-lived recent context
}

# Slot-specific expiry / weighting policy.
MEMORY_EXPIRY_DAYS: Dict[MemorySlot, Optional[int]] = {
    "profile":   None,   # never expires
    "preference": 365,   # soft expiry: one year
    "fact":      180,    # half year
    "recent":     7,     # one week
}


@dataclass
class MemoryItem:
    id: int
    slot: MemorySlot
    text: str                  # Possibly summarized text
    importance: int            # 0=low, 1=medium, 2=high, 3=aggregated high
    created_at: str            # ISO timestamp
    last_used_at: str          # Last retrieval timestamp
    extra: Dict[str, Any]      # Extra metadata, e.g. {"source_turn": 12, "kind": "career_direction"}


# ========== Initialization ==========

os.makedirs(MEMORY_DIR, exist_ok=True)

if not os.path.exists(MEMORY_TEXTS_PATH):
    with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# Embedding model (memory also uses E5), lazily loaded.
_emb_model = None
_emb_error = None
_emb_warned = False

mem_index = None
_mem_index_loaded = False

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
    print(f"[Memory] Embedding model unavailable: {_emb_error}. Falling back to lexical retrieval.")
    _emb_warned = True

def _save_mem_index() -> None:
    if mem_index is None:
        return
    faiss.write_index(mem_index, MEMORY_INDEX_PATH)

def _ensure_mem_index(items: List[MemoryItem]):
    global mem_index, _mem_index_loaded
    model = _load_embedding_model()
    if model is None:
        _note_embedding_unavailable()
        return None
    if not _mem_index_loaded:
        _mem_index_loaded = True
        if os.path.exists(MEMORY_INDEX_PATH):
            mem_index = faiss.read_index(MEMORY_INDEX_PATH)
        else:
            mem_index = None
    if mem_index is None:
        _rebuild_mem_index(items)
    return mem_index

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())

def _lexical_score(q_tokens: Set[str], text: str) -> float:
    if not q_tokens:
        return 0.0
    t_tokens = set(_tokenize(text))
    if not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / (len(q_tokens) + 1)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


# ========== Common Utilities ==========

def _build_dialogue_snippet(question: str, answer: str) -> str:
    return f"User: {question}\nAssistant: {answer}"


def summarize_dialogue_with_qwen(snippet: str, max_words: int = 80) -> str:
    """
    Compress a longer dialogue into a short memory sentence.
    """
    system_prompt = (
        "You are a memory compression assistant.\n"
        "Task: Summarize the given user–assistant dialogue into a short factual memory.\n"
        "Rules:\n"
        f"1) Use at most {max_words} English words.\n"
        "2) Capture stable facts (user profile, preferences, goals, important decisions).\n"
        "3) Do not include ephemeral details or step-by-step reasoning.\n"
        "4) Output only the summary, no explanation."
    )

    user_prompt = (
        "Dialogue:\n"
        f"{snippet}\n\n"
        "Summarize this into a single short memory sentence."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        summary = call_qwen(messages, max_tokens=128, temperature=0.1, top_p=0.8)
        summary = (summary or "").strip()
        return summary if summary else snippet[:200]
    except Exception:
        # Fallback: truncated raw snippet.
        return snippet[:200]


def _load_memories() -> List[MemoryItem]:
    # Missing file -> no memories.
    if not os.path.exists(MEMORY_TEXTS_PATH):
        return []

    try:
        with open(MEMORY_TEXTS_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Empty file content -> treat as [].
        if not content:
            return []

        arr = json.loads(content)
    except Exception:
        # Corrupted or invalid JSON -> reset to [].
        arr = []
        with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    return [MemoryItem(**m) for m in arr]


def _save_memories(items: List[MemoryItem]) -> None:
    with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in items], f, ensure_ascii=False, indent=2)


def _rebuild_mem_index(items: Optional[List[MemoryItem]] = None) -> None:
    """
    Rebuild mem_index from all current MemoryItems.
    Aggregated memories can update existing entries, so incremental add is not enough.
    """
    global mem_index, _mem_index_loaded

    if items is None:
        items = _load_memories()

    model = _load_embedding_model()
    if model is None:
        _note_embedding_unavailable()
        mem_index = None
        _mem_index_loaded = True
        if os.path.exists(MEMORY_INDEX_PATH):
            try:
                os.remove(MEMORY_INDEX_PATH)
            except OSError:
                pass
        return

    _mem_index_loaded = True

    # No memory yet: create an empty index.
    if not items:
        dummy = model.encode(
            ["query: dummy"], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        d = len(dummy)
        mem_index = faiss.IndexFlatIP(d)
        _save_mem_index()
        return

    # Regular rebuild.
    dummy = model.encode(
        ["query: dummy"], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    d = len(dummy)
    mem_index = faiss.IndexFlatIP(d)

    texts = [f"passage: {m.text}" for m in items]
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    mem_index.add(embs)
    _save_mem_index()


def reset_memories() -> None:
    """
    Clear all memories:
    - Reset memories.json to []
    - Reset mem_index
    """
    global mem_index, _mem_index_loaded

    # 1) Reset JSON file to a valid empty array.
    with open(MEMORY_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    # 2) Reset Faiss index (or remove index file if embeddings are unavailable).
    model = _load_embedding_model()
    if model is None:
        _note_embedding_unavailable()
        mem_index = None
        _mem_index_loaded = True
        if os.path.exists(MEMORY_INDEX_PATH):
            try:
                os.remove(MEMORY_INDEX_PATH)
            except OSError:
                pass
        return

    dummy = model.encode(
        ["query: dummy"], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    d = len(dummy)
    mem_index = faiss.IndexFlatIP(d)
    _save_mem_index()


# ========== Slot Classification & Aggregators ==========

class MemoryAggregator:
    """
    Base class for aggregated memories:
    - kind: stored in extra["kind"] (e.g. 'career_direction', 'profile_aggregate')
    - slot: target memory slot (profile / preference / fact / recent)
    - base_importance: default importance for aggregated entries
    """
    kind: str
    slot: MemorySlot
    base_importance: int

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entity list for this aggregation type from one dialogue turn.
        Return [] when no entity is found. Must be overridden by subclasses.
        """
        raise NotImplementedError

    def render_text(self, entities: List[str]) -> str:
        """
        Render entities as a human-readable memory text.
        Can be overridden by subclasses.
        """
        return f"{self.kind}: " + ", ".join(entities)

    def upsert(
        self,
        items: List[MemoryItem],
        question: str,
        answer: str,
        source_turn: int,
    ) -> List[MemoryItem]:
        """
        Upsert one aggregated memory in MemoryItem list:
        - Find existing item with extra['kind'] == self.kind
        - Merge entities
        - Update text / last_used_at / importance
        - Create a new item if not found
        """
        raw = (question or "") + "\n" + (answer or "")
        entities = self.extract_entities(raw)
        if not entities:
            return items

        now = datetime.utcnow().isoformat()
        existing: Optional[MemoryItem] = None
        for m in items:
            if m.extra.get("kind") == self.kind:
                existing = m
                break

        if existing is not None:
            old_entities = set(existing.extra.get("entities", []))
            merged = sorted(old_entities | set(entities))
            existing.extra["entities"] = merged
            existing.text = self.render_text(merged)
            existing.last_used_at = now
            existing.importance = self.base_importance
        else:
            new_id = (max((m.id for m in items), default=0) + 1) if items else 1
            merged = sorted(set(entities))
            mem = MemoryItem(
                id=new_id,
                slot=self.slot,
                text=self.render_text(merged),
                importance=self.base_importance,
                created_at=now,
                last_used_at=now,
                extra={
                    "kind": self.kind,
                    "entities": merged,
                    "source_turn": source_turn,
                },
            )
            items.append(mem)

        return items


def classify_memory_slot(question: str, answer: str) -> tuple[MemorySlot, int]:
    """
    Heuristic slot classifier:
    - School/major/background -> profile
    - Intent/interests/preferences -> preference
    - Explicit factual decisions/plans -> fact
    - Otherwise -> recent
    importance: 2=high, 1=medium, 0=temporary
    """
    q = (question or "").lower()
    # Slot classification should prioritize user statement itself.
    # Assistant text can be noisy and should not dominate slot assignment.
    text = q

    # profile
    if any(k in text for k in ["nyu", "tandon", "ece", "major", "degree", "master", "phd", "graduate"]):
        return "profile", 2

    # preference
    if any(k in text for k in ["i want to learn", "i want to do", "i prefer", "my interest is", "i'm interested in"]):
        return "preference", 2

    # important fact
    if any(k in text for k in ["i decided", "i will", "i plan to", "i have enrolled", "i already switched"]):
        return "fact", 2

    # default recent
    return "recent", 0


class CareerDirectionAggregator(MemoryAggregator):
    kind = "career_direction"
    slot: MemorySlot = "preference"
    base_importance = 3

    def extract_entities(self, text: str) -> List[str]:
        t = text.lower()
        dirs = set()

        # AI infra / low-level systems
        if any(k in t for k in ["ai infra", "ai infrastructure", "system-level", "low-level"]):
            dirs.add("AI infrastructure / low-level systems")

        # Embedded systems
        if any(k in t for k in ["embedded"]):
            dirs.add("embedded systems")

        # Distributed / storage / GPU scheduling
        if any(k in t for k in ["distributed system", "distributed systems"]):
            dirs.add("distributed systems")
        if any(k in t for k in ["storage"]):
            dirs.add("storage systems")
        if any(k in t for k in ["gpu scheduling", "gpu scheduler"]):
            dirs.add("GPU scheduling")

        # Chip / VLSI / ASIC / IC design
        if any(k in t for k in ["chip", "ic design"]):
            dirs.add("chip / IC design")
        if "vlsi" in t:
            dirs.add("VLSI design")
        if "asic" in t:
            dirs.add("ASIC design")

        return sorted(dirs)

    def render_text(self, entities: List[str]) -> str:
        return "User's long-term career directions: " + ", ".join(entities)


class ProfileAggregator(MemoryAggregator):
    kind = "profile_aggregate"
    slot: MemorySlot = "profile"
    base_importance = 3

    def extract_entities(self, text: str) -> List[str]:
        t = text.lower()
        ents = set()

        if "nyu" in t and "tandon" in t:
            ents.add("NYU Tandon")
        if "ece" in t or "electrical and computer engineering" in t:
            ents.add("ECE master's student")
        if "brooklyn" in t:
            ents.add("based in Brooklyn")
        if "master" in t or "graduate" in t:
            ents.add("graduate student")

        return sorted(ents)

    def render_text(self, entities: List[str]) -> str:
        return "User's profile: " + ", ".join(entities)


AGGREGATORS: List[MemoryAggregator] = [
    CareerDirectionAggregator(),
    ProfileAggregator(),
    # Add more aggregators here, e.g. SkillsAggregator, PreferenceAggregator.
]


# ========== Memory Write & Retrieval ==========

def add_memory_from_turn(question: str, answer: str, source_turn: int) -> None:
    """
    Called after every turn:
    - Build a question/answer snippet and optionally summarize with Qwen
    - Add one raw memory item
    - Let all aggregators update/create aggregated memory items
    - Persist data and rebuild Faiss index on schedule
    """
    snippet = _build_dialogue_snippet(question, answer)
    # Summarize when snippet is too long.
    if len(snippet) > 700:
        text = summarize_dialogue_with_qwen(snippet, max_words=80)
    else:
        text = snippet

    items = _load_memories()
    now = datetime.utcnow().isoformat()
    # 1) Raw memory item with dedup/merge.
    slot, importance = classify_memory_slot(question, answer)
    q_tokens = set(_tokenize(text))
    duplicate: Optional[MemoryItem] = None
    best_score = 0.0

    for m in items:
        # Dedup only on raw memory entries.
        if m.extra.get("kind") is not None:
            continue
        m_tokens = set(_tokenize(m.text))
        score = _jaccard(q_tokens, m_tokens)
        if score > best_score:
            best_score = score
            duplicate = m

    if duplicate is not None and best_score >= 0.82:
        duplicate.last_used_at = now
        duplicate.importance = max(duplicate.importance, importance)
        duplicate.slot = duplicate.slot if duplicate.slot == slot else slot
        duplicate.extra["source_turn"] = source_turn
    else:
        new_id = (max((m.id for m in items), default=0) + 1) if items else 1
        raw_mem = MemoryItem(
            id=new_id,
            slot=slot,
            text=text,
            importance=importance,
            created_at=now,
            last_used_at=now,
            extra={"source_turn": source_turn},
        )
        items.append(raw_mem)

    # 2) Run all aggregators to update aggregated entries.
    for agg in AGGREGATORS:
        items = agg.upsert(items, question, answer, source_turn)

    # 3) Persist and rebuild vector index on schedule.
    _save_memories(items)
    # Keep index aligned with memory file to reduce fallback drift.
    if MEMORY_REINDEX_EVERY <= 1:
        _rebuild_mem_index(items)
    elif source_turn % MEMORY_REINDEX_EVERY == 0:
        _rebuild_mem_index(items)


def _time_decay(slot: MemorySlot, created_at: str) -> float:
    """
    Compute a [0,1] time weight using slot type and creation time.
    """
    expiry_days = MEMORY_EXPIRY_DAYS.get(slot)
    if expiry_days is None:
        return 1.0  # never expires

    created = datetime.fromisoformat(created_at)
    now = datetime.utcnow()
    delta_days = (now - created).days

    if delta_days <= 0:
        return 1.0
    if delta_days >= expiry_days:
        return 0.0

    # Simple linear decay.
    return max(0.0, 1.0 - delta_days / expiry_days)


def _time_decay_from_item(item: MemoryItem) -> float:
    """
    Prefer last-used freshness; fall back to creation time.
    """
    ts = item.last_used_at or item.created_at
    try:
        return _time_decay(item.slot, ts)
    except Exception:
        return _time_decay(item.slot, item.created_at)


def _select_top_memories(
    items: List[MemoryItem],
    scored: List[tuple[float, MemoryItem]],
    top_k: int,
) -> List[MemoryItem]:
    scored.sort(key=lambda x: x[0], reverse=True)
    top_items = [m for _, m in scored[:top_k]]

    now = datetime.utcnow().isoformat()
    id_set = {m.id for m in top_items}
    for m in items:
        if m.id in id_set:
            m.last_used_at = now
    _save_memories(items)

    return top_items


def retrieve_memories(
    question: str,
    top_k: int = 5,
    alpha: float = 0.5,
    beta: float = 0.4,
) -> List[MemoryItem]:
    """
    Retrieve memories:
    overall_score = embedding_sim + alpha * time_decay + beta * importance
    embedding_sim = inner-product similarity ([-1,1])
    """
    items = _load_memories()
    if not items:
        return []

    index = _ensure_mem_index(items)
    model = _load_embedding_model()
    # Index missing/model unavailable/empty/mismatch -> lexical fallback.
    # This works with periodic rebuild in add_memory_from_turn.
    if index is None or model is None or index.ntotal == 0 or index.ntotal != len(items):
        q_tokens = set(_tokenize(question))
        scored: List[tuple[float, MemoryItem]] = []
        for m in items:
            sim = _lexical_score(q_tokens, m.text)
            t_decay = _time_decay_from_item(m)
            imp = m.importance

            overall = float(sim) + alpha * t_decay + beta * imp

            kind = m.extra.get("kind")
            if kind == "career_direction":
                overall += 0.35
            elif kind == "profile_aggregate":
                overall += 0.25

            scored.append((overall, m))

        return _select_top_memories(items, scored, top_k)

    # Query embedding.
    q_text = f"query: {question}"
    q_emb = model.encode(
        [q_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Vector retrieval.
    k = min(top_k * 4, index.ntotal)  # retrieve extra candidates first
    sims, idxs = index.search(q_emb, k)  # inner product: larger is more similar
    sims = sims[0]
    idxs = idxs[0]

    scored: List[tuple[float, MemoryItem]] = []

    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(items):
            continue
        m = items[idx]
        t_decay = _time_decay_from_item(m)
        imp = m.importance

        overall = float(sim) + alpha * t_decay + beta * imp

        # Extra bias for aggregated memories.
        kind = m.extra.get("kind")
        if kind == "career_direction":
            overall += 0.35
        elif kind == "profile_aggregate":
            overall += 0.25

        scored.append((overall, m))

    return _select_top_memories(items, scored, top_k)


def format_memories_block(mem_items: List[MemoryItem]) -> str:
    """
    Format memory items into a multi-line string for Qwen prompts.
    """
    if not mem_items:
        return "(No retrieved memories.)"
    lines = []
    for m in mem_items:
        lines.append(f"[{m.slot} | importance={m.importance}] {m.text}")
    return "\n".join(lines)


def export_all_memories() -> List[Dict[str, Any]]:
    """
    Return raw dict list for all current memory items (for logging/export).
    """
    return [asdict(m) for m in _load_memories()]


__all__ = [
    "MemoryItem",
    "MemorySlot",
    "reset_memories",
    "add_memory_from_turn",
    "retrieve_memories",
    "format_memories_block",
    "export_all_memories",
]
