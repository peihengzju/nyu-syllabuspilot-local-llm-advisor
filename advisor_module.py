# advisor_module.py
"""
Module for academic advisor logic:

- Memory-only chat (no syllabus retrieval)
- Syllabus RAG + memory hybrid response
- Course recommendation (select from COURSE_PROFILES + syllabus + memory)
- Course comparison (two-course syllabus + memory)

Main public functions:
    - chat_with_memory_only(original_question: str) -> str
    - call_qwen_with_rag(original_question: str, retrieval_question: Optional[str]) -> str
    - answer_course_selection_question(original_question: str, retrieval_question: Optional[str]) -> str
    - answer_course_comparison_question(original_question: str, retrieval_question: Optional[str]) -> str
    - classify_course_for_selection(question: str) -> Optional[str]
    - COURSE_PROFILES: List[dict]
"""

from __future__ import annotations
import json
import re
from typing import List, Optional, Dict

from memory_module import retrieve_memories, format_memories_block
from rag_module import retrieve_context
from qwen_client import call_qwen
from config.course_catalog import load_course_catalog

# ================== Course Config (single source here) ==================

COURSE_PROFILES: List[Dict] = [
    {
        "code": c["code"],
        "name": c["name"],
        "focus": c["focus"],
        "keywords": c.get("keywords", []),
    }
    for c in load_course_catalog()
]


# ================== Utility: Extract Course Codes (comparison only) ==================

def _extract_course_codes(question: str) -> List[str]:
    """
    Extract course codes like ECE-GY 6143 / CS-GY 6923 from text.
    Same matching strategy as rag_module, but used only for comparison.
    """
    pattern = re.compile(r"\b[A-Z]{2,4}-?GY\s*\d{3,4}\b")
    codes = pattern.findall(question)
    return [c.strip() for c in codes]


def _normalize_course_code(code: str) -> str:
    c = (code or "").upper().replace("_", " ").replace("-", " ").strip()
    parts = c.split()
    if len(parts) == 3:
        return f"{parts[0]}-{parts[1]} {parts[2]}"
    return code.strip()


def _has_project_heavy_signal(text: str) -> bool:
    t = (text or "").lower()
    keys = [
        "project-heavy",
        "project heavy",
        "project based",
        "project-based",
        "hands-on",
        "practice-oriented",
        "build projects",
        "project focused",
    ]
    return any(k in t for k in keys)


def _project_signal_score(contexts: List[str]) -> int:
    blob = "\n".join(contexts).lower()
    score = 0
    score += blob.count("project")
    score += blob.count("phase")
    score += blob.count("hands-on")
    score += blob.count("lab")
    score += blob.count("assignment")
    score += blob.count("implement")
    return score


def _choose_project_heavy_course(
    candidate_codes: List[str],
    retrieval_question: str,
    original_question: str,
) -> Optional[str]:
    best_code = None
    best_score = -1
    for code in candidate_codes:
        ctx = retrieve_context(
            question=retrieval_question,
            forced_course=code,
            analysis_question=original_question,
        )
        score = _project_signal_score(ctx)
        if score > best_score:
            best_score = score
            best_code = code
    return best_code if best_score > 0 else None


def _tokenize_topic(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _choose_topic_matched_course(question: str) -> Optional[str]:
    """
    Deterministic topical matcher:
    pick the course with the strongest overlap between the user question and
    course keywords/focus/name. This stabilizes obvious cases like
    \"learn machine learning\" -> ML course.
    """
    q_tokens = _tokenize_topic(question)
    if not q_tokens:
        return None

    best_code = None
    best_score = 0
    second_score = 0

    for c in COURSE_PROFILES:
        fields = " ".join(
            [
                c.get("name", ""),
                c.get("focus", ""),
                " ".join(c.get("keywords", [])),
            ]
        ).lower()
        c_tokens = _tokenize_topic(fields)
        overlap = len(q_tokens & c_tokens)
        if overlap > best_score:
            second_score = best_score
            best_score = overlap
            best_code = c["code"]
        elif overlap > second_score:
            second_score = overlap

    # Require a meaningful and not-tied overlap.
    if best_score >= 2 and best_score > second_score:
        return best_code
    return None


# ================== Memory-Only Chat ==================

def chat_with_memory_only(original_question: str) -> str:
    """
    Pure conversation mode: no syllabus retrieval, memory + general chat only.
    Used for introductions, career planning, life questions, and casual chat.
    """
    mem_items = retrieve_memories(original_question, top_k=8)
    memory_block = format_memories_block(mem_items)

    system_prompt = (
        "You are a helpful assistant in a multi-turn chat.\n"
        "You see the user's latest message and some internal memory snippets about them.\n"
        "Your job is to reply like in a normal conversation, not to write a third-person profile.\n\n"
        "Rules:\n"
        "1) Always address the user in second person (\"you\"), never as \"the student\" or \"the user\".\n"
        "2) Write a natural chat-style answer in 1–3 sentences.\n"
        "3) If the user is sharing background or future plans, briefly acknowledge it and, if helpful, "
        "   restate their background/goals in second person.\n"
        "4) You may use the memory snippets to stay consistent, but DO NOT copy them verbatim; rephrase in your own words.\n"
        "5) In this mode you MUST NOT recommend or mention any specific NYU course codes "
        "   (e.g. 'ECE-GY 6143', 'ECE-GY 6483', 'CS-GY 6923'), "
        "   unless the user explicitly asks which course to take or mentions a course code in their message.\n"
        "6) Do not mention the existence of memory snippets explicitly."
    )

    user_prompt = (
        "===== Memory Snippets (for your reference only) =====\n"
        f"{memory_block}\n\n"
        "===== Latest User Message =====\n"
        f"{original_question}\n\n"
        "Now reply to the user in English, following the rules above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = call_qwen(
        messages,
        max_tokens=512,
        temperature=0.3,
        top_p=0.8,
    )
    return answer


# ================== Syllabus RAG + Memory ==================

def call_qwen_with_rag(
    original_question: str,
    retrieval_question: Optional[str] = None,
) -> str:
    """
    Syllabus RAG + memory:
    - Retrieve syllabus snippets via retrieve_context
    - Feed memory snippets and syllabus snippets to Qwen
    """
    rq = retrieval_question or original_question

    # 1) syllabus context
    contexts = retrieve_context(
        question=rq,
        analysis_question=original_question,
    )
    syllabus_block = (
        "\n\n".join(contexts)
        if contexts
        else "(No relevant syllabus snippets were retrieved.)"
    )

    # 2) memory context
    mem_items = retrieve_memories(original_question, top_k=5)
    memory_block = format_memories_block(mem_items)

    system_prompt = (
        "You are a teaching assistant familiar with NYU Tandon courses, and you also remember "
        "long-term facts and preferences about the student based on past conversations.\n\n"
        "You must obey these rules:\n"
        "1) For questions about courses/syllabi, rely primarily on the provided syllabus snippets.\n"
        "2) Use the memory snippets only for personalizing the answer (e.g., relating to the student's goals), "
        "   not for guessing missing syllabus details.\n"
        "3) Never fabricate syllabus details (grading breakdown, exam dates, policies) if they are not explicitly stated.\n"
        "4) Always answer the student's direct question first, then optionally add one short personalized remark.\n"
        "5) Keep answers short and focused: 1–3 sentences, at most 120 words.\n"
    )

    user_prompt = (
        "===== Student Memory Snippets =====\n"
        f"{memory_block}\n\n"
        "===== Syllabus Snippets =====\n"
        f"{syllabus_block}\n\n"
        "The student asked:\n"
        f"{original_question}\n\n"
        "Based only on the syllabus snippets and optionally using relevant memory snippets for personalization, "
        "answer the student's question in English.\n"
        "If a requested syllabus detail is not specified, say clearly that it is not specified.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = call_qwen(
        messages,
        max_tokens=512,
        temperature=0.2,
        top_p=0.8,
    )
    return answer


# ================== Course Selection (single recommendation) ==================

def classify_course_for_selection(question: str) -> Optional[str]:
    """
    Use Qwen to choose one best-matching course from COURSE_PROFILES.
    Returns course code.
    """
    courses_block_lines = []
    for c in COURSE_PROFILES:
        line = f"- {c['code']}: {c['name']} (focus: {c['focus']})"
        courses_block_lines.append(line)
    courses_block = "\n".join(courses_block_lines)

    system_prompt = (
        "You are an academic advisor at NYU Tandon.\n"
        "You will be given:\n"
        "1) A list of available courses with their code, name, and focus.\n"
        "2) A student's question about what they want to learn.\n\n"
        "Your job: choose exactly one best course from the list that matches the student's interests.\n"
        "If none of the courses clearly match, answer \"unknown\".\n"
        "Output ONLY a JSON object like:\n"
        "{\"course\": \"ECE-GY 6913\"}\n"
        "or\n"
        "{\"course\": \"unknown\"}\n"
    )

    user_prompt = (
        "Available courses:\n"
        f"{courses_block}\n\n"
        "Student question:\n"
        f"{question}\n\n"
        "Choose exactly one course code from the list above, or \"unknown\" if nothing fits."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw = call_qwen(
        messages,
        max_tokens=128,
        temperature=0.1,
        top_p=0.8,
    )

    try:
        result = json.loads(raw)
        course = result.get("course")
        if course and course != "unknown":
            return course
    except Exception:
        # If output is not valid JSON, treat as classification failure.
        pass
    return None


def answer_course_selection_question(
    original_question: str,
    retrieval_question: Optional[str] = None,
) -> str:
    """
    Course recommendation flow:
    use course list + memory + syllabus RAG for personalized recommendation.
    """

    # 1) Retrieve memory snippets for personalization.
    mem_items = retrieve_memories(original_question, top_k=5)
    memory_block = format_memories_block(mem_items)

    rq = retrieval_question or original_question
    # Project-heavy preference should come from the current question,
    # not historical memory, to avoid stale preference carry-over.
    prefer_project_heavy = _has_project_heavy_signal(original_question)

    # 2) Select the best-matching course from COURSE_PROFILES.
    course = None
    # First use deterministic topical matching for stability.
    course = _choose_topic_matched_course(original_question)
    if course is None and prefer_project_heavy:
        candidate_codes = [c["code"] for c in COURSE_PROFILES]
        course = _choose_project_heavy_course(
            candidate_codes=candidate_codes,
            retrieval_question=rq,
            original_question=original_question,
        )
    if course is None:
        course = classify_course_for_selection(original_question)
    if course is None:
        # Classification failed: fallback to generic syllabus RAG.
        return call_qwen_with_rag(original_question, retrieval_question)

    # 3) Run syllabus retrieval constrained to the selected course.
    contexts = retrieve_context(
        question=rq,
        forced_course=course,
        analysis_question=original_question,
    )
    context_block = (
        "\n\n".join(contexts)
        if contexts
        else "(No relevant syllabus snippets were retrieved.)"
    )

    # 4) Generate recommendation with memory + syllabus context.
    system_prompt = (
        "You are an academic advisor and teaching assistant at NYU Tandon.\n"
        "You know both the course syllabi and the student's long-term background and goals.\n"
        "You must use the memory snippets to personalize course recommendations.\n"
        "Keep answers short (1–3 sentences).\n"
        "The selected course is already the recommended one. Do not contradict this decision.\n"
        "Always answer the student's exact question first (e.g., compare the specific courses they mention), "
        "and only then briefly explain how the recommendation aligns with their long-term goals."
    )

    user_prompt = f"""
===== Student Memory Snippets =====
{memory_block}

===== Syllabus Snippets for the chosen course ({course}) =====
{context_block}

The student asked:
{original_question}

You (as advisor) have decided that the best matching course from the list is: {course}.

Using BOTH:
1) the student's background and goals from the memory snippets, and
2) the topics / workload shown in the syllabus snippets,

give a brief English answer (1–3 sentences) that:
- clearly recommends this course as the best fit, and
- if relevant, mentions why this course is better aligned with their goals than the other options mentioned in the question.
If the syllabus does not show any low-level / architecture / embedded content, state that clearly.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = call_qwen(
        messages,
        max_tokens=256,
        temperature=0.2,
        top_p=0.8,
    )
    return answer


# ================== Course Comparison (two courses) ==================

def answer_course_comparison_question(
    original_question: str,
    retrieval_question: Optional[str] = None,
) -> str:
    """
    Course comparison flow (A vs B):
    - Extract two course codes
    - Retrieve syllabus context for each course
    - Use memory snippets for personalized comparison
    """
    codes = _extract_course_codes(original_question)
    if len(codes) < 2:
        # If two course codes are not found, fallback to generic RAG path.
        return call_qwen_with_rag(original_question, retrieval_question)

    # Use first two codes only.
    code_a = _normalize_course_code(codes[0])
    code_b = _normalize_course_code(codes[1])

    rq = retrieval_question or original_question

    # Retrieve syllabus snippets for each course.
    contexts_a = retrieve_context(
        question=rq,
        forced_course=code_a,
        analysis_question=original_question,
    )
    contexts_b = retrieve_context(
        question=rq,
        forced_course=code_b,
        analysis_question=original_question,
    )

    context_block_a = (
        "\n\n".join(contexts_a) if contexts_a else "(No syllabus snippets retrieved for this course.)"
    )
    context_block_b = (
        "\n\n".join(contexts_b) if contexts_b else "(No syllabus snippets retrieved for this course.)"
    )

    # Use memory snippets for personalization.
    mem_items = retrieve_memories(original_question, top_k=5)
    memory_block = format_memories_block(mem_items)

    prefer_project_heavy = _has_project_heavy_signal(
        original_question + "\n" + memory_block
    )
    if prefer_project_heavy:
        score_a = _project_signal_score(contexts_a)
        score_b = _project_signal_score(contexts_b)
        if abs(score_a - score_b) >= 2:
            better = code_a if score_a > score_b else code_b
            other = code_b if better == code_a else code_a
            return (
                f"I recommend {better} because it appears more project-heavy in the syllabus snippets. "
                f"Compared with {other}, it shows stronger hands-on/project signals, which aligns better with your project-first preference."
            )

    system_prompt = (
        "You are an academic advisor and teaching assistant at NYU Tandon.\n"
        "You will compare two specific courses for the same student using both syllabi and the student's background.\n\n"
        "Rules:\n"
        "1) First, directly answer the student's comparison question (which course is more suitable for them).\n"
        "2) Then briefly justify your choice using concrete topics / workload from the syllabus snippets of both courses.\n"
        "3) Use the memory snippets only to personalize the reasoning (e.g., relate to their interest in embedded systems, AI infra, etc.).\n"
        "4) Do NOT fabricate syllabus details that are not explicitly present in the snippets.\n"
        "5) Keep the answer concise: 2–4 sentences at most."
    )

    user_prompt = f"""
===== Student Memory Snippets =====
{memory_block}

===== Syllabus Snippets for {code_a} =====
{context_block_a}

===== Syllabus Snippets for {code_b} =====
{context_block_b}

The student asked:
{original_question}

Based on the two syllabi and the student's background, decide which of {code_a} and {code_b} is more suitable for this student.
First clearly state which course you recommend, then briefly compare them and explain why that choice fits the student's goals better.
Answer in English, in 2–4 sentences.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = call_qwen(
        messages,
        max_tokens=320,
        temperature=0.2,
        top_p=0.8,
    )
    return answer


__all__ = [
    "COURSE_PROFILES",
    "chat_with_memory_only",
    "call_qwen_with_rag",
    "classify_course_for_selection",
    "answer_course_selection_question",
    "answer_course_comparison_question",
]
