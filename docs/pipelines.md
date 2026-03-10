# Pipeline Design and Improvement Plan

This document summarizes current pipelines and prioritized improvements.

## P0: Request Routing (Intent + Confidence)
Current:
- Rule-driven intent scoring with confidence.
- Low-confidence non-chat requests fallback to generic syllabus RAG.

Why it matters:
- Prevents hard misroutes when intent is ambiguous.

Next improvements:
1. Add optional LLM-based intent classifier as second-stage resolver.
2. Add offline intent evaluation set.

## P0: Retrieval (Hybrid + Rerank)
Current:
- Dense retrieval (Faiss) + lexical retrieval.
- RRF fusion.
- Heuristic boosting by question type and course code hints.
- Grading/exam lexical fallback snippets.

Why it matters:
- Improves robustness on structured syllabus facts.

Next improvements:
1. Add feature-weight calibration from eval data.
2. Add rerank score attribution in debug output.

## P1: Single Source of Course Truth
Current:
- Course metadata loaded from `config/courses.json` via `config/course_catalog.py`.
- Both advisor and retrieval logic consume same source.

Why it matters:
- Avoids config drift across modules.

Next improvements:
1. Add schema validation and CI check for `courses.json`.

## P1: Memory Write/Retrieve Policy
Current:
- Every turn writes memory.
- Index rebuild occurs periodically (`MEMORY_REINDEX_EVERY`).
- Retrieval falls back to lexical mode when index is unavailable/mismatched.

Why it matters:
- Reduces per-turn overhead while keeping retrieval continuity.

Next improvements:
1. Add session vs long-term memory separation.
2. Add pruning policy for stale low-value entries.

## P2: Incremental Ingestion
Current:
- File hash (sha256) based reuse.
- Per-file docs/embedding cache.
- Rebuilds global index from cached + changed files.
- Writes manifest report.

Why it matters:
- Avoids full recomputation when only a few PDFs change.

Next improvements:
1. Add per-course index shards and merge strategy.
2. Add ingestion regression snapshot tests.

## P2: Observability
Current:
- request_id propagation in responses/logs.
- Retrieval debug payload includes mode, course route source, and top candidates.

Why it matters:
- Faster diagnosis of quality or latency regressions.

Next improvements:
1. Structured JSON logs for request lifecycle.
2. Latency histograms by route and model call count.
