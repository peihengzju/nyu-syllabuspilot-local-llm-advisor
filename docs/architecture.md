# Architecture

## Goal
This project is a Flask-based academic assistant that combines local LLM generation (Qwen), syllabus retrieval (RAG), and long-term memory to answer course questions and provide personalized recommendations.

## Layers
1. Interface layer
- `app.py`
- Routes: `/ask`, `/reset_memory`, `/debug_retrieval`, `/log_memories`

2. Business logic layer
- `advisor_module.py`
- Flows:
  - memory-only chat
  - syllabus + memory answer
  - course recommendation
  - two-course comparison

3. Retrieval layer
- `rag_module.py`
- Components:
  - query rewrite
  - intent detection
  - hybrid retrieval (dense + lexical)
  - reranking and course focusing
  - debug trace (`LAST_RETRIEVAL_DEBUG`)

4. Memory layer
- `memory_module.py`
- Components:
  - turn-level memory write
  - aggregated profile/career memories
  - retrieval with similarity + time decay + importance

5. Ingestion layer
- `ingest_syllabi.py`
- Components:
  - PDF extraction
  - structured chunk extraction by section type
  - embedding + Faiss index build
  - incremental build via manifest/cache

## Runtime Request Flow (`POST /ask`)
1. Read question.
2. Detect intent + confidence.
3. Route to one of: comparison / selection / syllabus-rag / memory-chat.
4. Retrieve memory and syllabus snippets where needed.
5. Call local Qwen API for final response.
6. Persist memory for the turn.
7. Return answer + request_id.

## Retrieval Debug Flow (`POST /debug_retrieval`)
1. Rewrite query.
2. Retrieve contexts.
3. Return contexts + `LAST_RETRIEVAL_DEBUG` fields.

## State & Storage
- Syllabus vectors: `vector_store/index.faiss`, `vector_store/texts.json`
- Ingestion manifest/cache: `vector_store/ingest_manifest.json`, `vector_store/cache/*`
- Conversation memory: `memory_store/memories.json`, `memory_store/mem_index.faiss`
- Memory logs: `memory_store/memory_export.log`

## Key Design Choices
- Rule-based intent detection with confidence fallback.
- Hybrid retrieval with RRF fusion and heuristic boosting.
- Course-focused context narrowing for non-comparison queries.
- Periodic memory index rebuild with lexical fallback.
