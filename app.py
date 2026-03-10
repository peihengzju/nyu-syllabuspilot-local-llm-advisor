# app.py
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from itertools import count

from flask import Flask, request, jsonify, render_template
# ===== Local project modules =====
from memory_module import (
    reset_memories,
    add_memory_from_turn,
    export_all_memories,
)
from rag_module import (
    refine_question_with_qwen,
    classify_intent,
    is_course_selection_question,
    is_syllabus_question,
    is_course_comparison_question,
    retrieve_context,
)
import rag_module
from advisor_module import (
    chat_with_memory_only,
    call_qwen_with_rag,
    answer_course_selection_question,
    answer_course_comparison_question,
)
from schedule_module import try_generate_schedule_from_dialog, reset_schedule_state
from config.paths import (
    RESET_MEMORY_ON_INDEX,
    INTENT_CONFIDENCE_THRESHOLD,
    ENABLE_REQUEST_LOG,
)

# Qwen HTTP calls are encapsulated in qwen_client.
# app.py does not call Qwen directly.

app = Flask(__name__)

TURN_COUNTER = count(1)  # 1,2,3,...
MEMORY_LOG_PATH = os.path.join("memory_store", "memory_export.log")

# Memory export logger
mem_logger = logging.getLogger("memory_export")
if not mem_logger.handlers:
    os.makedirs(os.path.dirname(MEMORY_LOG_PATH), exist_ok=True)
    handler = logging.FileHandler(MEMORY_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    mem_logger.setLevel(logging.INFO)
    mem_logger.addHandler(handler)

# ============== Web Page + Memory Reset ==============

@app.route("/", methods=["GET"])
def index():
    """
    Optionally reset memory when opening the main page.
    Controlled by RESET_MEMORY_ON_INDEX.
    """
    if RESET_MEMORY_ON_INDEX:
        reset_memories()
    return render_template("index.html")


@app.route("/reset_memory", methods=["POST"])
def reset_memory_route():
    """
    Explicit endpoint to reset memory.
    """
    reset_memories()
    reset_schedule_state()
    return jsonify({"status": "ok"})


# ============== Main Q&A Endpoint ==============

@app.route("/ask", methods=["POST"])
def ask():
    request_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    # 1) Read question
    if request.is_json:
        question = request.json.get("question", "")
    else:
        question = request.form.get("question", "")

    if not question or not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    try:
        refined = None
        route = None
        intent = "unknown"
        confidence = 0.0

        handled, payload = try_generate_schedule_from_dialog(question=question)
        if handled:
            route = "schedule"
            if payload.get("type") == "html":
                answer = "Here is your generated weekly schedule."
                turn_id = next(TURN_COUNTER)
                add_memory_from_turn(question, answer, source_turn=turn_id)
                if request.is_json:
                    return jsonify(
                        {
                            "answer": answer,
                            "request_id": request_id,
                            "intent": "schedule",
                            "schedule_html": payload.get("html", ""),
                            "conflicts": payload.get("conflicts", []),
                        }
                    )
                return payload.get("html", "")

            answer = payload.get("message", "I need more information to build the schedule.")
            turn_id = next(TURN_COUNTER)
            add_memory_from_turn(question, answer, source_turn=turn_id)
            if request.is_json:
                return jsonify(
                    {
                        "answer": answer,
                        "request_id": request_id,
                        "intent": "schedule",
                        "candidates": payload.get("candidates", []),
                    }
                )
            return answer

        intent_debug = classify_intent(question)
        intent = intent_debug["intent"]
        confidence = float(intent_debug["confidence"])

        # 2) Intent routing: comparison / selection / syllabus / general chat
        if confidence < INTENT_CONFIDENCE_THRESHOLD and intent != "chat":
            # Conservative fallback for low-confidence intent.
            route = "fallback_rag_low_confidence"
            refined = refine_question_with_qwen(question)
            answer = call_qwen_with_rag(
                original_question=question,
                retrieval_question=refined,
            )

        elif intent == "comparison" or is_course_comparison_question(question):
            # Course comparison with dual-syllabus retrieval + memory.
            route = "comparison"
            refined = refine_question_with_qwen(question)
            answer = answer_course_comparison_question(
                original_question=question,
                retrieval_question=refined,
            )

        elif intent == "selection" or is_course_selection_question(question):
            # Course recommendation from COURSE_PROFILES + syllabus + memory.
            route = "selection"
            refined = refine_question_with_qwen(question)
            answer = answer_course_selection_question(
                original_question=question,
                retrieval_question=refined,
            )

        elif intent == "syllabus" or is_syllabus_question(question):
            # Syllabus detail Q&A with syllabus RAG + memory.
            route = "syllabus_rag"
            refined = refine_question_with_qwen(question)
            answer = call_qwen_with_rag(
                original_question=question,
                retrieval_question=refined,
            )

        else:
            # General conversation: memory-only chat without syllabus retrieval.
            route = "memory_chat"
            answer = chat_with_memory_only(question)
            # Safety: suppress random course-code mentions.
            answer = re.sub(
                r"\b[A-Z]{2,4}-?GY\s*\d{3,4}\b",
                "this course",
                answer,
            )

        # 3) Persist memory for every turn.
        turn_id = next(TURN_COUNTER)
        add_memory_from_turn(question, answer, source_turn=turn_id)

        if ENABLE_REQUEST_LOG:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            app.logger.info(
                "request_id=%s route=%s intent=%s confidence=%.3f refined=%s latency_ms=%d",
                request_id,
                route,
                intent,
                confidence,
                bool(refined),
                elapsed_ms,
            )

    except Exception as e:
        return jsonify({
            "error": f"Failed to process request: {e}",
            "request_id": request_id,
        }), 500

    # 4) Response format: JSON mode / form mode
    if request.is_json:
        return jsonify({
            "answer": answer,
            "request_id": request_id,
            "intent": intent,
        })
    else:
        return f"""
        <html>
        <head><meta charset="utf-8"><title>Answer</title></head>
        <body>
          <p><b>Question:</b> {question}</p>
          <p><b>Request ID:</b> {request_id}</p>
          <hr>
          <pre>{answer}</pre>
          <a href="/">Back to main page</a>
        </body>
        </html>
        """


# ============== RAG Debug Endpoint ==============

@app.route("/debug_retrieval", methods=["POST"])
def debug_retrieval():
    """
    Debug syllabus retrieval:
    - Return refined query
    - Return retrieved contexts
    - Return rag_module.LAST_RETRIEVAL_DEBUG
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"error": "Question must not be empty."}), 400

    request_id = uuid.uuid4().hex[:12]
    refined = refine_question_with_qwen(question)
    contexts = retrieve_context(
        question=refined,
        analysis_question=question,
    )

    return jsonify({
        "request_id": request_id,
        "question": question,
        "refined": refined,
        "num_contexts": len(contexts),
        "contexts": contexts,
        "debug": rag_module.LAST_RETRIEVAL_DEBUG,
    })


@app.route("/log_memories", methods=["POST"])
def log_memories():
    """
    Called by frontend on close/refresh to snapshot memory into logs.
    """
    data = request.get_json(force=True, silent=True) or {}
    reason = data.get("reason", "unload")
    now = datetime.utcnow().isoformat()

    try:
        snapshot = export_all_memories()
        mem_logger.info(
            "Memory snapshot reason=%s at=%s count=%d payload=%s",
            reason,
            now,
            len(snapshot),
            json.dumps(snapshot, ensure_ascii=False),
        )
        return jsonify({"status": "logged", "count": len(snapshot)})
    except Exception as e:
        mem_logger.exception("Failed to export memories on unload: %s", e)
        return jsonify({"error": "Failed to export memories"}), 500


if __name__ == "__main__":
    print("RAG + Qwen server is running.")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)
