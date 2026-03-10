"""Schedule intent handling and HTML timetable rendering."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

from flask import render_template, url_for

from course_db import CourseDB
from course_scheduler import build_schedule, minutes_to_time_str, parse_meeting

SCHEDULE_INTENT_PATTERNS = [
    r"\b(generate|build|create|make)\s+(a\s+)?(schedule|timetable)\b",
    r"\b(my\s+)?schedule\b",
    r"\b(add|drop|remove)\s+.*\b(schedule|timetable)\b",
    r"\b(schedule|timetable)\s+(planner|plan)\b",
]

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,6}-?(?:GY)?\s*\d{3,4}\b", re.I)
COURSE_NUM_RE = re.compile(r"\b\d{3,5}\b")
LINE_PARSE_RE = re.compile(r"^\s*([^|,]+?)[|,]\s*([^|,]+?)(?:[|,]\s*(.+))?$")


_course_db_singleton: Optional[CourseDB] = None
SCHEDULE_STATE_PATH = os.path.join("memory_store", "schedule_state.json")


def _get_course_db() -> CourseDB:
    global _course_db_singleton
    if _course_db_singleton is None:
        _course_db_singleton = CourseDB()
    return _course_db_singleton


def detect_schedule_intent(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(p, lowered) for p in SCHEDULE_INTENT_PATTERNS)


def _normalize_code(code: str) -> str:
    s = (code or "").upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    m = re.match(r"^([A-Z]{2,6})(?:GY)?(\d{3,5})$", s)
    if not m:
        return s
    return f"{m.group(1)}-GY{m.group(2)}"


def _schedule_action(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["clear schedule", "reset schedule", "new schedule from scratch"]):
        return "reset"
    if "schedule" in t and any(k in t for k in ["also", "as well", "too", "if i'll take", "if i take"]):
        return "add"
    if any(k in t for k in ["add ", "append ", "include "]) and "schedule" in t:
        return "add"
    if any(k in t for k in ["remove ", "drop ", "delete "]) and "schedule" in t:
        return "remove"
    if any(k in t for k in ["generate", "build", "create", "make"]):
        return "replace"
    if "check" in t and "conflict" in t and "schedule" in t:
        return "inspect"
    # Default to replace to keep behavior intuitive for first schedule creation.
    return "replace"


def _load_schedule_state() -> List[Dict[str, str]]:
    if not os.path.exists(SCHEDULE_STATE_PATH):
        return []
    try:
        with open(SCHEDULE_STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            return []
        return [x for x in payload if isinstance(x, dict)]
    except Exception:
        return []


def _save_schedule_state(courses: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(SCHEDULE_STATE_PATH), exist_ok=True)
    with open(SCHEDULE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)


def reset_schedule_state() -> None:
    _save_schedule_state([])


def _line_to_course(line: str) -> Optional[Dict[str, str]]:
    line = line.strip()
    if not line:
        return None

    m = LINE_PARSE_RE.match(line)
    if m:
        return {
            "code": m.group(1).strip(),
            "name": m.group(2).strip(),
            "meetings": (m.group(3) or "").strip(),
        }

    code_match = COURSE_CODE_RE.search(line)
    if code_match:
        code = code_match.group(0).strip()
        return {"code": code, "name": code, "meetings": ""}

    return None


def extract_courses_from_text(text: str) -> List[Dict[str, str]]:
    if not text:
        return []

    candidates: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        codes = [c.strip() for c in COURSE_CODE_RE.findall(line)]
        if len(set(codes)) >= 2:
            for code in sorted(set(codes)):
                candidates.append({"code": code, "name": code, "meetings": ""})
            continue

        parsed = _line_to_course(line)
        if parsed:
            candidates.append(parsed)

    if candidates:
        return candidates

    seen = set()
    for code in COURSE_CODE_RE.findall(text):
        c = code.strip()
        if c in seen:
            continue
        seen.add(c)
        candidates.append({"code": c, "name": c, "meetings": ""})
    if candidates:
        return candidates

    # Support shorthand like "add 6913".
    for num in COURSE_NUM_RE.findall(text):
        if num in seen:
            continue
        seen.add(num)
        candidates.append({"code": num, "name": num, "meetings": ""})
    return candidates


def _fill_from_db(courses: List[Dict[str, str]]) -> None:
    db = _get_course_db()
    for c in courses:
        query = c.get("code") or c.get("name") or ""
        info = db.find_course_info(query)
        # Second-pass fallback: if query formatting is messy, try numeric ID.
        if (not info.get("meetings")) and query:
            m = re.search(r"\b(\d{3,5})\b", str(query))
            if m:
                info = db.find_course_info(m.group(1))
        if info and info.get("name") != "Unknown Course" and info.get("code"):
            c["code"] = str(info["code"]).strip()
        if not c.get("name") and info.get("name"):
            c["name"] = info["name"]
        if not c.get("meetings") and info.get("meetings"):
            c["meetings"] = "; ".join(info["meetings"])
        if not c.get("room") and info.get("rooms"):
            c["room"] = "; ".join(info["rooms"])


def generate_schedule_html_from_courses(
    courses: List[Dict[str, str]],
    day_start: int = 8 * 60,
    day_end: int = 20 * 60,
    slot_minutes: int = 30,
) -> Tuple[str, List[Dict]]:
    grid, conflicts = build_schedule(
        courses,
        day_start=day_start,
        day_end=day_end,
        slot_minutes=slot_minutes,
    )

    n_slots = len(next(iter(grid.values())))
    display_map: Dict[str, str] = {}
    course_entries: List[Dict[str, str]] = []

    for c in courses:
        code = str(c.get("code") or "").strip()
        name = str(c.get("name") or "").strip()
        display = f"{code} {name}".strip() if name and name.lower() != code.lower() else (code or name)
        display_map[code or name] = display
        course_entries.append({
            "code": code,
            "name": name,
            "meetings": c.get("meetings", ""),
            "room": c.get("room", ""),
        })

    slot_range_map: Dict[Tuple[str, int, int], Tuple[str, str, int]] = {}
    for c in courses:
        key_code = str(c.get("code") or c.get("name") or "").strip()
        for day_idx, smin, emin in parse_meeting(str(c.get("meetings") or "")):
            start_slot = max(0, (smin - day_start) // slot_minutes)
            end_slot = min(n_slots, max(0, math.ceil((emin - day_start) / slot_minutes)))
            slot_range_map[(key_code, day_idx, start_slot)] = (
                minutes_to_time_str(smin),
                minutes_to_time_str(emin),
                end_slot,
            )

    cell_info = {d: [None for _ in range(n_slots)] for d in range(7)}
    for day_idx in range(7):
        slot = 0
        while slot < n_slots:
            cell_str = grid[day_idx][slot][0]
            if not cell_str:
                cell_info[day_idx][slot] = {"html": "", "rowspan": 1}
                slot += 1
                continue

            span = 1
            while slot + span < n_slots and grid[day_idx][slot + span][0] == cell_str:
                span += 1

            parts = [p.strip() for p in cell_str.split(",") if p.strip()]
            display_parts = [display_map.get(p, p) for p in parts]
            html = "<br>".join(display_parts)

            start_str = ""
            end_str = ""
            use_span = span
            if parts:
                key = (parts[0], day_idx, slot)
                if key in slot_range_map:
                    start_str, end_str, end_slot = slot_range_map[key]
                    use_span = max(1, end_slot - slot)

            cell_info[day_idx][slot] = {
                "html": html,
                "multiple": len(parts) > 1,
                "rowspan": use_span,
                "start_time": start_str,
                "end_time": end_str,
            }
            for offset in range(1, use_span):
                cell_info[day_idx][slot + offset] = {"skip": True}
            slot += use_span

    rows = []
    for slot in range(n_slots):
        t0 = day_start + slot * slot_minutes
        rows.append(
            {
                "time": minutes_to_time_str(t0),
                "time_label": minutes_to_time_str(t0),
                "cells": [cell_info[d][slot] for d in range(7)],
            }
        )

    html = render_template(
        "schedule.html",
        rows=rows,
        day_names=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        conflicts=conflicts,
        course_entries=course_entries,
        slot_minutes=slot_minutes,
        logo_url=url_for("static", filename="nyu-logo.png", _external=True),
        home_url=url_for("index", _external=True),
    )
    return html, conflicts


def try_generate_schedule_from_dialog(question: str, answer: str = "") -> Tuple[bool, Dict]:
    combined = f"{question or ''}\n{answer or ''}"
    if not detect_schedule_intent(combined):
        return False, {}

    action = _schedule_action(combined)
    existing = _load_schedule_state()
    candidates = extract_courses_from_text(combined)

    if action == "reset":
        _save_schedule_state([])
        return True, {
            "type": "ask_for_confirmation",
            "message": "Schedule has been cleared. Add courses to generate a new timetable.",
            "candidates": [],
        }

    if action == "inspect":
        if not existing:
            return True, {
                "type": "ask_for_confirmation",
                "message": "Your schedule is empty. Add courses first, then I can check conflicts.",
                "candidates": [],
            }
        html, conflicts = generate_schedule_html_from_courses(existing)
        return True, {"type": "html", "html": html, "conflicts": conflicts}

    if not candidates and action in {"add", "remove"}:
        verb = "add" if action == "add" else "remove"
        return True, {
            "type": "ask_for_confirmation",
            "message": (
                f"I understood a schedule update ({verb}), but I need course codes first "
                "(example: 'Add ECE-GY 6913 to my schedule')."
            ),
            "candidates": [],
        }
    if not candidates:
        return True, {
            "type": "ask_for_confirmation",
            "message": (
                "I can generate a weekly schedule, but I need course codes first "
                "(example: 'Generate schedule for ECE-GY 6143 and ECE-GY 6483')."
            ),
            "candidates": [],
        }

    _fill_from_db(candidates)
    merged: List[Dict[str, str]]
    if action == "add":
        by_code = {_normalize_code(c.get("code", "")): c for c in existing}
        for c in candidates:
            code = _normalize_code(c.get("code", ""))
            if not code:
                continue
            if code in by_code:
                old = by_code[code]
                if not old.get("meetings") and c.get("meetings"):
                    old["meetings"] = c["meetings"]
                if not old.get("room") and c.get("room"):
                    old["room"] = c["room"]
                if (not old.get("name")) and c.get("name"):
                    old["name"] = c["name"]
            else:
                by_code[code] = c
        merged = list(by_code.values())
    elif action == "remove":
        remove_codes = {_normalize_code(c.get("code", "")) for c in candidates}
        merged = [
            c for c in existing
            if _normalize_code(c.get("code", "")) not in remove_codes
        ]
    else:
        # replace: generate a new schedule from provided candidates.
        merged = candidates

    # Heal stale schedule state by filling all courses from catalog, not only new candidates.
    _fill_from_db(merged)

    if any(not c.get("meetings") for c in merged):
        return True, {
            "type": "ask_for_confirmation",
            "message": (
                "I found course candidates, but at least one course has no meeting time in your catalog. "
                "Please provide explicit meeting times (example: ECE-GY 6143|Intro ML|Tue 11:00-13:30)."
            ),
            "candidates": merged,
        }

    _save_schedule_state(merged)
    html, conflicts = generate_schedule_html_from_courses(merged)
    return True, {"type": "html", "html": html, "conflicts": conflicts}
