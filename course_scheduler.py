"""Utilities for parsing course meetings, building a weekly grid, and conflict detection."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

DAY_MAP = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def time_str_to_minutes(raw: str) -> int:
    m = re.match(r"^(\d{1,2}):(\d{2})$", raw.strip())
    if not m:
        raise ValueError(f"Invalid time string: {raw!r}")
    return int(m.group(1)) * 60 + int(m.group(2))


def minutes_to_time_str(value: int) -> str:
    h = value // 60
    mm = value % 60
    return f"{h:02d}:{mm:02d}"


def _split_days_token(token: str) -> List[str]:
    token = token.strip()
    if not token:
        return []
    for sep in ["/", ",", "&", ";", "|"]:
        if sep in token:
            return [p.strip() for p in token.split(sep) if p.strip()]
    parts = re.findall(r"[A-Za-z]{2,10}\\.?", token)
    return parts or [token]


def parse_meeting(meeting_str: str) -> List[Tuple[int, int, int]]:
    """Parse meeting strings like 'Tue 11:00-13:30; Thu 11:00-13:30'."""
    if not meeting_str:
        return []

    parts = [p.strip() for p in re.split(r"[;|]", meeting_str) if p.strip()]
    time_re = re.compile(r"(\d{1,2}:\d{2})\s*[-\u2013]\s*(\d{1,2}:\d{2})")
    out: List[Tuple[int, int, int]] = []

    for part in parts:
        m = time_re.search(part)
        if not m:
            continue

        start = time_str_to_minutes(m.group(1))
        end = time_str_to_minutes(m.group(2))
        if end <= start:
            continue

        day_token = part[: m.start()].strip() or part[m.end() :].strip()
        if not day_token:
            continue

        for raw_day in _split_days_token(day_token):
            key = raw_day.lower().strip().rstrip(".")
            day_idx = DAY_MAP.get(key)
            if day_idx is None:
                key2 = re.sub(r"[^a-z]", "", key)
                day_idx = DAY_MAP.get(key2)
            if day_idx is None:
                for dname, idx in DAY_MAP.items():
                    if key.startswith(dname):
                        day_idx = idx
                        break
            if day_idx is None:
                continue
            out.append((day_idx, start, end))

    return out


def build_schedule(
    courses: List[Dict[str, Any]],
    day_start: int = 8 * 60,
    day_end: int = 20 * 60,
    slot_minutes: int = 30,
) -> Tuple[Dict[int, List[List[str]]], List[Dict[str, Any]]]:
    n_slots = math.ceil((day_end - day_start) / slot_minutes)
    grid = {d: [[""] for _ in range(n_slots)] for d in range(7)}

    meetings: List[Tuple[str, int, int, int]] = []
    for course in courses:
        code = str(course.get("code") or course.get("name") or "").strip()
        for day_idx, start_min, end_min in parse_meeting(str(course.get("meetings") or "")):
            meetings.append((code, day_idx, start_min, end_min))

    conflicts: List[Dict[str, Any]] = []
    by_day: Dict[int, List[Tuple[str, int, int, int]]] = {d: [] for d in range(7)}
    for item in meetings:
        by_day[item[1]].append(item)

    for day_idx in range(7):
        day_meetings = sorted(by_day[day_idx], key=lambda x: x[2])
        for i, (code, _d, start_min, end_min) in enumerate(day_meetings):
            start_slot = max(0, (start_min - day_start) // slot_minutes)
            end_slot = min(n_slots, math.ceil((end_min - day_start) / slot_minutes))

            for slot in range(start_slot, end_slot):
                current = grid[day_idx][slot][0]
                if not current:
                    grid[day_idx][slot][0] = code
                else:
                    parts = [p.strip() for p in current.split(",") if p.strip()]
                    if code not in parts:
                        parts.append(code)
                    grid[day_idx][slot][0] = ",".join(parts)

            for j in range(i + 1, len(day_meetings)):
                code_b, _d2, s2, e2 = day_meetings[j]
                if s2 >= end_min:
                    break
                conflicts.append(
                    {
                        "course_a": code,
                        "course_b": code_b,
                        "day": day_idx,
                        "overlap_start": max(start_min, s2),
                        "overlap_end": min(end_min, e2),
                    }
                )

    return grid, conflicts


def format_conflicts(conflicts: List[Dict[str, Any]]) -> str:
    if not conflicts:
        return "No conflicts detected."
    lines = []
    for c in conflicts:
        day = DAY_NAMES[c["day"]]
        start = minutes_to_time_str(c["overlap_start"])
        end = minutes_to_time_str(c["overlap_end"])
        lines.append(f"{c['course_a']} conflicts with {c['course_b']} on {day} {start}-{end}")
    return "\n".join(lines)
