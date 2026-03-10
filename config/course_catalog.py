import json
from functools import lru_cache
from typing import Dict, List, Tuple

from config.paths import COURSES_CONFIG_PATH


def _normalize_code(code: str) -> str:
    c = (code or "").upper().replace("_", " ").replace("-", " ").strip()
    parts = c.split()
    if len(parts) == 3:
        return f"{parts[0]}-{parts[1]} {parts[2]}"
    return code.strip()


@lru_cache(maxsize=1)
def load_course_catalog() -> List[Dict]:
    with open(COURSES_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    courses = data.get("courses", [])
    if not isinstance(courses, list) or not courses:
        raise ValueError("config/courses.json must contain a non-empty 'courses' list")

    seen = set()
    normalized = []

    for entry in courses:
        code = _normalize_code(entry.get("code", ""))
        if not code:
            raise ValueError("Every course must have a non-empty code")
        if code in seen:
            raise ValueError(f"Duplicate course code found: {code}")

        keywords = entry.get("keywords", [])
        if not isinstance(keywords, list) or not keywords:
            raise ValueError(f"Course {code} must define non-empty keywords")

        seen.add(code)
        normalized.append(
            {
                "code": code,
                "name": entry.get("name", ""),
                "focus": entry.get("focus", ""),
                "keywords": [str(k) for k in keywords if str(k).strip()],
            }
        )

    return normalized


@lru_cache(maxsize=1)
def load_course_keyword_map() -> Dict[str, List[str]]:
    return {c["code"]: c.get("keywords", []) for c in load_course_catalog()}


@lru_cache(maxsize=1)
def load_course_lookup() -> Dict[str, Dict]:
    return {c["code"]: c for c in load_course_catalog()}


__all__ = [
    "load_course_catalog",
    "load_course_keyword_map",
    "load_course_lookup",
]
