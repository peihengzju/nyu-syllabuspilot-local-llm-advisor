# config/__init__.py
import json
import os
from .paths import PROJECT_ROOT

_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "courses.json")

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _COURSE_DATA = json.load(f)

# Used by RAG keyword routing.
COURSE_FILE_HINTS = {
    c["code"]: c.get("keywords", [])
    for c in _COURSE_DATA["courses"]
}

# Used by course advisor selection logic.
COURSE_PROFILES = [
    {
        "code": c["code"],
        "name": c["name"],
        "focus": c["focus"],
    }
    for c in _COURSE_DATA["courses"]
]
