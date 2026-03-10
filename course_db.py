"""In-memory course metadata index for schedule completion and lookup."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_COURSE_PATH = Path(__file__).resolve().parent / "config" / "courses.json"


class CourseDB:
    def __init__(self, course_json_path: Optional[str | Path] = None):
        self.path = Path(course_json_path) if course_json_path else DEFAULT_COURSE_PATH
        self.courses_map: Dict[str, Dict] = {}
        self._load_data()

    def _load_data(self) -> None:
        if not self.path.exists():
            print(f"[CourseDB] Missing file: {self.path}")
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"[CourseDB] Invalid JSON at {self.path}: {exc}")
            return

        if isinstance(payload, dict):
            if "courses" in payload and isinstance(payload["courses"], list):
                course_list = payload["courses"]
            elif all(isinstance(v, dict) for v in payload.values()):
                course_list = [{"code": k, **v} for k, v in payload.items()]
            else:
                print(f"[CourseDB] Unsupported JSON structure in {self.path}")
                return
        elif isinstance(payload, list):
            course_list = payload
        else:
            print(f"[CourseDB] Unsupported root type: {type(payload)}")
            return

        self.courses_map.clear()
        for item in course_list:
            if not isinstance(item, dict):
                continue

            raw_code = str(item.get("code") or "").strip()
            if not raw_code:
                continue

            norm_code = self._normalize_code(raw_code)
            if not norm_code:
                continue

            meetings = self._normalize_list(item.get("meetings") or item.get("schedule"))
            rooms = self._normalize_list(item.get("rooms") or item.get("location"))
            instructors = self._normalize_list(item.get("instructors") or item.get("instructor"))

            self.courses_map[norm_code] = {
                "code": raw_code,
                "name": str(item.get("name") or "").strip(),
                "meetings": meetings,
                "rooms": rooms,
                "instructors": instructors,
                "description": str(item.get("description") or "").strip(),
            }

    def _normalize_list(self, value: Iterable | None) -> List[str]:
        if value is None:
            return []
        values = [value] if isinstance(value, str) else list(value)
        out: List[str] = []
        seen = set()
        for v in values:
            s = str(v).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _normalize_code(self, code: str) -> str:
        s = code.upper().strip()
        if re.fullmatch(r"\d{3,5}", s):
            return s

        s = re.sub(r"[^A-Z0-9]", "", s)
        m = re.match(r"^([A-Z]{2,6})(?:GY)?(\d{3,5})", s)
        if not m:
            return ""
        return f"{m.group(1)}-GY{m.group(2)}"

    def find_course_info(self, query_code: str) -> Dict:
        norm = self._normalize_code(str(query_code or ""))
        if not norm:
            return self._unknown(query_code)

        if norm in self.courses_map:
            return self.courses_map[norm]

        if re.fullmatch(r"\d{3,5}", norm):
            hits = [v for k, v in self.courses_map.items() if k.endswith(norm)]
            if hits:
                return hits[0]

        return self._unknown(query_code)

    def _unknown(self, query_code: str) -> Dict:
        return {
            "code": query_code,
            "name": "Unknown Course",
            "meetings": [],
            "rooms": [],
            "instructors": [],
            "description": "",
        }
