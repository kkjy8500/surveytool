from __future__ import annotations

import re
import textwrap
from matplotlib.colors import to_hex, to_rgb

DEFAULT_NEUTRAL_LABELS = "보통, 중립, 해당 없음, 모름"

STANDARD_THEME = {
    "background_color": "#FFFFFF",
    "axis_color": "#D9D9D9",
    "axis_text_color": "#334155",
    "title_color": "#0F172A",
    "value_text_color": "#0F172A",
    "font_size_title": 15,
    "font_size_tick": 10,
    "font_size_value": 10,
    "font_size_note": 9,
    "title_y": 0.965,
}

CHART_PRESETS = {
    "기본": {"chart_type": "세로 막대그래프", "color": "#1D4ED8"},
    "가로형": {"chart_type": "가로 막대그래프", "color": "#1D4ED8"},
    "보고서형": {"chart_type": "세로 막대그래프", "color": "#334155"},
}


def get_preset_defaults(name: str) -> dict:
    return CHART_PRESETS.get(name, CHART_PRESETS["기본"]).copy()


def safe_hex(value: str, fallback: str = "#1D4ED8") -> str:
    try:
        return to_hex(value)
    except Exception:
        return fallback


def wrap_label(text: str, width: int = 14) -> str:
    parts = textwrap.wrap(str(text), width=max(4, width), break_long_words=True)
    return "\n".join(parts[:2]) if parts else ""


def recommended_chart_label(question_type: str, n_labels: int, longest_label: int) -> str:
    if question_type in {"scale", "rank"}:
        return "세로 막대그래프"
    if n_labels >= 6 or longest_label >= 16:
        return "가로 막대그래프"
    return "세로 막대그래프"


def should_use_pie(question_type: str, n_labels: int) -> bool:
    return question_type in {"single", "profile"} and 2 <= n_labels <= 5


def _mix(color: str, target: str, ratio: float) -> str:
    c1, c2 = to_rgb(safe_hex(color)), to_rgb(target)
    return to_hex(tuple((1-ratio)*a + ratio*b for a, b in zip(c1, c2)))


def resolve_colors(labels, values, question_type="single", base_color="#1D4ED8", color_palette=None, neutral_labels_text=None, category_color_mode="단일색"):
    base = safe_hex(base_color)
    n = len(labels)
    if category_color_mode == "값 비례 그라데이션" and n:
        max_v = max(values) or 1
        return [_mix(base, "#FFFFFF", 0.65 * (1 - float(v)/max_v)) for v in values]
    if category_color_mode == "척도형 자동색상" and n:
        return [_mix(base, "#FFFFFF", 0.72 * (i / max(1, n-1))) for i in range(n)]
    return [base] * n


def estimate_bottom_margin(labels) -> float:
    lines = max((str(x).count("\n") + 1 for x in labels), default=1)
    return min(0.18 + 0.06 * lines, 0.45)


def estimate_left_margin(labels) -> float:
    longest = max((len(str(x).replace("\n", "")) for x in labels), default=0)
    return min(0.18 + longest * 0.008, 0.46)


def compute_scale_mean(labels, pcts):
    nums = []
    for label in labels:
        match = re.search(r"-?\d+(?:\.\d+)?", str(label))
        if not match:
            return None
        nums.append(float(match.group()))
    total = sum(float(p) for p in pcts)
    return sum(v * float(p) for v, p in zip(nums, pcts)) / total if total else None
