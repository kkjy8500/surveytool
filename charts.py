import zipfile
from io import BytesIO
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import font_manager, rcParams
from matplotlib.patches import Patch

from chart_theme import (
    DEFAULT_NEUTRAL_LABELS,
    STANDARD_THEME,
    get_preset_defaults,
    recommended_chart_label,
    resolve_colors,
    safe_hex,
    should_use_pie,
    wrap_label,
    estimate_bottom_margin,
    estimate_left_margin,
    compute_scale_mean,
)
from config import CHART_TYPE_OPTIONS
from display_format import format_chart_count_text, format_pct, format_stat
from metadata import get_var_label
from tabulation import compute_multiresponse_distribution, compute_single_distribution
from utils import safe_filename


SCALE_COLOR_PALETTES = {
    k: v for k, v in {
        "기본 블루": None,
        "청록": None,
        "그레이": None,
    }.items()
}

UNIFIED_PRIMARY_COLOR = "#1D4ED8"
UNIFIED_MUTED_COLOR = "#CBD5E1"
UNIFIED_SECONDARY_COLOR = "#94A3B8"
MEAN_LINE_COLOR = "#E05C5C"


def _auto_figure_height(chart_type: str, n_labels: int, user_height: float) -> float:
    """카테고리 수에 따른 적정 차트 높이 자동 계산."""
    if chart_type == "barh":
        auto = max(3.6, n_labels * 0.65 + 1.8)
        return max(user_height, auto)
    if chart_type == "barv":
        auto = max(4.2, min(6.8, 4.2 + max(0, n_labels - 5) * 0.22))
        return max(user_height, auto)
    if chart_type == "pie":
        return max(user_height, 5.0)
    return max(user_height, 4.5)


def render_radar_chart(labels, values, title):
    labels = [str(x) for x in labels]
    values = [float(x) if x is not None else 0.0 for x in values]
    if labels and values:
        labels = labels + labels[:1]
        values = values + values[:1]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill="toself", line=dict(color="#1D4ED8", width=2.5), name=title))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title=title, template="plotly_white")
    return fig


def register_local_korean_fonts():
    search_dirs = [Path(__file__).resolve().parent / "fonts", Path.cwd() / "fonts", Path("/mnt/data/fonts")]
    for font_dir in search_dirs:
        if not font_dir.exists():
            continue
        for pattern in ("*.ttf", "*.otf"):
            for path in font_dir.glob(pattern):
                try:
                    font_manager.fontManager.addfont(str(path))
                except Exception:
                    continue


def try_set_korean_font(font_name=None):
    register_local_korean_fonts()
    candidates = ["KoPubDotum", "KoPub돋움체 Medium", "Pretendard", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans KR", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    if font_name and font_name in available:
        rcParams["font.family"] = font_name
        rcParams["axes.unicode_minus"] = False
        return font_name
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return name
    rcParams["axes.unicode_minus"] = False
    return None


FONT_SET = try_set_korean_font()


def get_available_korean_fonts():
    register_local_korean_fonts()
    candidates = ["KoPubDotum", "KoPub돋움체 Medium", "Pretendard", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans KR", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in candidates if f in available]




def get_chart_preset_names():
    from chart_theme import CHART_PRESETS
    return list(CHART_PRESETS.keys())

def get_scale_palette_options():
    return ["기본 블루", "청록", "그레이"]


def compute_single_rows(df, var_name: str, metadata: dict, empty_include: bool, missing_codes: list, pct_base: str, missing_rules_by_var: dict | None = None, weight_col: str | None = None):
    summary = compute_single_distribution(df=df, var_name=var_name, metadata=metadata, empty_include=empty_include, missing_codes=missing_codes, pct_base=pct_base, missing_rules_by_var=missing_rules_by_var, weight_col=weight_col)
    return {"base": summary["base"], "rows": summary["rows"]}


def compute_multiresponse_rows(df, mr_vars: list, metadata: dict, missing_codes: list, pct_base: str, mr_selected_mode: str, mr_selected_codes: list, missing_rules_by_var: dict | None = None, weight_col: str | None = None):
    summary = compute_multiresponse_distribution(df=df, mr_vars=mr_vars, metadata=metadata, missing_codes=missing_codes, pct_base=pct_base, mr_selected_mode=mr_selected_mode, mr_selected_codes=mr_selected_codes, missing_rules_by_var=missing_rules_by_var, weight_col=weight_col)
    return {"base": summary["base"], "rows": summary["rows"]}


def build_chart_result_for_var(df, var_name: str, metadata: dict, empty_include: bool, missing_codes: list, pct_base: str, result_type: str = "single", missing_rules_by_var: dict | None = None, weight_col: str | None = None):
    computed = compute_single_rows(df=df, var_name=var_name, metadata=metadata, empty_include=empty_include, missing_codes=missing_codes, pct_base=pct_base, missing_rules_by_var=missing_rules_by_var, weight_col=weight_col)
    return {"question_id": str(var_name), "question_text": get_var_label(var_name, metadata), "question_type": result_type, "base": computed["base"], "rows": computed["rows"]}


def build_multiresponse_chart_result(df, mr_group_name: str, mr_vars: list, metadata: dict, missing_codes: list, pct_base: str, mr_selected_mode: str, mr_selected_codes: list, missing_rules_by_var: dict | None = None, weight_col: str | None = None):
    computed = compute_multiresponse_rows(df=df, mr_vars=mr_vars, metadata=metadata, missing_codes=missing_codes, pct_base=pct_base, mr_selected_mode=mr_selected_mode, mr_selected_codes=mr_selected_codes, missing_rules_by_var=missing_rules_by_var, weight_col=weight_col)
    return {"question_id": str(mr_group_name), "question_text": str(mr_group_name), "question_type": "multiresponse", "base": computed["base"], "rows": computed["rows"]}


def build_chart_results(df, dep_vars: list, profile_vars: list, metadata: dict, empty_include: bool, missing_codes: list, pct_base: str, mr_group_map: dict, selected_mr_groups: list, mr_selected_mode: str = "값 있으면 선택", mr_selected_codes: list | None = None, mr_group_definitions: dict | None = None, question_type_map: dict | None = None, missing_rules_by_var: dict | None = None, weight_col: str | None = None):
    results = []
    mr_selected_codes = mr_selected_codes or [1]
    mr_group_definitions = mr_group_definitions or {}
    question_type_map = question_type_map or {}
    all_mr_cols = set()
    for cols in mr_group_map.values():
        all_mr_cols.update(cols)
    final_dep_vars = [v for v in dep_vars if v not in all_mr_cols]
    for var in final_dep_vars:
        qtype = {"척도형": "scale", "순위형": "rank"}.get(question_type_map.get(var, "범주형"), "single")
        results.append(build_chart_result_for_var(df=df, var_name=var, metadata=metadata, empty_include=empty_include, missing_codes=missing_codes, pct_base=pct_base, result_type=qtype, missing_rules_by_var=missing_rules_by_var, weight_col=weight_col))
    for mr_group_name in selected_mr_groups:
        mr_vars = mr_group_map.get(mr_group_name, [])
        if not mr_vars:
            continue
        mr_def = mr_group_definitions.get(mr_group_name, {}) or {}
        results.append(build_multiresponse_chart_result(df=df, mr_group_name=mr_group_name, mr_vars=mr_vars, metadata=metadata, missing_codes=missing_codes, pct_base=mr_def.get("pct_base", pct_base) or pct_base, mr_selected_mode=mr_def.get("selected_mode", mr_selected_mode), mr_selected_codes=mr_def.get("selected_codes", mr_selected_codes), missing_rules_by_var=missing_rules_by_var, weight_col=weight_col))
    for var in profile_vars:
        results.append(build_chart_result_for_var(df=df, var_name=var, metadata=metadata, empty_include=empty_include, missing_codes=missing_codes, pct_base=pct_base, result_type="profile", missing_rules_by_var=missing_rules_by_var, weight_col=weight_col))
    deduped = []
    seen = set()
    for item in results:
        qid = str(item.get("question_id", "")).strip()
        if qid and qid not in seen:
            seen.add(qid)
            deduped.append(item)
    return deduped


def validate_chart_renderable(result: dict, max_categories: int = 10) -> dict:
    rows = [r for r in result.get("rows", []) if r.get("pct") is not None]
    n_categories = len(rows)
    if n_categories < 2:
        return {"ok": False, "reason": f"범주 수({n_categories}개)가 너무 적습니다 (최소 2개 필요)."}
    if n_categories > max_categories:
        return {"ok": False, "reason": f"범주 수({n_categories}개)가 최대 허용치({max_categories}개)를 초과했습니다."}
    return {"ok": True, "reason": ""}


def should_render_chart(result: dict, max_categories: int = 10) -> bool:
    return validate_chart_renderable(result, max_categories)["ok"]


def _prepare_chart_rows(result: dict):
    valid_rows = [r for r in result.get("rows", []) if r.get("pct") is not None]
    raw_labels = [str(r.get("label", "")) for r in valid_rows]
    pcts = [float(r.get("pct") or 0) for r in valid_rows]
    ns = [int(r.get("n") or 0) for r in valid_rows]
    return valid_rows, raw_labels, pcts, ns


def _group_small_slices(raw_labels: list[str], pcts: list[float], ns: list[int], threshold: float = 5.0):
    rows = []
    other_pct = 0.0
    other_n = 0
    for label, pct, n in zip(raw_labels, pcts, ns):
        if pct < threshold:
            other_pct += pct
            other_n += n
        else:
            rows.append((label, pct, n))
    if other_pct > 0:
        rows.append(("기타", other_pct, other_n))
    if not rows:
        return raw_labels, pcts, ns
    labels2, pcts2, ns2 = zip(*rows)
    return list(labels2), list(pcts2), list(ns2)


def _apply_common_figure_style(fig, ax):
    fig.patch.set_facecolor(STANDARD_THEME["background_color"])
    ax.set_facecolor(STANDARD_THEME["background_color"])
    for spine_name in ["top", "right", "left"]:
        if spine_name in ax.spines:
            ax.spines[spine_name].set_visible(False)
    if "bottom" in ax.spines:
        ax.spines["bottom"].set_color(STANDARD_THEME["axis_color"])
        ax.spines["bottom"].set_linewidth(1.0)



def _resolve_chart_title(result: dict) -> str:
    """그래프 제목은 최종 표시명(question_text)을 우선 사용한다.
    표시명이 없을 때만 question_id로 폴백한다.
    """
    title = str(result.get("question_text") or "").strip()
    if not title:
        title = str(result.get("question_id") or "").strip()
    return title or "문항 결과"


def _wrap_chart_title(title: str, fig_width: float = 8.0, max_chars: int = 72) -> str:
    """제목을 그래프 폭에 맞춰 1~2줄로 줄바꿈하고, 지나치게 길면 말줄임한다."""
    text = str(title or "").strip()
    if not text:
        return ""

    # 한국어는 공백 기준 wrap이 잘 안 되므로 글자 수 기준으로 먼저 제한한다.
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"

    width = max(28, min(46, int(float(fig_width or 8.0) * 5.2)))
    if len(text) <= width:
        return text

    lines = []
    remaining = text
    while remaining and len(lines) < 2:
        if len(remaining) <= width:
            lines.append(remaining)
            remaining = ""
        else:
            cut = remaining.rfind(" ", 0, width + 1)
            if cut < max(10, int(width * 0.55)):
                cut = width
            lines.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()

    if remaining and lines:
        lines[-1] = lines[-1].rstrip(" …") + "…"
    return "\n".join(lines)


def _add_standard_header(fig, title: str, base: int, show_legend: bool, legend_handles=None, legend_labels=None, active_font=None):
    """
    차트 제목과 N을 분리해 배치한다.
    - 제목은 상단 중앙에 두되, 긴 문항명은 자동 줄바꿈한다.
    - N은 제목 아래/그래프 영역 위쪽의 오른쪽 끝에 고정한다.
    """
    fig_width = float(fig.get_size_inches()[0]) if fig is not None else 8.0
    wrapped_title = _wrap_chart_title(title, fig_width=fig_width)
    title_line_count = wrapped_title.count("\n") + 1 if wrapped_title else 1

    title_y = STANDARD_THEME.get("title_y", 0.965)
    n_y = max(0.835, title_y - (0.038 * title_line_count) - 0.018)

    fig.text(
        0.5,
        title_y,
        wrapped_title,
        ha="center",
        va="top",
        fontsize=STANDARD_THEME["font_size_title"],
        fontweight="bold",
        color=STANDARD_THEME["title_color"],
        fontfamily=active_font,
        linespacing=1.18,
    )
    fig.text(
        0.985,
        n_y,
        f"(N={int(round(float(base))):,})" if pd.notna(base) else "(N=-)",
        ha="right",
        va="top",
        fontsize=STANDARD_THEME["font_size_note"],
        color=STANDARD_THEME["value_text_color"],
        fontfamily=active_font,
    )
    if show_legend and legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, frameon=False, bbox_to_anchor=(0.985, max(0.78, n_y - 0.02)), loc="upper right", prop={"family": active_font, "size": 8.5} if active_font else {"size": 8.5})


def render_chart_png(result: dict, chart_type: str, chart_color: str, font_name: str, fig_width: float, fig_height: float, show_legend: bool, show_n_label: bool = False, color_palette: str | None = None, neutral_labels_text: str | None = None, category_color_mode: str = "단일색", pie_threshold: float = 5.0):
    active_font = try_set_korean_font(font_name) or font_name or FONT_SET
    _, raw_labels, pcts, ns = _prepare_chart_rows(result)
    question_type = str(result.get("question_type", "single"))
    longest_label = max((len(x) for x in raw_labels), default=0)

    if chart_type == "pie" and not should_use_pie(question_type, len(raw_labels)):
        chart_type = "barh" if longest_label >= 16 or len(raw_labels) >= 6 else "barv"
    if question_type in {"scale", "rank"} and chart_type == "pie":
        chart_type = "barv"

    label_wrap_width = 9 if chart_type == "barv" and (longest_label >= 14 or len(raw_labels) >= 6) else 14
    wrapped_labels = [wrap_label(x, width=label_wrap_width) for x in raw_labels]

    colors = resolve_colors(raw_labels, pcts, question_type=question_type, base_color=chart_color, color_palette=color_palette, neutral_labels_text=neutral_labels_text, category_color_mode=category_color_mode)
    title = _resolve_chart_title(result)

    dynamic_width = max(fig_width, 8.0)
    if chart_type == "barv":
        # 사용자가 선택한 세로 막대그래프는 유지하되, 범주 수와 보기 길이에 따라 폭을 넓힌다.
        width_by_count = 7.6 + len(raw_labels) * 0.32
        width_by_label = 8.0 + max(0, longest_label - 10) * 0.08
        dynamic_width = max(dynamic_width, min(15.5, max(width_by_count, width_by_label)))
    dynamic_height = _auto_figure_height(chart_type, len(raw_labels), fig_height)
    if chart_type == "barv":
        max_label_lines = max((str(x).count("\n") + 1 for x in wrapped_labels), default=1)
        dynamic_height = max(dynamic_height, fig_height + max(0, max_label_lines - 2) * 0.25)

    if chart_type == "radar":
        fig, ax = plt.subplots(figsize=(dynamic_width, dynamic_height), subplot_kw={"polar": True})
    else:
        fig, ax = plt.subplots(figsize=(dynamic_width, dynamic_height))
    _apply_common_figure_style(fig, ax)

    legend_handles = None
    legend_labels = None

    if chart_type == "barv":
        x = np.arange(len(raw_labels))
        rotation = 0
        tick_font_size = max(7.8, STANDARD_THEME["font_size_tick"] - max(0, len(raw_labels) - 5) * 0.25)
        bars = ax.bar(x, pcts, color=colors, width=0.58, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_labels, fontsize=tick_font_size, color=STANDARD_THEME["axis_text_color"], fontfamily=active_font)
        ax.tick_params(axis="x", length=0, pad=12)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ymax = max(100 if question_type == "scale" else 0, max(pcts) * 1.20 if pcts else 100)
        ax.set_ylim(0, ymax)
        for bar, pct, n in zip(bars, pcts, ns):
            txt = f"{format_pct(pct)}%"
            if show_n_label:
                txt += f"\n{format_chart_count_text(n)}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.03, txt, ha="center", va="bottom", fontsize=STANDARD_THEME["font_size_value"], fontfamily=active_font, color=STANDARD_THEME["value_text_color"])
        if show_legend and len(set(raw_labels)) == len(raw_labels) and len(raw_labels) <= 6:
            legend_handles = [Patch(facecolor=colors[i], edgecolor="none") for i in range(len(raw_labels))]
            legend_labels = raw_labels
        # 척도형 평균선
        if question_type == "scale":
            mean_val = compute_scale_mean(raw_labels, pcts)
            if mean_val is not None:
                # x 위치: 레이블이 숫자이면 그 값 기준, 아니면 인덱스 기준
                try:
                    codes = [float(l) for l in raw_labels]
                    step = codes[1] - codes[0] if len(codes) > 1 else 1
                    mean_x = (mean_val - codes[0]) / step
                except Exception:
                    mean_x = sum(i * p / 100 for i, p in enumerate(pcts))
                ax.axvline(x=mean_x, color=MEAN_LINE_COLOR, linewidth=2.0, linestyle="--", alpha=0.85, zorder=5)
                ax.text(
                    mean_x + 0.08, ymax * 0.90,
                    f"평균 {format_stat(mean_val)}",
                    color=MEAN_LINE_COLOR, fontsize=9.5, fontweight="bold",
                    fontfamily=active_font, va="top",
                )
        bottom_margin = min(max(estimate_bottom_margin(wrapped_labels), 0.22), 0.54)
        fig.subplots_adjust(top=0.76, bottom=bottom_margin, left=0.07, right=0.985)

    elif chart_type == "barh":
        y = np.arange(len(raw_labels))
        bars = ax.barh(y, pcts, color=colors, height=0.58, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(wrapped_labels, fontsize=STANDARD_THEME["font_size_tick"], color=STANDARD_THEME["axis_text_color"], fontfamily=active_font)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.invert_yaxis()
        xmax = max(100 if question_type == "scale" else 0, max(pcts) * 1.22 if pcts else 100)
        ax.set_xlim(0, xmax)
        for bar, pct, n in zip(bars, pcts, ns):
            txt = f"{format_pct(pct)}%"
            if show_n_label:
                txt += f" {format_chart_count_text(n)}"
            ax.text(bar.get_width() + xmax * 0.01, bar.get_y() + bar.get_height() / 2, txt, ha="left", va="center", fontsize=STANDARD_THEME["font_size_value"], fontfamily=active_font, color=STANDARD_THEME["value_text_color"])
        if show_legend and len(set(raw_labels)) == len(raw_labels) and len(raw_labels) <= 6:
            legend_handles = [Patch(facecolor=colors[i], edgecolor="none") for i in range(len(raw_labels))]
            legend_labels = raw_labels
        # 척도형 평균선 (가로 막대에서는 y축)
        if question_type == "scale":
            mean_val = compute_scale_mean(raw_labels, pcts)
            if mean_val is not None:
                try:
                    codes = [float(l) for l in raw_labels]
                    step = codes[1] - codes[0] if len(codes) > 1 else 1
                    mean_y = (mean_val - codes[0]) / step
                except Exception:
                    mean_y = sum(i * p / 100 for i, p in enumerate(pcts))
                ax.axhline(y=mean_y, color=MEAN_LINE_COLOR, linewidth=2.0, linestyle="--", alpha=0.85, zorder=5)
                ax.text(
                    xmax * 0.97, mean_y - 0.22,
                    f"평균 {format_stat(mean_val)}",
                    color=MEAN_LINE_COLOR, fontsize=9.5, fontweight="bold",
                    fontfamily=active_font, ha="right", va="top",
                )
        fig.subplots_adjust(top=0.76, bottom=0.10, left=estimate_left_margin(wrapped_labels), right=0.97)

    elif chart_type == "pie":
        raw_labels, pcts, ns = _group_small_slices(raw_labels, pcts, ns, threshold=float(pie_threshold or 5.0))
        colors = resolve_colors(raw_labels, pcts, question_type="single", base_color=chart_color, color_palette=color_palette, neutral_labels_text=neutral_labels_text, category_color_mode="단일색")
        wedges, texts, autotexts = ax.pie(pcts, labels=None, autopct=lambda pct: f"{format_pct(pct)}%", startangle=90, counterclock=False, colors=colors, wedgeprops={"width": 0.92, "edgecolor": "white", "linewidth": 1.1}, textprops={"fontsize": 9, "fontfamily": active_font, "color": STANDARD_THEME["value_text_color"]})
        if show_legend:
            legend_handles = wedges
            legend_labels = raw_labels
        fig.subplots_adjust(top=0.78, bottom=0.06, left=0.06, right=0.78 if show_legend else 0.96)

    elif chart_type == "line":
        x = np.arange(len(raw_labels))
        ax.plot(x, pcts, marker="o", color=safe_hex(chart_color), linewidth=2.4, markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_labels, fontsize=STANDARD_THEME["font_size_tick"], color=STANDARD_THEME["axis_text_color"], fontfamily=active_font)
        ax.tick_params(axis="x", length=0, pad=12)
        ax.grid(False)
        ymax = max(max(pcts) * 1.2 if pcts else 100, 100)
        ax.set_ylim(0, ymax)
        for i, pct in enumerate(pcts):
            ax.text(i, pct + ymax * 0.03, f"{format_pct(pct)}%", ha="center", va="bottom", fontsize=STANDARD_THEME["font_size_value"], fontfamily=active_font)
        fig.subplots_adjust(top=0.78, bottom=estimate_bottom_margin(wrapped_labels), left=0.07, right=0.98)

    elif chart_type == "radar":
        n = len(raw_labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values = pcts + pcts[:1]
        angles = angles + angles[:1]
        ax.plot(angles, values, color=safe_hex(chart_color), linewidth=2)
        ax.fill(angles, values, color=safe_hex(chart_color), alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(wrapped_labels, fontsize=9, fontfamily=active_font)
        ax.set_ylim(0, max(100, max(pcts) * 1.1 if pcts else 100))
        fig.subplots_adjust(top=0.78, bottom=0.10, left=0.08, right=0.92)

    else:
        plt.close(fig)
        raise ValueError(f"지원하지 않는 차트 유형입니다: {chart_type}")

    _add_standard_header(fig, title=title, base=result.get("base", 0), show_legend=show_legend, legend_handles=legend_handles, legend_labels=legend_labels, active_font=active_font)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", facecolor=STANDARD_THEME["background_color"])
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def chart_result_to_dataframe(result: dict):
    rows = result.get("rows", [])
    df_table = pd.DataFrame(rows)
    if not df_table.empty:
        if "pct" in df_table.columns:
            df_table = df_table.sort_values("pct", ascending=False).reset_index(drop=True)
        df_table = df_table.rename(columns={"label": "보기", "n": "N", "pct": "%"})
    return df_table


def build_chart_settings_df(chart_results, default_chart_type=None, default_color=None, default_font=None, default_width=None, default_height=None, default_legend=None, default_palette: str = "기본 블루"):
    rows = []
    for result in chart_results:
        _, raw_labels, _, _ = _prepare_chart_rows(result)
        longest_label = max((len(x) for x in raw_labels), default=0)
        recommended = recommended_chart_label(str(result.get("question_type", "single")), len(raw_labels), longest_label)
        question_type = str(result.get("question_type", "single"))
        chart_type_key = CHART_TYPE_OPTIONS.get(default_chart_type or recommended or "세로 막대그래프", "barv")
        auto_height = _auto_figure_height(chart_type_key, len(raw_labels), float(default_height or 4.8))
        rows.append({
            "question_id": result["question_id"],
            "include": question_type != "profile",
            "question_text": result["question_text"],
            "recommended_chart": recommended,
            "chart_type": default_chart_type or recommended or "세로 막대그래프",
            "color": default_color or UNIFIED_PRIMARY_COLOR,
            "font": default_font or "NanumGothic",
            "width": float(default_width or 8.2),
            "height": auto_height,
            "show_legend": bool(default_legend if default_legend is not None else False),
            "color_palette": default_palette,
            "category_color_mode": (
                "척도형 자동색상" if question_type == "scale"
                else "단일색" if question_type == "rank"
                else "값 비례 그라데이션"
            ),
            "neutral_labels": DEFAULT_NEUTRAL_LABELS,
            "pie_threshold": 5.0,
        })
    return pd.DataFrame(rows)


def chart_settings_df_to_dict(df_settings):
    settings = {}
    for _, row in df_settings.iterrows():
        if "include" in row and not bool(row["include"]):
            continue
        settings[str(row["question_id"])] = {
            "chart_type": row["chart_type"],
            "color": row.get("color", UNIFIED_PRIMARY_COLOR) or UNIFIED_PRIMARY_COLOR,
            "font": row["font"],
            "width": float(row["width"]),
            "height": float(row["height"]),
            "show_legend": bool(row["show_legend"]),
            "color_palette": row.get("color_palette", row.get("scale_palette", "기본 블루")),
            "category_color_mode": row.get("category_color_mode", "단일색"),
            "neutral_labels": row.get("neutral_labels", DEFAULT_NEUTRAL_LABELS),
            "pie_threshold": float(row.get("pie_threshold", 5.0)),
        }
    return settings


def build_all_charts(chart_results: list, chart_settings: dict, show_n_label: bool = False, max_categories: int = 10):
    chart_files = {}
    for result in chart_results:
        validation = validate_chart_renderable(result, max_categories=max_categories)
        if not validation["ok"]:
            print(f"[CHART SKIPPED] 문항 {result.get('question_id')} 그래프 생성 건너뜀: {validation['reason']}")
            continue
        cfg = chart_settings.get(result["question_id"], {})
        rows = [r for r in result.get("rows", []) if r.get("pct") is not None]
        if not rows:
            continue
        chart_label = cfg.get("chart_type", "세로 막대그래프")
        chart_type = CHART_TYPE_OPTIONS.get(chart_label, "barv")
        if chart_type == "pie" and not should_use_pie(str(result.get("question_type", "single")), len(rows)):
            longest_label = max((len(str(r.get("label", ""))) for r in rows), default=0)
            chart_type = "barh" if longest_label >= 16 or len(rows) >= 6 else "barv"
        
        # 파일명 충돌 방지 로직
        base_filename = f"{safe_filename(result['question_id'])}.png"
        filename = base_filename
        counter = 2
        while filename in chart_files:
            filename = f"{safe_filename(result['question_id'])}_{counter}.png"
            counter += 1

        chart_files[filename] = render_chart_png(
            result=result,
            chart_type=chart_type,
            chart_color=cfg.get("color", "#406A9F"),
            font_name=cfg.get("font", FONT_SET),
            fig_width=float(cfg.get("width", 8.2)),
            fig_height=float(cfg.get("height", 4.8)),
            show_legend=bool(cfg.get("show_legend", False)),
            show_n_label=show_n_label,
            color_palette=cfg.get("color_palette", cfg.get("scale_palette", "기본 블루")),
            neutral_labels_text=cfg.get("neutral_labels", DEFAULT_NEUTRAL_LABELS),
            category_color_mode=cfg.get("category_color_mode", "단일색"),
            pie_threshold=float(cfg.get("pie_threshold", 5.0)),
        )
    return chart_files


def build_zip_bytes(file_dict: dict) -> bytes:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, file_bytes in file_dict.items():
            zf.writestr(filename, file_bytes)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
