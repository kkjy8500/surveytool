from datetime import datetime
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re

from metadata import get_var_label, get_value_label
from charts import compute_single_rows, compute_multiresponse_rows, render_radar_chart
from tabulation import compute_single_distribution, get_categories
from display_format import format_pct, format_stat
from font_utils import browser_font_face_css, browser_primary_font_name, local_font_assets
from utils import (
    build_zip_bytes_from_mapping,
    dict_to_json_bytes,
    safe_filename,
    safe_numeric,
    slugify_text,
    sort_var_names,
)

MEAN_LINE_COLOR = "#E05C5C"


def _auto_plotly_height(chart_type: str, n_labels: int) -> int:
    """카테고리 수 기반 Plotly 차트 높이 자동 계산."""
    if chart_type == "barh":
        return max(360, 90 + n_labels * 48)
    if chart_type in {"bar"}:
        return max(400, 380 + max(0, n_labels - 5) * 15)
    if chart_type == "donut":
        return 480
    return 460


def _apply_drilldown_filter(df: pd.DataFrame, filter_state: dict, metadata: dict):
    """필터 상태에 따라 df를 필터링. (filtered_df, n) 반환."""
    if not filter_state:
        return df, None
    var = filter_state.get("question_id")
    label = filter_state.get("label")
    if not var or not label or var not in df.columns:
        return df, None

    value_labels = metadata.get(var, {}).get("value_labels", {}) or {}
    display_value_labels = metadata.get(var, {}).get("display_value_labels", {}) or {}

    code = None
    for lbl_dict in [display_value_labels, value_labels]:
        for k, v in lbl_dict.items():
            if str(v).strip() == str(label).strip():
                code = k
                break
        if code is not None:
            break

    if code is None:
        return df, None

    try:
        mask = safe_numeric(df[var]) == float(code)
    except Exception:
        mask = df[var].astype(str) == str(code)

    filtered = df[mask].copy()
    return filtered, int(len(filtered))


DASHBOARD_CHART_OPTIONS = {
    "가로 막대": "barh",
    "세로 막대": "bar",
    "도넛": "donut",
    "레이더": "radar",
}


DEFAULT_DASHBOARD_CONFIG_COLUMNS = [
    "include",
    "order",
    "section",
    "question_id",
    "display_label",
    "description",
    "question_type",
    "default_banner",
    "default_chart",
]


BASE_BANNER_OPTIONS = {
    "전체": "__total__",
}


PLOTLY_COLOR_SEQUENCE = [
    "#1D4ED8",
    "#94A3B8",
    "#CBD5E1",
    "#0F172A",
    "#64748B",
    "#DBEAFE",
]


def _dashboard_type_priority(question_type: str) -> int:
    return {
        "single": 0,
        "scale": 1,
        "multi": 2,
        "rank": 3,
    }.get(str(question_type), 9)


def _get_dashboard_recommended_ids(ordered_ids: list[str], selected_mr_groups: list, question_type_map: dict | None = None, max_items: int = 6) -> set[str]:
    question_type_map = question_type_map or {}
    scored = []
    for idx, qid in enumerate(ordered_ids):
        if qid in (selected_mr_groups or []):
            qtype = "multi"
        else:
            qtype = {"척도형": "scale", "순위형": "rank"}.get(question_type_map.get(qid, "범주형"), "single")
        scored.append((_dashboard_type_priority(qtype), idx, qid))
    scored.sort(key=lambda x: (x[0], x[1]))
    return {qid for _, _, qid in scored[:max_items]}


def _get_default_dashboard_chart(question_type: str) -> str:
    if question_type == "multi":
        return "가로 막대"
    if question_type in {"scale", "rank"}:
        return "세로 막대"
    return "가로 막대"


def get_dashboard_banner_options(profile_vars: list, metadata: dict) -> dict:
    options = BASE_BANNER_OPTIONS.copy()
    for var in sort_var_names(profile_vars):
        options[var] = var
    return options


def build_dashboard_config_df(dep_vars: list, selected_mr_groups: list, metadata: dict, question_type_map: dict | None = None, section_map: dict | None = None) -> pd.DataFrame:
    rows = []
    question_type_map = question_type_map or {}
    section_map = section_map or {}
    ordered_ids = sort_var_names(list(dict.fromkeys(list(dep_vars or []) + list(selected_mr_groups or []))))

    for idx, qid in enumerate(ordered_ids, start=1):
        if qid in (selected_mr_groups or []):
            rows.append({
                "include": True,
                "order": idx,
                "section": section_map.get(qid, "미분류"),
                "question_id": qid,
                "display_label": qid,
                "description": "",
                "question_type": "multi",
                "default_banner": "전체",
                "default_chart": "세로 막대",
            })
            continue

        qtype = {"척도형": "scale", "순위형": "rank"}.get(question_type_map.get(qid, "범주형"), "single")
        default_chart = "세로 막대"
        rows.append({
            "include": True,
            "order": idx,
            "section": section_map.get(qid, "미분류"),
            "question_id": qid,
            "display_label": get_var_label(qid, metadata),
            "description": "",
            "question_type": qtype,
            "default_banner": "전체",
            "default_chart": default_chart,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=DEFAULT_DASHBOARD_CONFIG_COLUMNS)
    return df[DEFAULT_DASHBOARD_CONFIG_COLUMNS].copy()


def normalize_dashboard_config_df(
    df: pd.DataFrame,
    dep_vars: list,
    selected_mr_groups: list,
    metadata: dict,
    question_type_map: dict | None = None,
    section_map: dict | None = None,
) -> pd.DataFrame:
    fresh = build_dashboard_config_df(dep_vars, selected_mr_groups, metadata, question_type_map=question_type_map, section_map=section_map)
    if df is None or df.empty:
        return fresh

    current = df.copy()
    for col in DEFAULT_DASHBOARD_CONFIG_COLUMNS:
        if col not in current.columns:
            current[col] = ""

    current["question_id"] = current["question_id"].astype(str)
    fresh["question_id"] = fresh["question_id"].astype(str)

    merged = fresh.set_index("question_id")
    existing = current.set_index("question_id")

    for qid in merged.index:
        if qid in existing.index:
            for col in DEFAULT_DASHBOARD_CONFIG_COLUMNS:
                if col == "question_id":
                    continue
                val = existing.at[qid, col]
                if pd.notna(val) and str(val) != "":
                    merged.at[qid, col] = val

    out = merged.reset_index()
    out = out[DEFAULT_DASHBOARD_CONFIG_COLUMNS]
    out["include"] = out["include"].fillna(False).astype(bool)
    out["order"] = pd.to_numeric(out["order"], errors="coerce").fillna(999).astype(int)
    out["section"] = out["section"].fillna("미분류")
    out["display_label"] = out["display_label"].fillna(out["question_id"])
    out["description"] = out["description"].fillna("")
    out["question_type"] = out["question_type"].fillna("single")
    out["default_banner"] = out["default_banner"].fillna("전체")
    out["default_chart"] = out["default_chart"].fillna("가로 막대")
    return out.sort_values(["include", "order", "question_id"], ascending=[False, True, True]).reset_index(drop=True)


def get_selected_dashboard_items(config_df: pd.DataFrame) -> pd.DataFrame:
    if config_df is None or config_df.empty:
        return pd.DataFrame(columns=DEFAULT_DASHBOARD_CONFIG_COLUMNS)
    out = config_df.copy()
    out = out[out["include"] == True].copy()
    if out.empty:
        return out
    return out.sort_values(["order", "question_id"]).reset_index(drop=True)


def _filter_df_for_banner(df: pd.DataFrame, banner_var: str | None, banner_value):
    if banner_var is None:
        return df.copy()
    if banner_var not in df.columns:
        return df.iloc[0:0].copy()
    try:
        mask = safe_numeric(df[banner_var]) == float(banner_value)
    except Exception:
        mask = df[banner_var].astype(str) == str(banner_value)
    return df[mask].copy()


def _build_bannered_single_result(
    df: pd.DataFrame,
    var: str,
    banner_var: str,
    metadata: dict,
    empty_include: bool,
    missing_codes: list,
    pct_base: str,
    result_type: str = "single",
    missing_rules_by_var: dict | None = None,
):
    categories = get_categories(df[banner_var], banner_var, metadata, empty_include, missing_codes, missing_rules_by_var=missing_rules_by_var)
    banner_rows = []

    for cat in categories:
        sub_df = _filter_df_for_banner(df, banner_var, cat)
        if sub_df.empty:
            continue

        computed = compute_single_rows(
            df=sub_df,
            var_name=var,
            metadata=metadata,
            empty_include=empty_include,
            missing_codes=missing_codes,
            pct_base=pct_base,
            missing_rules_by_var=missing_rules_by_var,
        )

        banner_rows.append({
            "banner_label": get_value_label(banner_var, cat, metadata),
            "base": computed["base"],
            "rows": computed["rows"],
        })

    return {
        "question_id": str(var),
        "question_text": get_var_label(var, metadata),
        "question_type": result_type,
        "banner_var": banner_var,
        "banner_var_label": get_var_label(banner_var, metadata),
        "banner_rows": banner_rows,
        "rows": [],
        "base": None,
    }


def _build_bannered_multi_result(
    df: pd.DataFrame,
    mr_group_name: str,
    mr_vars: list,
    metadata: dict,
    banner_var: str,
    empty_include: bool,
    missing_codes: list,
    pct_base: str,
    mr_selected_mode: str,
    mr_selected_codes: list,
    missing_rules_by_var: dict | None = None,
):
    categories = get_categories(df[banner_var], banner_var, metadata, empty_include, missing_codes, missing_rules_by_var=missing_rules_by_var)
    banner_rows = []

    for cat in categories:
        sub_df = _filter_df_for_banner(df, banner_var, cat)
        if sub_df.empty:
            continue

        computed = compute_multiresponse_rows(
            df=sub_df,
            mr_vars=mr_vars,
            metadata=metadata,
            missing_codes=missing_codes,
            pct_base=pct_base,
            mr_selected_mode=mr_selected_mode,
            mr_selected_codes=mr_selected_codes,
            missing_rules_by_var=missing_rules_by_var,
        )

        banner_rows.append({
            "banner_label": get_value_label(banner_var, cat, metadata),
            "base": computed["base"],
            "rows": computed["rows"],
        })

    return {
        "question_id": str(mr_group_name),
        "question_text": str(mr_group_name),
        "question_type": "multi",
        "banner_var": banner_var,
        "banner_var_label": get_var_label(banner_var, metadata),
        "banner_rows": banner_rows,
        "rows": [],
        "base": None,
    }


def build_dashboard_result(
    df: pd.DataFrame,
    question_id: str,
    question_type: str,
    metadata: dict,
    missing_codes: list,
    pct_base: str,
    banner_var: str | None,
    mr_group_map: dict | None,
    mr_selected_mode: str,
    mr_selected_codes: list,
    empty_include: bool = True,
    missing_rules_by_var: dict | None = None,
):
    if banner_var in [None, "", "__total__"]:
        banner_var = None

    result_type = question_type if question_type in {"scale", "rank"} else "single"

    if question_type == "multi":
        if not mr_group_map or question_id not in mr_group_map:
            return None
        mr_vars = mr_group_map.get(question_id, [])
        if banner_var:
            return _build_bannered_multi_result(
                df=df,
                mr_group_name=question_id,
                mr_vars=mr_vars,
                metadata=metadata,
                banner_var=banner_var,
                empty_include=empty_include,
                missing_codes=missing_codes,
                pct_base=pct_base,
                mr_selected_mode=mr_selected_mode,
                mr_selected_codes=mr_selected_codes,
                missing_rules_by_var=missing_rules_by_var,
            )

        computed = compute_multiresponse_rows(
            df=df,
            mr_vars=mr_vars,
            metadata=metadata,
            missing_codes=missing_codes,
            pct_base=pct_base,
            mr_selected_mode=mr_selected_mode,
            mr_selected_codes=mr_selected_codes,
            missing_rules_by_var=missing_rules_by_var,
        )
        return {
            "question_id": str(question_id),
            "question_text": str(question_id),
            "question_type": "multi",
            "base": computed["base"],
            "rows": computed["rows"],
            "banner_var": None,
            "banner_var_label": None,
            "banner_rows": [],
        }

    if banner_var:
        return _build_bannered_single_result(
            df=df,
            var=question_id,
            banner_var=banner_var,
            metadata=metadata,
            empty_include=empty_include,
            missing_codes=missing_codes,
            pct_base=pct_base,
            result_type=result_type,
            missing_rules_by_var=missing_rules_by_var,
        )

    computed = compute_single_rows(
        df=df,
        var_name=question_id,
        metadata=metadata,
        empty_include=empty_include,
        missing_codes=missing_codes,
        pct_base=pct_base,
        missing_rules_by_var=missing_rules_by_var,
    )

    return {
        "question_id": str(question_id),
        "question_text": get_var_label(question_id, metadata),
        "question_type": result_type,
        "base": computed["base"],
        "rows": computed["rows"],
        "banner_var": None,
        "banner_var_label": None,
        "banner_rows": [],
    }


def _prepare_rows(result: dict) -> pd.DataFrame:
    if result.get("banner_var"):
        rows = []
        for item in result.get("banner_rows", []):
            banner_label = item.get("banner_label", "")
            for row in item.get("rows", []):
                pct = row.get("pct")
                if pct is None:
                    continue
                rows.append({
                    "group": str(banner_label),
                    "label": str(row.get("label", "")),
                    "n": row.get("n"),
                    "pct": float(pct),
                })
        return pd.DataFrame(rows)

    rows = []
    for row in result.get("rows", []):
        pct = row.get("pct")
        if pct is None:
            continue
        rows.append({
            "label": str(row.get("label", "")),
            "n": row.get("n"),
            "pct": float(pct),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("pct", ascending=False).reset_index(drop=True)


def dashboard_result_to_dataframe(result: dict) -> pd.DataFrame:
    df = _prepare_rows(result)
    if df.empty:
        if result.get("banner_var"):
            return pd.DataFrame(columns=["독립변수", "보기", "응답수", "비율(%)"])
        return pd.DataFrame(columns=["보기", "응답수", "비율(%)"])

    if "group" in df.columns:
        return df.rename(columns={"group": "독립변수", "label": "보기", "n": "응답수", "pct": "비율(%)"})
    return df.rename(columns={"label": "보기", "n": "응답수", "pct": "비율(%)"})


def build_dashboard_summary_text(result: dict) -> str:
    df = _prepare_rows(result)
    if df.empty:
        return "표시할 결과가 없습니다."

    if "group" in df.columns:
        top = df.sort_values("pct", ascending=False).iloc[0]
        return f"가장 높은 응답은 '{top['group']}' 집단의 '{top['label']}' {format_pct(top['pct'])}%입니다."

    top = df.iloc[0]
    bottom = df.iloc[-1]
    if len(df) == 1:
        return f"{top['label']}이(가) {format_pct(top['pct'])}%입니다."
    gap = top["pct"] - bottom["pct"]
    return (
        f"가장 높은 응답은 '{top['label']}' {format_pct(top['pct'])}%이고, "
        f"가장 낮은 응답은 '{bottom['label']}' {format_pct(bottom['pct'])}%입니다. "
        f"격차는 {format_pct(gap)}%p입니다."
    )


def _apply_dashboard_plotly_layout(fig, title: str, height: int = 460, showlegend: bool = False):
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left"},
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "NanumGothic", "size": 12, "color": "#0F172A"},
        margin={"l": 20, "r": 20, "t": 64, "b": 20},
        height=height,
        showlegend=showlegend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    return fig


def _wrap_axis_labels(series: pd.Series, width: int = 14) -> list[str]:
    labels = []
    for value in series.astype(str).tolist():
        chunks = [value[i:i+width] for i in range(0, len(value), width)]
        labels.append("<br>".join(chunks[:2]))
    return labels


def _build_radar_figure(df: pd.DataFrame, title: str):
    labels = df["label"].astype(str).tolist()
    values = df["pct"].astype(float).tolist()
    return render_radar_chart(labels, values, title)


def render_dashboard_chart(result: dict, chart_type: str):
    df = _prepare_rows(result)
    if df.empty:
        return None

    title = f"{result.get('question_id', '')} | {result.get('question_text', '')}"
    question_type = str(result.get("question_type", "single"))
    n_labels = int(df["label"].nunique()) if "label" in df.columns else len(df)
    auto_h = _auto_plotly_height(chart_type, n_labels)

    if "group" in df.columns:
        if chart_type in {"donut", "radar"}:
            chart_type = "bar"

        if chart_type == "barh":
            fig = px.bar(
                df, x="pct", y="label", color="group", orientation="h",
                text="pct", barmode="group", color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                title=title, xaxis_title="%", yaxis_title="",
                legend_title_text=result.get("banner_var_label") or "독립변수",
                yaxis={"categoryorder": "total ascending"},
            )
        else:
            fig = px.bar(
                df, x="label", y="pct", color="group", text="pct",
                barmode="group", color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                title=title, xaxis_title="", yaxis_title="%",
                legend_title_text=result.get("banner_var_label") or "독립변수",
            )

        fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=60, b=20), height=auto_h)
        return fig

    # ── 단일 결과 (배너 없음) ──────────────────────────────────
    def _add_mean_line(fig, df_inner, orientation="v"):
        """척도/순위형에만 평균선 추가."""
        if question_type not in {"scale", "rank"}:
            return
        labels = df_inner["label"].astype(str).tolist()
        pcts = df_inner["pct"].astype(float).tolist()
        codes = []
        for lbl in labels:
            try:
                codes.append(float(lbl))
            except Exception:
                return
        if not codes:
            return
        total_pct = sum(pcts)
        if total_pct == 0:
            return
        mean_val = sum(c * p for c, p in zip(codes, pcts)) / total_pct
        # x 위치를 카테고리 인덱스로 변환
        step = codes[1] - codes[0] if len(codes) > 1 else 1
        mean_idx = (mean_val - codes[0]) / step

        if orientation == "v":
            fig.add_vline(
                x=mean_idx, line_dash="dash", line_color=MEAN_LINE_COLOR,
                line_width=2.2, opacity=0.85,
            )
            fig.add_annotation(
                x=mean_idx, y=max(pcts) * 0.92,
                text=f"<b>평균 {format_stat(mean_val)}</b>",
                showarrow=False, font=dict(color=MEAN_LINE_COLOR, size=12),
                xanchor="left", xshift=6,
            )
        else:
            fig.add_hline(
                y=mean_idx, line_dash="dash", line_color=MEAN_LINE_COLOR,
                line_width=2.2, opacity=0.85,
            )
            fig.add_annotation(
                y=mean_idx, x=max(pcts) * 0.92,
                text=f"<b>평균 {format_stat(mean_val)}</b>",
                showarrow=False, font=dict(color=MEAN_LINE_COLOR, size=12),
                yanchor="bottom", yshift=4,
            )

    if chart_type == "barh":
        fig = px.bar(df, x="pct", y="label", orientation="h", text="pct",
                     color_discrete_sequence=[PLOTLY_COLOR_SEQUENCE[0]])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          marker_color=PLOTLY_COLOR_SEQUENCE[0])
        fig.update_layout(title=title, xaxis_title="%", yaxis_title="",
                          yaxis={"categoryorder": "total ascending"})
        _add_mean_line(fig, df, orientation="h")
    elif chart_type == "bar":
        fig = px.bar(df, x="label", y="pct", text="pct",
                     color_discrete_sequence=[PLOTLY_COLOR_SEQUENCE[0]])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          marker_color=PLOTLY_COLOR_SEQUENCE[0])
        fig.update_layout(title=title, xaxis_title="", yaxis_title="%")
        _add_mean_line(fig, df, orientation="v")
    elif chart_type == "donut":
        fig = go.Figure(data=[go.Pie(
            labels=df["label"], values=df["pct"], hole=0.45,
            textinfo="label+percent",
            marker=dict(colors=PLOTLY_COLOR_SEQUENCE[:len(df)]),
        )])
        fig.update_layout(title=title, template="plotly_white",
                          margin=dict(l=20, r=20, t=60, b=20), height=auto_h)
        return fig
    elif chart_type == "radar":
        return _build_radar_figure(df, title)
    else:
        return None

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=auto_h,
        showlegend=False,
    )
    return fig


def _result_is_empty(result: dict | None) -> bool:
    if not result:
        return True
    if result.get("banner_var"):
        for row in result.get("banner_rows", []):
            if row.get("rows"):
                return False
        return True
    return len(result.get("rows", [])) == 0


def _coerce_numeric_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def _infer_scale_vars(config_df: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    if config_df is None or config_df.empty:
        return []
    mask = config_df["question_type"].astype(str).str.strip().eq("scale")
    vars_ = [str(v) for v in config_df.loc[mask, "question_id"].tolist() if str(v) in df.columns]
    return sort_var_names(list(dict.fromkeys(vars_)))


def _build_profile_summary(df: pd.DataFrame, profile_vars: list, metadata: dict, missing_codes: list, pct_base: str, missing_rules_by_var: dict | None = None) -> list:
    profile_summary = []
    for p_var in sort_var_names(profile_vars):
        if p_var not in df.columns:
            continue
        dist = compute_single_distribution(
            df=df,
            var_name=p_var,
            metadata=metadata,
            empty_include=True,
            missing_codes=missing_codes,
            pct_base=pct_base,
            missing_rules_by_var=missing_rules_by_var,
        )
        profile_summary.append({
            "var": p_var,
            "label": get_var_label(p_var, metadata),
            "data": dist.get("rows", []),
            "base": dist.get("base"),
        })
    return profile_summary


def _build_correlation_payload(df: pd.DataFrame, scale_vars: list, metadata: dict) -> dict | None:
    scale_vars = [v for v in scale_vars if v in df.columns]
    if len(scale_vars) < 2:
        return None
    numeric_df = _coerce_numeric_df(df, scale_vars)
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.shape[1] < 2:
        return None
    corr_df = numeric_df.corr()
    if corr_df.empty:
        return None
    cols = list(corr_df.columns)
    return {
        "vars": cols,
        "labels": [get_var_label(v, metadata) for v in cols],
        "values": corr_df.fillna(0).round(4).values.tolist(),
    }


def _build_executive_summary(items: list, correlation_matrix: dict | None, profile_summary: list) -> list[str]:
    points = []

    item_scores = []
    for item in items:
        default_key = item.get("default_banner_key")
        result = item.get("results", {}).get(default_key)
        if not result:
            continue
        df_rows = _prepare_rows(result)
        if df_rows.empty:
            continue
        top = df_rows.sort_values("pct", ascending=False).iloc[0]
        item_scores.append((float(top["pct"]), f"'{item.get('display_label', item.get('question_id'))}'에서 '{top['label']}' 응답이 {format_pct(top['pct'])}%로 가장 두드러집니다."))

    if item_scores:
        points.append(sorted(item_scores, key=lambda x: x[0], reverse=True)[0][1])

    if correlation_matrix and correlation_matrix.get("values"):
        labels = correlation_matrix.get("labels", [])
        values = correlation_matrix.get("values", [])
        best = None
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                try:
                    val = float(values[i][j])
                except Exception:
                    continue
                if best is None or abs(val) > abs(best[0]):
                    best = (val, labels[i], labels[j])
        if best is not None:
            sign_text = "정(+)" if best[0] >= 0 else "부(-)"
            points.append(f"지표 관계에서는 '{best[1]}'와 '{best[2]}'의 상관이 가장 크게 나타났습니다. ({sign_text} 상관 {format_stat(best[0])})")

    if profile_summary:
        first_profile = profile_summary[0]
        data = first_profile.get("data", [])
        if data:
            top_row = sorted(data, key=lambda x: float(x.get("pct") or 0), reverse=True)[0]
            points.append(f"응답자 특성 기준으로는 '{first_profile.get('label')}'에서 '{top_row.get('label')}' 비중이 가장 높았습니다. ({format_pct(float(top_row.get('pct') or 0))}%)")

    return points[:3]


def build_dashboard_bundle(
    config_df: pd.DataFrame,
    df: pd.DataFrame,
    metadata: dict,
    profile_vars: list,
    missing_codes: list,
    pct_base: str,
    mr_group_map: dict,
    mr_selected_mode: str,
    mr_selected_codes: list,
    app_title: str,
    app_subtitle: str = "",
    preferred_subdomain: str = "",
    missing_rules_by_var: dict | None = None,
):
    selected_df = get_selected_dashboard_items(config_df)
    banner_options = get_dashboard_banner_options(profile_vars, metadata)
    banner_label_to_key = {label: key for label, key in banner_options.items()}
    banner_items = [{"label": label, "key": key} for label, key in banner_options.items()]

    items = []
    for row in selected_df.to_dict("records"):
        default_banner_key = banner_label_to_key.get(row.get("default_banner", "전체"), "__total__")
        results_map = {}

        for banner_label, banner_key in banner_options.items():
            result = build_dashboard_result(
                df=df,
                question_id=row["question_id"],
                question_type=row.get("question_type", "single"),
                metadata=metadata,
                missing_codes=missing_codes,
                pct_base=pct_base,
                banner_var=banner_key,
                mr_group_map=mr_group_map,
                mr_selected_mode=mr_selected_mode,
                mr_selected_codes=mr_selected_codes,
                empty_include=True,
                missing_rules_by_var=missing_rules_by_var,
            )
            if _result_is_empty(result):
                continue
            results_map[banner_key] = result

        if not results_map:
            continue
        if default_banner_key not in results_map:
            default_banner_key = next(iter(results_map.keys()))

        items.append({
            "question_id": row["question_id"],
            "display_label": row.get("display_label") or row["question_id"],
            "description": row.get("description", ""),
            "question_type": row.get("question_type", "single"),
            "section": row.get("section", "추천 문항"),
            "default_banner_key": default_banner_key,
            "default_chart": row.get("default_chart", "가로 막대"),
            "available_banners": [{"label": label, "key": key} for label, key in banner_options.items() if key in results_map],
            "results": results_map,
        })

    scale_vars = _infer_scale_vars(config_df, df)
    profile_summary = _build_profile_summary(df, profile_vars, metadata, missing_codes, pct_base, missing_rules_by_var=missing_rules_by_var)
    correlation_matrix = _build_correlation_payload(df, scale_vars, metadata)
    summary_points = _build_executive_summary(items, correlation_matrix, profile_summary)

    bundle = {
        "app_meta": {
            "title": app_title,
            "subtitle": app_subtitle,
            "preferred_subdomain": slugify_text(preferred_subdomain or app_title, default="dashboard"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pct_base": pct_base,
            "summary_points": summary_points,
        },
        "overview": {
            "profile_summary": profile_summary,
            "total_n": int(len(df)),
        },
        "relation": {
            "correlation": correlation_matrix,
        },
        "banner_options": banner_items,
        "items": items,
    }
    return bundle


def build_dashboard_bundle_json_bytes(bundle: dict) -> bytes:
    return dict_to_json_bytes(bundle)


def _build_streamlit_bundle_app_py() -> str:
    return '''import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PLOTLY_COLOR_SEQUENCE = ["#1d4ed8", "#475569", "#0ea5e9", "#4E79A7", "#F28E2B", "#E15759"]
DASHBOARD_CHART_OPTIONS = {"가로 막대": "barh", "세로 막대": "bar", "도넛": "donut", "레이더": "radar"}


def _load_bundle():
    bundle_path = Path(__file__).with_name("dashboard_bundle.json")
    with bundle_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_rows(result: dict) -> pd.DataFrame:
    if result.get("banner_var"):
        rows = []
        for item in result.get("banner_rows", []):
            banner_label = item.get("banner_label", "")
            for row in item.get("rows", []):
                pct = row.get("pct")
                if pct is None:
                    continue
                rows.append({"group": str(banner_label), "label": str(row.get("label", "")), "n": row.get("n"), "pct": float(pct)})
        return pd.DataFrame(rows)

    rows = []
    for row in result.get("rows", []):
        pct = row.get("pct")
        if pct is None:
            continue
        rows.append({"label": str(row.get("label", "")), "n": row.get("n"), "pct": float(pct)})
    df = pd.DataFrame(rows)
    return df.sort_values("pct", ascending=False).reset_index(drop=True) if not df.empty else df


def _summary_text(result: dict) -> str:
    df = _prepare_rows(result)
    if df.empty:
        return "표시할 결과가 없습니다."
    if "group" in df.columns:
        top = df.sort_values("pct", ascending=False).iloc[0]
        return f"가장 높은 응답은 '{top['group']}' 집단의 '{top['label']}' {format_pct(top['pct'])}%입니다."
    top = df.iloc[0]
    bottom = df.iloc[-1]
    if len(df) == 1:
        return f"{top['label']}이(가) {format_pct(top['pct'])}%입니다."
    return f"가장 높은 응답은 '{top['label']}' {format_pct(top['pct'])}%이고, 가장 낮은 응답은 '{bottom['label']}' {format_pct(bottom['pct'])}%입니다. 격차는 {format_pct(top['pct'] - bottom['pct'])}%p입니다."


def _result_to_dataframe(result: dict) -> pd.DataFrame:
    df = _prepare_rows(result)
    if df.empty:
        return pd.DataFrame(columns=["보기", "응답수", "비율(%)"])
    if "group" in df.columns:
        return df.rename(columns={"group": "독립변수", "label": "보기", "n": "응답수", "pct": "비율(%)"})
    return df.rename(columns={"label": "보기", "n": "응답수", "pct": "비율(%)"})


def _render_radar(labels, values, title):
    labels = list(labels)
    values = list(values)
    if labels and values:
        labels = labels + labels[:1]
        values = values + values[:1]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill="toself", line=dict(color="#1d4ed8")))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title=title, template="plotly_white")
    return fig


def _render_chart(result: dict, chart_type: str):
    df = _prepare_rows(result)
    if df.empty:
        return None
    title = f"{result.get('question_id', '')} | {result.get('question_text', '')}"

    if "group" in df.columns:
        if chart_type in {"donut", "레이더", "radar"}:
            chart_type = "bar"
        if chart_type == "barh":
            fig = px.bar(df, x="pct", y="label", color="group", orientation="h", text="pct", barmode="group", color_discrete_sequence=PLOTLY_COLOR_SEQUENCE)
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(title=title, xaxis_title="%", yaxis_title="", yaxis={"categoryorder": "total ascending"}, template="plotly_white")
            return fig
        fig = px.bar(df, x="label", y="pct", color="group", text="pct", barmode="group", color_discrete_sequence=PLOTLY_COLOR_SEQUENCE)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(title=title, yaxis_title="%", template="plotly_white")
        return fig

    if chart_type == "barh":
        fig = px.bar(df, x="pct", y="label", orientation="h", text="pct", color_discrete_sequence=[PLOTLY_COLOR_SEQUENCE[0]])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(title=title, xaxis_title="%", yaxis_title="", yaxis={"categoryorder": "total ascending"}, template="plotly_white")
        return fig
    if chart_type == "bar":
        fig = px.bar(df, x="label", y="pct", text="pct", color_discrete_sequence=[PLOTLY_COLOR_SEQUENCE[0]])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(title=title, yaxis_title="%", template="plotly_white")
        return fig
    if chart_type == "donut":
        fig = go.Figure(data=[go.Pie(labels=df["label"], values=df["pct"], hole=0.45)])
        fig.update_layout(title=title, template="plotly_white")
        return fig
    return _render_radar(df["label"], df["pct"], title)


def _render_overview(bundle: dict):
    overview = bundle.get("overview", {})
    meta = bundle.get("app_meta", {})
    st.subheader("조사 결과 핵심 요약")
    summary_points = meta.get("summary_points") or []
    if summary_points:
        st.info("\n".join([f"• {x}" for x in summary_points]))
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("전체 응답 수", f"{overview.get('total_n', 0):,}")
    with c2:
        profiles = overview.get("profile_summary", [])[:3]
        for prof in profiles:
            data = prof.get("data", [])
            top = sorted(data, key=lambda x: float(x.get("pct") or 0), reverse=True)[:3]
            st.markdown(f"**{prof.get('label', '')}**")
            for row in top:
                st.caption(f"- {row.get('label')}: {format_pct(float(row.get('pct') or 0))}%")


def _render_relation(bundle: dict):
    relation = bundle.get("relation", {})
    corr = relation.get("correlation")
    if not corr:
        st.info("지표 관계를 표시할 척도형 문항이 2개 이상 필요합니다.")
        return
    labels = corr.get("labels", [])
    values = corr.get("values", [])
    heatmap_df = pd.DataFrame(values, index=labels, columns=labels)
    fig = px.imshow(heatmap_df, text_auto=".2f", color_continuous_scale="Blues", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(template="plotly_white", height=max(420, 90 + len(labels) * 34), margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _render_analysis(bundle: dict):
    items = bundle.get("items", [])
    if not items:
        st.info("표시할 문항이 없습니다.")
        return

    option_map = {item["display_label"]: item for item in items}
    selected_label = st.selectbox("문항 선택", list(option_map.keys()))
    item = option_map[selected_label]
    banner_options = item.get("available_banners", [])
    banner_map = {x["label"]: x["key"] for x in banner_options}
    selected_banner_label = st.selectbox("비교 기준", list(banner_map.keys()))
    chart_options = list(DASHBOARD_CHART_OPTIONS.keys())
    default_chart = item.get("default_chart", "가로 막대")
    chart_idx = chart_options.index(default_chart) if default_chart in chart_options else 0
    selected_chart = st.selectbox("그래프 유형", chart_options, index=chart_idx)
    result = item["results"][banner_map[selected_banner_label]]
    fig = _render_chart(result, DASHBOARD_CHART_OPTIONS[selected_chart])
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    summary_text = _summary_text(result)
    st.caption(summary_text)
    st.code(summary_text, language="text")
    st.dataframe(_result_to_dataframe(result), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="조사 결과 대시보드", layout="wide")
    bundle = _load_bundle()
    meta = bundle.get("app_meta", {})
    st.title(meta.get("title", "조사 결과 대시보드"))
    if meta.get("subtitle"):
        st.caption(meta.get("subtitle"))
    tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Relation"])
    with tab1:
        _render_overview(bundle)
    with tab2:
        _render_analysis(bundle)
    with tab3:
        _render_relation(bundle)


if __name__ == "__main__":
    main()
'''


def _build_dashboard_deploy_readme(bundle: dict) -> str:
    meta = bundle.get("app_meta", {})
    title = meta.get("title", "조사 결과 대시보드")
    subtitle = meta.get("subtitle", "")
    generated_at = meta.get("generated_at", "")
    summary_points = meta.get("summary_points") or []
    lines = [
        f"# {title}",
        "",
        subtitle,
        "",
        f"- 생성일시: {generated_at}",
        f"- 권장 서브도메인: {meta.get('preferred_subdomain', 'dashboard')}",
        "",
        "## 포함 파일",
        "- dashboard_bundle.json : 대시보드 데이터 번들",
        "- streamlit_app.py : 배포용 Streamlit 앱",
        "- requirements.txt : 실행 패키지 목록",
    ]
    if summary_points:
        lines.extend(["", "## 핵심 요약"])
        lines.extend([f"- {point}" for point in summary_points])
    return "\n".join(lines)


def _build_static_dashboard_html(bundle: dict) -> str:
    meta = bundle.get("app_meta", {})
    title = str(meta.get("title", "조사 결과 대시보드"))
    subtitle = str(meta.get("subtitle", ""))
    bundle_json = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")
    font_css = browser_font_face_css()
    available_fonts = list(local_font_assets().keys())
    primary_font = browser_primary_font_name()
    browser_family = f"'{primary_font}', 'Pretendard', 'Noto Sans KR', 'Malgun Gothic', sans-serif" if primary_font else "'Pretendard', 'Noto Sans KR', 'Malgun Gothic', sans-serif"

    return f'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    {font_css}
    :root{{--bg:#f6f8fb;--panel:#fff;--line:#dfe5ee;--text:#172033;--muted:#667085;--accent:#2457d6;--accent-soft:#eaf0ff;--shadow:0 8px 28px rgba(20,34,66,.08)}}
    *{{box-sizing:border-box}} html,body{{margin:0;min-height:100%;background:var(--bg);color:var(--text);font-family:{browser_family}}}
    button,input,select{{font:inherit}} button{{cursor:pointer}}
    .shell{{display:grid;grid-template-columns:310px minmax(0,1fr);min-height:100vh}}
    .sidebar{{position:sticky;top:0;height:100vh;background:var(--panel);border-right:1px solid var(--line);padding:22px 18px;overflow:auto}}
    .brand h1{{font-size:21px;margin:0 0 6px}} .brand p{{font-size:13px;color:var(--muted);margin:0 0 20px;line-height:1.5}}
    .label{{display:block;font-size:12px;font-weight:700;color:var(--muted);margin:15px 0 6px}}
    .control{{width:100%;border:1px solid var(--line);border-radius:10px;background:#fff;padding:10px 11px;color:var(--text)}}
    .navrow{{display:grid;grid-template-columns:42px 1fr 42px;gap:7px;margin-top:12px}} .navbtn{{border:1px solid var(--line);border-radius:10px;background:#fff;padding:9px}}
    .question-list{{margin-top:12px;display:grid;gap:5px}} .qitem{{width:100%;text-align:left;border:0;border-radius:9px;background:transparent;padding:9px 10px;color:var(--text);line-height:1.35}}
    .qitem:hover,.qitem.active{{background:var(--accent-soft);color:var(--accent)}} .qitem small{{display:block;color:var(--muted);margin-bottom:2px}}
    .main{{padding:28px;min-width:0}} .topbar{{display:flex;gap:12px;align-items:flex-start;justify-content:space-between;margin-bottom:18px}}
    .eyebrow{{font-size:13px;color:var(--accent);font-weight:700;margin-bottom:5px}} .title{{font-size:25px;margin:0;line-height:1.35}} .desc{{color:var(--muted);margin:7px 0 0;line-height:1.6}}
    .favorite{{border:1px solid var(--line);background:#fff;border-radius:12px;padding:10px 13px;white-space:nowrap}} .favorite.on{{background:#fff6d8;border-color:#ead48a}}
    .metrics{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}} .metric{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:11px 14px;min-width:130px}}
    .metric span{{display:block;color:var(--muted);font-size:12px}} .metric strong{{display:block;font-size:20px;margin-top:3px}}
    .panel{{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);padding:18px}}
    .toolbar{{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin-bottom:8px}} .toolbar>label{{min-width:170px;flex:0 1 230px}}
    #chart{{width:100%;min-height:500px}} .note{{font-size:13px;color:var(--muted);line-height:1.6;margin-top:8px}}
    .empty{{padding:80px 20px;text-align:center;color:var(--muted)}}
    .fav-only{{display:flex;align-items:center;gap:8px;margin-top:13px;font-size:13px;color:var(--muted)}}
    @media(max-width:850px){{.shell{{display:block}}.sidebar{{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line)}}.main{{padding:18px}}.question-list{{max-height:260px;overflow:auto}}#chart{{min-height:430px}}}}
  </style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand"><h1>{title}</h1><p>{subtitle}</p></div>
    <label class="label" for="search">문항 검색</label><input id="search" class="control" placeholder="문항번호 또는 문항 내용">
    <label class="label" for="section">파트·문항영역</label><select id="section" class="control"></select>
    <label class="label" for="question">문항 선택</label><select id="question" class="control"></select>
    <div class="navrow"><button id="prev" class="navbtn" type="button">◀</button><button id="random" class="navbtn" type="button">선택 문항 보기</button><button id="next" class="navbtn" type="button">▶</button></div>
    <label class="fav-only"><input id="favOnly" type="checkbox"> 즐겨찾기만 보기</label>
    <div id="questionList" class="question-list"></div>
  </aside>
  <main class="main">
    <div id="content"></div>
  </main>
</div>
<script id="dashboard-data" type="application/json">{bundle_json}</script>
<script>
const bundle=JSON.parse(document.getElementById('dashboard-data').textContent);
const items=bundle.items||[];
const state={{index:0,search:'',section:'전체',favOnly:false,favorites:new Set(),banner:null,chart:null}};
const el=id=>document.getElementById(id);
const escapeHtml=s=>String(s??'').replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
function sections(){{return ['전체',...new Set(items.map(x=>x.section||'미분류'))]}}
function filtered(){{return items.filter(x=>{{const text=`${{x.question_id}} ${{x.display_label}} ${{x.description||''}}`.toLowerCase();return (!state.search||text.includes(state.search.toLowerCase()))&&(state.section==='전체'||(x.section||'미분류')===state.section)&&(!state.favOnly||state.favorites.has(x.question_id));}})}}
function currentList(){{const list=filtered();if(state.index>=list.length)state.index=Math.max(0,list.length-1);return list}}
function syncControls(){{
 const secs=sections(); el('section').innerHTML=secs.map(x=>`<option>${{escapeHtml(x)}}</option>`).join('');el('section').value=state.section;
 const list=currentList();el('question').innerHTML=list.map((x,i)=>`<option value="${{i}}">${{escapeHtml(x.question_id)}}. ${{escapeHtml(x.display_label)}}</option>`).join('');el('question').value=String(state.index);
 el('questionList').innerHTML=list.map((x,i)=>`<button type="button" class="qitem ${{i===state.index?'active':''}}" data-i="${{i}}"><small>${{escapeHtml(x.section||'미분류')}} · ${{escapeHtml(x.question_id)}}</small>${{escapeHtml(x.display_label)}}</button>`).join('');
 document.querySelectorAll('.qitem').forEach(b=>b.addEventListener('click',()=>{{state.index=Number(b.dataset.i);state.banner=null;state.chart=null;render();}}));
}}
function prepare(result){{
 if(result.banner_var){{const rows=[];(result.banner_rows||[]).forEach(g=>(g.rows||[]).forEach(r=>{{if(r.pct!=null)rows.push({{group:String(g.banner_label||''),label:String(r.label||''),n:r.n??0,pct:Number(r.pct)}})}}));return rows;}}
 return (result.rows||[]).filter(r=>r.pct!=null).map(r=>({{label:String(r.label||''),n:r.n??0,pct:Number(r.pct)}}));
}}
function plot(item,result,chartType){{
 const rows=prepare(result);if(!rows.length){{el('chart').innerHTML='<div class="empty">표시할 결과가 없습니다.</div>';return;}}
 const grouped=rows.some(r=>r.group!==undefined);let traces=[];let layout={{margin:{{t:35,r:25,b:95,l:65}},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{{family:{json.dumps(browser_family, ensure_ascii=False)}}},yaxis:{{title:'%',rangemode:'tozero',gridcolor:'#edf0f5'}},xaxis:{{tickangle:rows.length>6?-25:0}},legend:{{orientation:'h',y:-.25}},hovermode:'closest'}};
 if(grouped){{const groups=[...new Set(rows.map(r=>r.group))];traces=groups.map(g=>{{const d=rows.filter(r=>r.group===g);return {{type:'bar',name:g,x:d.map(r=>r.label),y:d.map(r=>r.pct),customdata:d.map(r=>r.n),text:d.map(r=>r.pct.toFixed(1)+'%'),textposition:'outside',hovertemplate:'%{{x}}<br>'+g+': %{{y:.1f}}%<br>N=%{{customdata}}<extra></extra>'}}}});layout.barmode='group';}}
 else if(chartType==='도넛'){{traces=[{{type:'pie',labels:rows.map(r=>r.label),values:rows.map(r=>r.pct),customdata:rows.map(r=>r.n),hole:.48,hovertemplate:'%{{label}}: %{{value:.1f}}%<br>N=%{{customdata}}<extra></extra>'}}];layout.margin={{t:20,r:20,b:20,l:20}};}}
 else if(chartType==='가로 막대'){{traces=[{{type:'bar',orientation:'h',y:rows.map(r=>r.label),x:rows.map(r=>r.pct),customdata:rows.map(r=>r.n),text:rows.map(r=>r.pct.toFixed(1)+'%'),textposition:'outside',marker:{{color:'#2457d6'}},hovertemplate:'%{{y}}: %{{x:.1f}}%<br>N=%{{customdata}}<extra></extra>'}}];layout.xaxis={{title:'%',rangemode:'tozero',gridcolor:'#edf0f5'}};layout.yaxis={{automargin:true,categoryorder:'array',categoryarray:rows.map(r=>r.label).slice().reverse()}};layout.margin={{t:25,r:45,b:55,l:150}};}}
 else{{traces=[{{type:'bar',x:rows.map(r=>r.label),y:rows.map(r=>r.pct),customdata:rows.map(r=>r.n),text:rows.map(r=>r.pct.toFixed(1)+'%'),textposition:'outside',marker:{{color:'#2457d6'}},hovertemplate:'%{{x}}: %{{y:.1f}}%<br>N=%{{customdata}}<extra></extra>'}}];}}
 Plotly.react('chart',traces,layout,{{responsive:true,displaylogo:false,toImageButtonOptions:{{format:'png',filename:item.question_id+'_chart',scale:2}}}});
}}
function render(){{
 syncControls();const list=currentList();if(!list.length){{el('content').innerHTML='<div class="panel empty">조건에 맞는 문항이 없습니다.</div>';return;}}const item=list[state.index];
 const banners=item.available_banners||[];const bkey=state.banner&&item.results[state.banner]?state.banner:(item.default_banner_key||banners[0]?.key);state.banner=bkey;
 const chart=state.chart||item.default_chart||'세로 막대';state.chart=chart;const result=item.results[bkey];
 const fav=state.favorites.has(item.question_id);const bannerLabel=(banners.find(x=>x.key===bkey)||{{label:'전체'}}).label;
 el('content').innerHTML=`<div class="topbar"><div><div class="eyebrow">${{escapeHtml(item.section||'미분류')}} · ${{escapeHtml(item.question_id)}}</div><h2 class="title">${{escapeHtml(item.display_label)}}</h2><p class="desc">${{escapeHtml(item.description||'')}}</p></div><button id="favorite" class="favorite ${{fav?'on':''}}" type="button">${{fav?'★ 주요 문항':'☆ 주요 문항'}}</button></div><div class="metrics"><div class="metric"><span>유효응답 N</span><strong>${{Number(result?.base||0).toLocaleString()}}</strong></div><div class="metric"><span>비교 기준</span><strong>${{escapeHtml(bannerLabel)}}</strong></div><div class="metric"><span>현재 위치</span><strong>${{state.index+1}} / ${{list.length}}</strong></div></div><section class="panel"><div class="toolbar"><label><span class="label">비교 기준</span><select id="banner" class="control">${{banners.map(x=>`<option value="${{escapeHtml(x.key)}}">${{escapeHtml(x.label)}}</option>`).join('')}}</select></label><label><span class="label">그래프 유형</span><select id="chartType" class="control"><option>세로 막대</option><option>가로 막대</option><option>도넛</option></select></label></div><div id="chart"></div><div class="note">막대 또는 범주에 마우스를 올리면 비율과 응답 수를 확인할 수 있습니다. 차트 우측 상단 카메라 버튼으로 PNG 저장이 가능합니다.</div></section>`;
 el('banner').value=bkey;el('chartType').value=chart;el('favorite').addEventListener('click',()=>{{fav?state.favorites.delete(item.question_id):state.favorites.add(item.question_id);render();}});el('banner').addEventListener('change',e=>{{state.banner=e.target.value;render();}});el('chartType').addEventListener('change',e=>{{state.chart=e.target.value;render();}});plot(item,result,chart);
}}
el('search').addEventListener('input',e=>{{state.search=e.target.value;state.index=0;render();}});el('section').addEventListener('change',e=>{{state.section=e.target.value;state.index=0;render();}});el('question').addEventListener('change',e=>{{state.index=Number(e.target.value);state.banner=null;state.chart=null;render();}});el('favOnly').addEventListener('change',e=>{{state.favOnly=e.target.checked;state.index=0;render();}});el('prev').addEventListener('click',()=>{{const n=currentList().length;if(n)state.index=(state.index-1+n)%n;state.banner=null;state.chart=null;render();}});el('next').addEventListener('click',()=>{{const n=currentList().length;if(n)state.index=(state.index+1)%n;state.banner=null;state.chart=null;render();}});el('random').addEventListener('click',render);render();
</script>
</body>
</html>'''

def build_dashboard_deploy_files(bundle: dict) -> dict[str, bytes]:
    file_mapping = {
        "index.html": _build_static_dashboard_html(bundle).encode("utf-8"),
        "data.json": build_dashboard_bundle_json_bytes(bundle),
        "README.md": _build_dashboard_deploy_readme(bundle).encode("utf-8"),
    }
    for filename, content in local_font_assets().items():
        file_mapping[f"assets/fonts/{filename}"] = content
    return file_mapping


def build_dashboard_deploy_zip(bundle: dict, package_name: str = "survey_dashboard_package") -> bytes:
    package_dir = safe_filename(package_name or "survey_dashboard_package")
    file_mapping = {f"{package_dir}/{name}": content for name, content in build_dashboard_deploy_files(bundle).items()}
    return build_zip_bytes_from_mapping(file_mapping)


def _render_profile_card(profile_item: dict):
    st.markdown(f"**{profile_item.get('label', '')}**")
    rows = profile_item.get("data", [])
    if not rows:
        st.caption("표시할 데이터 없음")
        return
    top_rows = sorted(rows, key=lambda x: float(x.get("pct") or 0), reverse=True)[:3]
    for row in top_rows:
        pct = float(row.get("pct") or 0)
        st.caption(f"{row.get('label')}: {format_pct(pct)}%")
        st.progress(min(max(pct / 100.0, 0.0), 1.0))


def _render_overview_tab(bundle: dict):
    meta = bundle.get("app_meta", {})
    overview = bundle.get("overview", {})
    summary_points = meta.get("summary_points") or []

    st.subheader("조사 결과 핵심 요약")
    if summary_points:
        st.info("\n".join([f"• {point}" for point in summary_points]))
    else:
        st.info("자동 요약 포인트가 아직 없습니다.")

    c1, c2 = st.columns([1.2, 2.0])
    with c1:
        st.metric("전체 응답 수", f"{overview.get('total_n', 0):,}")
        st.caption(f"생성 시각: {meta.get('generated_at', '')}")
        st.caption(f"비율 기준: {'유효응답' if meta.get('pct_base') == 'valid' else '전체응답'}")
    with c2:
        st.markdown("#### 응답자 특성 요약")
        profiles = overview.get("profile_summary", [])
        if not profiles:
            st.caption("응답자 특성 요약 데이터가 없습니다.")
        else:
            cols = st.columns(min(3, len(profiles[:3])))
            for idx, profile_item in enumerate(profiles[:3]):
                with cols[idx % len(cols)]:
                    _render_profile_card(profile_item)

    profiles = overview.get("profile_summary", [])
    if profiles:
        st.markdown("#### 응답자 특성 분포")
        selected_profile_label = st.selectbox("응답자 특성 변수", [p["label"] for p in profiles], key="dashboard_overview_profile")
        selected_profile = next((p for p in profiles if p["label"] == selected_profile_label), None)
        if selected_profile:
            prof_df = pd.DataFrame(selected_profile.get("data", []))
            if not prof_df.empty:
                fig = px.bar(
                    prof_df,
                    x="label",
                    y="pct",
                    text="pct",
                    color_discrete_sequence=[PLOTLY_COLOR_SEQUENCE[0]],
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="", yaxis_title="%")
                st.plotly_chart(fig, use_container_width=True)


def _render_analysis_tab(bundle: dict, extra_params: dict | None = None):
    """
    extra_params: df, metadata, missing_codes, pct_base,
                  mr_group_map, mr_selected_mode, mr_selected_codes
    — 이 값이 있을 때만 드릴다운 필터가 활성화됩니다.
    """
    items = bundle.get("items", [])
    if not items:
        st.info("추천 문항으로 노출할 항목을 먼저 1개 이상 선택해줘.")
        return

    # ── 드릴다운 필터 배지 ────────────────────────────────────
    active_filter = st.session_state.get("_drilldown_filter")
    filtered_df = None
    filter_n = None

    if active_filter and extra_params:
        df_orig = extra_params.get("df")
        metadata = extra_params.get("metadata", {})
        if df_orig is not None:
            filtered_df, filter_n = _apply_drilldown_filter(df_orig, active_filter, metadata)

    if active_filter:
        n_text = f" (N={filter_n:,})" if filter_n is not None else ""
        col_msg, col_btn = st.columns([4, 1])
        with col_msg:
            st.info(
                f"🔍 **드릴다운 필터 적용 중** — "
                f"{active_filter.get('var_label', '')} = **{active_filter.get('label')}**{n_text}"
            )
        with col_btn:
            if st.button("필터 초기화 ✕", key="drilldown_reset", use_container_width=True):
                st.session_state["_drilldown_filter"] = None
                st.rerun()

    # ── 문항 선택 ─────────────────────────────────────────────
    option_map = {item["display_label"]: item for item in items}
    selected_label = st.selectbox("문항 선택", list(option_map.keys()), key="dashboard_analysis_question")
    item = option_map[selected_label]
    current_qid = item["question_id"]
    question_type = item.get("question_type", "single")

    # 필터가 현재 문항 자신이면 경고
    is_self_filter = (active_filter and active_filter.get("question_id") == current_qid)
    if is_self_filter:
        st.warning("이 문항이 현재 필터 기준입니다. 다른 문항을 선택하면 필터된 분포를 확인할 수 있습니다.")

    # ── 배너 / 차트 유형 선택 (필터 적용 중에는 배너 비활성) ──
    use_filter_result = (
        filtered_df is not None
        and not is_self_filter
        and extra_params is not None
    )

    if not use_filter_result:
        banner_options = item.get("available_banners", [])
        banner_map = {x["label"]: x["key"] for x in banner_options}
        selected_banner_label = st.selectbox(
            "비교 기준", list(banner_map.keys()), key="dashboard_analysis_banner"
        )
        selected_banner_key = banner_map[selected_banner_label]
        result = item["results"][selected_banner_key]
    else:
        st.caption("💡 드릴다운 필터 적용 중: 비교 기준은 '전체'로 고정됩니다.")
        # 필터된 df로 결과 재계산
        ep = extra_params
        if question_type == "multi":
            mr_vars = ep.get("mr_group_map", {}).get(current_qid, [])
            computed = compute_multiresponse_rows(
                df=filtered_df, mr_vars=mr_vars,
                metadata=ep["metadata"], missing_codes=ep["missing_codes"],
                pct_base=ep["pct_base"],
                mr_selected_mode=ep["mr_selected_mode"],
                mr_selected_codes=ep["mr_selected_codes"],
            )
        else:
            computed = compute_single_rows(
                df=filtered_df, var_name=current_qid,
                metadata=ep["metadata"], empty_include=True,
                missing_codes=ep["missing_codes"], pct_base=ep["pct_base"],
            )
        result = {
            "question_id": current_qid,
            "question_text": item.get("display_label", current_qid),
            "question_type": question_type,
            "base": computed["base"],
            "rows": computed["rows"],
            "banner_var": None,
            "banner_rows": [],
        }

    chart_labels = list(DASHBOARD_CHART_OPTIONS.keys())
    default_chart_label = item.get("default_chart", "가로 막대")
    default_chart_index = chart_labels.index(default_chart_label) if default_chart_label in chart_labels else 0
    selected_chart_label = st.selectbox(
        "그래프 유형", chart_labels, index=default_chart_index, key="dashboard_analysis_chart"
    )
    chart_type_key = DASHBOARD_CHART_OPTIONS[selected_chart_label]

    # ── 차트 렌더 ─────────────────────────────────────────────
    fig = render_dashboard_chart(result, chart_type_key)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    # ── 요약 텍스트 ──────────────────────────────────────────
    summary_text = build_dashboard_summary_text(result)
    st.markdown("#### 결과 해석")
    st.caption(summary_text)
    st.code(summary_text, language="text")
    st.download_button(
        "현재 분석 결과 텍스트로 복사용 다운로드",
        data=summary_text.encode("utf-8"),
        file_name=f"{safe_filename(item['question_id'])}_summary.txt",
        mime="text/plain",
        use_container_width=False,
    )

    st.markdown("#### 표 데이터")
    st.dataframe(dashboard_result_to_dataframe(result), use_container_width=True, hide_index=True)

    # ── 드릴다운 필터 설정 UI ────────────────────────────────
    if extra_params and not active_filter:
        st.markdown("---")
        with st.expander("📌 이 응답자만 보기 (드릴다운 필터)", expanded=False):
            st.caption(
                "항목을 선택하면 해당 응답자들 기준으로 다른 문항의 분포를 확인할 수 있습니다. "
                "필터는 상단 배지에서 언제든지 해제할 수 있습니다."
            )
            df_rows = _prepare_rows(result)
            if df_rows.empty or "label" not in df_rows.columns:
                st.info("이 문항은 드릴다운 필터를 지원하지 않습니다.")
            else:
                labels_list = df_rows["label"].tolist()
                pcts_list = df_rows["pct"].tolist()
                ns_list = df_rows["n"].tolist() if "n" in df_rows.columns else [None] * len(labels_list)
                n_cols = min(3, len(labels_list))
                btn_cols = st.columns(n_cols)
                for i, (lbl, pct, n) in enumerate(zip(labels_list, pcts_list, ns_list)):
                    n_text = f" ({int(n):,}명)" if n else ""
                    btn_label = f"**{lbl}**  \n{format_pct(pct)}%{n_text}"
                    with btn_cols[i % n_cols]:
                        if st.button(btn_label, key=f"drilldown_btn_{i}", use_container_width=True):
                            st.session_state["_drilldown_filter"] = {
                                "question_id": current_qid,
                                "var_label": item.get("display_label", current_qid),
                                "label": lbl,
                            }
                            st.rerun()


def _render_relation_tab(bundle: dict):
    relation = bundle.get("relation", {})
    corr = relation.get("correlation")

    st.subheader("주요 지표 간 상관관계 분석")
    st.caption("색상이 짙을수록 관계가 크며, 값이 1 또는 -1에 가까울수록 상관이 강합니다.")

    if not corr:
        st.info("지표 관계를 표시할 척도형 문항이 2개 이상 필요합니다.")
        return

    labels = corr.get("labels", [])
    vars_ = corr.get("vars", [])
    values = corr.get("values", [])
    heatmap_df = pd.DataFrame(values, index=labels, columns=labels)

    fig = px.imshow(
        heatmap_df,
        text_auto=".2f",
        color_continuous_scale="Blues",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", height=max(420, 90 + len(labels) * 34), margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if len(vars_) >= 2:
        var_label_map = {labels[i]: vars_[i] for i in range(min(len(labels), len(vars_)))}
        sc1, sc2 = st.columns(2)
        with sc1:
            x_label = st.selectbox("산점도 X축", labels, index=0, key="dashboard_relation_x")
        with sc2:
            y_default = 1 if len(labels) > 1 else 0
            y_label = st.selectbox("산점도 Y축", labels, index=y_default, key="dashboard_relation_y")

        if x_label != y_label:
            st.markdown("#### 산점도")
            x_var = var_label_map[x_label]
            y_var = var_label_map[y_label]
            plot_df = pd.DataFrame({
                x_label: pd.to_numeric(st.session_state.get("_dashboard_relation_df", pd.DataFrame()).get(x_var, pd.Series(dtype=float)), errors="coerce"),
                y_label: pd.to_numeric(st.session_state.get("_dashboard_relation_df", pd.DataFrame()).get(y_var, pd.Series(dtype=float)), errors="coerce"),
            })
            if plot_df.empty:
                st.caption("현재 미리보기 세션에서는 산점도 원데이터가 없어 패키지 번들 기준 heatmap만 표시합니다.")


def render_dashboard_preview(
    config_df: pd.DataFrame,
    df: pd.DataFrame,
    metadata: dict,
    profile_vars: list,
    missing_codes: list,
    pct_base: str,
    mr_group_map: dict,
    mr_selected_mode: str,
    mr_selected_codes: list,
    missing_rules_by_var: dict | None = None,
):
    bundle = build_dashboard_bundle(
        config_df=config_df,
        df=df,
        metadata=metadata,
        profile_vars=profile_vars,
        missing_codes=missing_codes,
        pct_base=pct_base,
        mr_group_map=mr_group_map,
        mr_selected_mode=mr_selected_mode,
        mr_selected_codes=mr_selected_codes,
        app_title="고객용 대시보드 프리뷰",
        app_subtitle="Overview / Analysis / Relation 구조 미리보기",
        preferred_subdomain="dashboard-preview",
        missing_rules_by_var=missing_rules_by_var,
    )

    st.session_state["_dashboard_relation_df"] = _coerce_numeric_df(df, _infer_scale_vars(config_df, df))
    # 드릴다운 필터 초기화 (최초 진입 시)
    if "_drilldown_filter" not in st.session_state:
        st.session_state["_drilldown_filter"] = None

    extra_params = {
        "df": df,
        "metadata": metadata,
        "missing_codes": missing_codes,
        "pct_base": pct_base,
        "mr_group_map": mr_group_map,
        "mr_selected_mode": mr_selected_mode,
        "mr_selected_codes": mr_selected_codes,
    }

    st.markdown("## 📊 고객용 대시보드 프리뷰")
    tab_ov, tab_an, tab_rel = st.tabs(["📌 조사 개요", "🔍 문항 분석", "🔗 지표 관계"])

    with tab_ov:
        _render_overview_tab(bundle)
    with tab_an:
        _render_analysis_tab(bundle, extra_params=extra_params)
    with tab_rel:
        _render_relation_tab(bundle)
