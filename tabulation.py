import re

import pandas as pd

from utils import (
    filter_valid_series,
    safe_numeric,
    fmt_pct,
    fmt_num,
    is_missing_value,
    make_subtotal_col_name,
    guess_rank_group_name,
    parse_subtotal_groups,
)
from config import DEFAULT_SUBTOTAL_TEXT
from metadata import get_var_label, get_value_label
from display_format import format_pct




def _build_unique_label_map(pairs: list[tuple]) -> dict:
    """
    [(key, label)] -> {key: unique_label}
    같은 표시명이 반복되면 코드/변수명을 덧붙여 충돌을 방지한다.
    """
    counts = {}
    for _, label in pairs:
        base = str(label).strip()
        counts[base] = counts.get(base, 0) + 1

    used = set()
    label_map = {}
    for key, label in pairs:
        base = str(label).strip() or str(key)
        if counts.get(base, 0) <= 1 and base not in used:
            unique = base
        else:
            unique = f"{base} [{key}]"
            suffix = 2
            while unique in used:
                unique = f"{base} [{key}-{suffix}]"
                suffix += 1
        used.add(unique)
        label_map[key] = unique
    return label_map

def _get_var_missing_rule(var_name: str, missing_rules_by_var: dict | None):
    if not missing_rules_by_var:
        return None
    return missing_rules_by_var.get(str(var_name))


def _normalize_code_for_compare(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except Exception:
        return text


def _build_code_membership_mask(series: pd.Series, codes: list) -> pd.Series:
    if series is None or len(series) == 0 or not codes:
        return pd.Series(False, index=getattr(series, "index", None))
    normalized_targets = {_normalize_code_for_compare(code) for code in codes}
    normalized_targets.discard("")
    if not normalized_targets:
        return pd.Series(False, index=series.index)
    return series.map(_normalize_code_for_compare).isin(normalized_targets).fillna(False)


def _resolve_rank_group_vars(
    dep_var: str,
    df: pd.DataFrame,
    question_type_map: dict | None = None,
) -> list[str]:
    question_type_map = question_type_map or {}
    base = guess_rank_group_name(dep_var)
    vars_ = []

    for col in df.columns:
        if guess_rank_group_name(col) != base:
            continue
        if question_type_map and question_type_map.get(col) != "순위형":
            continue
        vars_.append(str(col))

    def _sort_key(name: str):
        m = re.search(r"_(\d+)$", str(name))
        if m:
            return (0, int(m.group(1)), str(name))
        return (1, 999999, str(name))

    vars_ = sorted(set(vars_), key=_sort_key)
    return vars_ if len(vars_) >= 2 else [dep_var]


def _get_rank_categories_from_group(
    df: pd.DataFrame,
    rank_vars: list[str],
    metadata: dict,
    empty_include: bool,
    missing_codes: list,
    missing_rules_by_var: dict | None = None,
):
    if not rank_vars:
        return []

    first_var = rank_vars[0]
    value_map = metadata.get(first_var, {}).get("value_labels", {}) or {}

    if empty_include and value_map:
        cats = list(value_map.keys())
    else:
        combined = []
        for var in rank_vars:
            if var not in df.columns:
                continue
            valid = filter_valid_series(
                df[var],
                missing_codes,
                sav_rule=_get_var_missing_rule(var, missing_rules_by_var),
            )
            combined.extend(valid.dropna().tolist())

        numeric_values = []
        non_numeric_values = []

        for v in combined:
            try:
                fv = float(v)
                if fv.is_integer():
                    numeric_values.append(int(fv))
                else:
                    numeric_values.append(float(fv))
            except Exception:
                non_numeric_values.append(str(v))

        cats = sorted(set(numeric_values), key=lambda x: float(x)) + sorted(set(non_numeric_values))

    cleaned = []
    for c in cats:
        if not is_missing_value(
            c,
            missing_codes,
            sav_rule=_get_var_missing_rule(first_var, missing_rules_by_var),
        ):
            cleaned.append(c)

    return cleaned


def compute_ranking_distribution(
    df: pd.DataFrame,
    rank_vars: list[str],
    metadata: dict,
    empty_include: bool,
    missing_codes: list,
    rank_top_k: int,
    missing_rules_by_var: dict | None = None,
    weight_col: str | None = None,
):
    """
    Computes distribution for ranking-type questions (Point 10).
    Supports 1st rank %, Top-K %, and Mean Rank with full weight support (Point 1).
    """
    rank_vars = [v for v in rank_vars if v in df.columns]

    categories = _get_rank_categories_from_group(
        df=df,
        rank_vars=rank_vars,
        metadata=metadata,
        empty_include=empty_include,
        missing_codes=missing_codes,
        missing_rules_by_var=missing_rules_by_var,
    )

    # Valid mask: respondent must have at least one non-missing rank response
    valid_mask = pd.Series(False, index=df.index)
    for var in rank_vars:
        sav_rule = _get_var_missing_rule(var, missing_rules_by_var)
        valid_mask |= ~df[var].apply(lambda x: is_missing_value(x, missing_codes, sav_rule=sav_rule))

    stats = _get_n_stats(df, valid_mask, weight_col)
    valid_df = df[valid_mask].copy()
    valid_n = stats["valid_n"]

    top_k = max(1, min(int(rank_top_k or 2), len(rank_vars) if rank_vars else 1))
    rows = []

    for cat in categories:
        label = get_value_label(rank_vars[0], cat, metadata) if rank_vars else str(cat)
        
        # Track counts
        first_count = 0.0
        top_k_count = 0.0
        weighted_rank_sum = 0.0
        total_times_ranked = 0.0

        for idx, var in enumerate(rank_vars, start=1):
            series = valid_df[var]
            try:
                # Use float comparison if possible
                mask = safe_numeric(series) == float(cat)
            except Exception:
                mask = series.astype(str) == str(cat)

            count = _weighted_sum_for_mask(valid_df, mask, weight_col)
            
            if idx == 1:
                first_count = count
            if idx <= top_k:
                top_k_count += count
                
            if count > 0:
                weighted_rank_sum += idx * count
                total_times_ranked += count

        # Percentages (Point 8: Return as floats)
        first_pct = (first_count / valid_n * 100) if valid_n > 0 else None
        top_k_pct = (top_k_count / valid_n * 100) if valid_n > 0 else None
        mean_rank = (weighted_rank_sum / total_times_ranked) if total_times_ranked > 0 else None

        rows.append({
            "code": cat,
            "label": label,
            "first_n": _format_weighted_n(first_count),
            "first_pct": first_pct,
            "top_k_n": _format_weighted_n(top_k_count),
            "top_k_pct": top_k_pct,
            "mean_rank": mean_rank,
        })

    return {
        "base": _format_weighted_n(valid_n),
        "total_n": _format_weighted_n(stats["total_n"]),
        "valid_n": _format_weighted_n(stats["valid_n"]),
        "unweighted_total_n": stats["unweighted_total_n"],
        "unweighted_valid_n": stats["unweighted_valid_n"],
        "rows": rows,
        "rank_var_count": len(rank_vars),
        "top_k": top_k,
    }


# =========================================================
# 1. 기본 카테고리 처리
# =========================================================
def get_categories(
    series: pd.Series,
    var_name: str,
    metadata: dict,
    empty_include: bool,
    missing_codes: list,
    missing_rules_by_var: dict | None = None,
):
    value_map = metadata.get(var_name, {}).get("value_labels", {})

    if empty_include and value_map:
        cats = list(value_map.keys())
    else:
        valid = filter_valid_series(
            series,
            missing_codes,
            sav_rule=_get_var_missing_rule(var_name, missing_rules_by_var),
        )
        values = pd.Series(valid.dropna().unique()).tolist()

        numeric_values = []
        non_numeric_values = []

        for v in values:
            try:
                fv = float(v)
                if fv.is_integer():
                    numeric_values.append(int(fv))
                else:
                    numeric_values.append(float(fv))
            except Exception:
                non_numeric_values.append(str(v))

        cats = sorted(set(numeric_values), key=lambda x: float(x)) + sorted(set(non_numeric_values))

    cleaned = []
    for c in cats:
        if not is_missing_value(
            c,
            missing_codes,
            sav_rule=_get_var_missing_rule(var_name, missing_rules_by_var),
        ):
            cleaned.append(c)

    return cleaned




def _get_weight_series(df: pd.DataFrame, weight_col: str | None = None) -> pd.Series:
    """
    Returns a numeric weight series for the dataframe.
    - If weight_col exists: converts to numeric, fills NaN with 0, clips negative values.
    - If weight_col is None: returns a series of 1.0s.
    """
    if weight_col and weight_col in df.columns:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
        w = w.clip(lower=0)
        return w.astype(float)
    return pd.Series(1.0, index=df.index, dtype="float64")


def _weighted_sum_for_mask(df: pd.DataFrame, mask: pd.Series, weight_col: str | None = None) -> float:
    """Calculates the sum of weights for a given boolean mask."""
    if mask is None or len(df) == 0:
        return 0.0
    # Ensure mask aligns with df index
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    weights = _get_weight_series(df, weight_col)
    return float(weights[mask].sum())


def _get_n_stats(df: pd.DataFrame, valid_mask: pd.Series, weight_col: str | None = None) -> dict:
    """
    Returns a comprehensive set of N-statistics (weighted and unweighted).
    Useful for Point 2: Clear separation of N-types.

    valid_mask may be created from a filtered Series. In that case its index
    can be a subset of df.index, so align it safely before indexing weights.
    """
    weights = _get_weight_series(df, weight_col)

    if valid_mask is None:
        aligned_mask = pd.Series(False, index=df.index)
    else:
        aligned_mask = pd.Series(valid_mask)
        if not aligned_mask.index.equals(df.index):
            aligned_mask = aligned_mask.reindex(df.index, fill_value=False)
        aligned_mask = aligned_mask.fillna(False).astype(bool)

    # Unweighted
    unweighted_total = len(df)
    unweighted_valid = int(aligned_mask.sum())

    # Weighted
    weighted_total = float(weights.sum())
    weighted_valid = float(weights[aligned_mask].sum())

    return {
        "unweighted_total_n": unweighted_total,
        "unweighted_valid_n": unweighted_valid,
        "weighted_total_n": weighted_total,
        "weighted_valid_n": weighted_valid,
        "total_n": weighted_total,  # Default for display/calc
        "valid_n": weighted_valid,  # Default for display/calc
    }


def _format_weighted_n(value: float):
    """
    Formats N for internal storage. Returns int if whole number, else float.
    """
    if value is None or pd.isna(value):
        return 0
    value = float(value)
    if abs(value - round(value)) < 1e-7:
        return int(round(value))
    return round(value, 4)


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float | None:
    """Calculates weighted mean, handling NaN values in either series."""
    # Align and drop NaNs
    combined = pd.concat([series, weights], axis=1).dropna()
    if combined.empty:
        return None
    vals = combined.iloc[:, 0]
    ws = combined.iloc[:, 1]
    denom = ws.sum()
    if denom == 0:
        return None
    return float((vals * ws).sum() / denom)


def _weighted_std(series: pd.Series, weights: pd.Series) -> float | None:
    """Calculates weighted standard deviation."""
    combined = pd.concat([series, weights], axis=1).dropna()
    if combined.empty:
        return None
    vals = combined.iloc[:, 0]
    ws = combined.iloc[:, 1]
    
    mean = _weighted_mean(vals, ws)
    if mean is None:
        return None
    
    denom = ws.sum()
    if denom <= 0:
        return None
        
    # Using the weighted variance formula
    variance = (ws * (vals - mean)**2).sum() / denom
    return float(variance**0.5)


def _calculate_index_score(mean: float | None, scale_type: int) -> float | None:
    """
    Point 2: Convert scale mean to 100-point index.
    5-pt: (mean - 1) / 4 * 100
    10-pt: mean / 10 * 100
    """
    if mean is None:
        return None
    if scale_type == 5:
        return (mean - 1) / 4 * 100
    if scale_type == 10:
        return mean / 10 * 100
    return None


# =========================================================
# 2. 공통 집계 엔진
# =========================================================
def compute_single_distribution(
    df: pd.DataFrame,
    var_name: str,
    metadata: dict,
    empty_include: bool,
    missing_codes: list,
    pct_base: str,
    missing_rules_by_var: dict | None = None,
    weight_col: str | None = None,
):
    """
    Computes frequency distribution for a single categorical/scale variable.
    """
    categories = get_categories(
        df[var_name],
        var_name,
        metadata,
        empty_include,
        missing_codes,
        missing_rules_by_var=missing_rules_by_var,
    )

    # Missing handling
    sav_rule = _get_var_missing_rule(var_name, missing_rules_by_var)
    valid_mask = ~df[var_name].apply(lambda x: is_missing_value(x, missing_codes, sav_rule=sav_rule))
    
    # N Stats (Point 2)
    stats = _get_n_stats(df, valid_mask, weight_col)
    
    # Determine denominator (Point 3)
    denom = stats["valid_n"] if pct_base == "valid" else stats["total_n"]

    rows = []
    valid_df = df[valid_mask].copy()
    numeric_series = safe_numeric(valid_df[var_name])

    for cat in categories:
        label = get_value_label(var_name, cat, metadata)

        # Build mask for this category within the valid subset
        try:
            # Try numeric comparison first
            cat_mask = numeric_series == float(cat)
        except Exception:
            # Fallback to string comparison
            cat_mask = valid_df[var_name].astype(str) == str(cat)

        # Weighted count (Point 1)
        count = _weighted_sum_for_mask(valid_df, cat_mask, weight_col)
        
        # Percentage (Point 8: keep as float)
        pct = (count / denom * 100) if denom > 0 else None

        rows.append({
            "code": cat,
            "label": label,
            "n": _format_weighted_n(count),
            "pct": pct,  # Return as float
        })

    return {
        "base": _format_weighted_n(denom),
        "total_n": _format_weighted_n(stats["total_n"]),
        "valid_n": _format_weighted_n(stats["valid_n"]),
        "unweighted_total_n": stats["unweighted_total_n"],
        "unweighted_valid_n": stats["unweighted_valid_n"],
        "rows": rows,
    }


# =========================================================
# 3. 트리형 배너 경로/깊이 처리
# =========================================================
def normalize_banner_tree_for_tabulation(banner_tree: list) -> list:
    if not banner_tree:
        return []

    def _walk(nodes: list):
        out = []
        for node in nodes:
            var = str(node.get("var", "")).strip()
            label = str(node.get("label", "")).strip() or var
            children = _walk(node.get("children", []))

            if var:
                out.append(
                    {
                        "var": var,
                        "label": label,
                        "children": children,
                    }
                )
        return out

    return _walk(banner_tree)


def extract_banner_var_paths(banner_tree: list) -> list:
    banner_tree = normalize_banner_tree_for_tabulation(banner_tree)
    if not banner_tree:
        return []

    all_paths = []

    def _walk(node: dict, path: list):
        new_path = path + [node["var"]]
        all_paths.append(new_path)

        for child in node.get("children", []):
            _walk(child, new_path)

    for root in banner_tree:
        _walk(root, [])

    deduped = []
    seen = set()
    for p in all_paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped


def build_banner_tree(paths: list):
    tree = {}

    for path in paths:
        current_level = tree
        for i, var in enumerate(path):
            if var not in current_level:
                current_level[var] = {
                    "_end": False,
                    "_children": {},
                }

            if i == len(path) - 1:
                current_level[var]["_end"] = True

            current_level = current_level[var]["_children"]

    return tree


def get_banner_max_depth(paths: list):
    if not paths:
        return 0
    return max(len(p) for p in paths)


def build_banner_level_columns(paths: list):
    depth = get_banner_max_depth(paths)
    n_cols = max(1, depth * 2)
    return [f"구분{i+1}" for i in range(n_cols)]


def _filter_df_by_category(df: pd.DataFrame, var: str, cat):
    try:
        mask = safe_numeric(df[var]) == float(cat)
    except Exception:
        mask = df[var].astype(str) == str(cat)
    return df[mask].copy()


def generate_banner_rows(
    df: pd.DataFrame,
    banner_tree: list,
    metadata: dict,
    empty_include: bool,
    missing_codes: list,
    missing_rules_by_var: dict | None = None,
):
    banner_paths = extract_banner_var_paths(banner_tree)
    if not banner_paths:
        return []

    tree = build_banner_tree(banner_paths)
    rows = []

    def walk(current_df: pd.DataFrame, subtree: dict, labels_so_far: list):
        for var, node in subtree.items():
            if var not in current_df.columns:
                continue

            var_label = get_var_label(var, metadata)
            cats = get_categories(
                current_df[var],
                var,
                metadata,
                empty_include,
                missing_codes,
                missing_rules_by_var=missing_rules_by_var,
            )

            for cat in cats:
                sub_df = _filter_df_by_category(current_df, var, cat)
                if len(sub_df) == 0:
                    continue

                cat_label = get_value_label(var, cat, metadata)
                new_labels = labels_so_far + [var_label, cat_label]

                if node.get("_end", False):
                    rows.append(
                        {
                            "labels": new_labels,
                            "sub_df": sub_df,
                        }
                    )

                child_tree = node.get("_children", {})
                if child_tree:
                    walk(sub_df, child_tree, new_labels)

    walk(df, tree, [])
    return rows


# =========================================================
# 4. 한 집단 통계 계산
# =========================================================
def summarize_one_group(
    sub_df: pd.DataFrame,
    dep_var: str,
    dep_categories: list,
    metadata: dict,
    show_n: bool,
    show_pct: bool,
    pct_display_mode: str,
    show_subtotal: bool,
    subtotal_groups: list,
    show_mean: bool,
    show_std: bool,
    pct_base: str,
    scale_vars: list,
    missing_codes: list,
    missing_rules_by_var: dict | None = None,
    dep_label_map: dict | None = None,
    weight_col: str | None = None,
):
    row = {}
    dep_label_map = dep_label_map or {cat: get_value_label(dep_var, cat, metadata) for cat in dep_categories}

    summary = compute_single_distribution(
        df=sub_df,
        var_name=dep_var,
        metadata=metadata,
        empty_include=False,
        missing_codes=missing_codes,
        pct_base=pct_base,
        missing_rules_by_var=missing_rules_by_var,
        weight_col=weight_col,
    )

    total_n = summary["total_n"]
    valid_n = summary["valid_n"]
    valid_series = filter_valid_series(
        sub_df[dep_var],
        missing_codes,
        sav_rule=_get_var_missing_rule(dep_var, missing_rules_by_var),
    )
    numeric_valid = safe_numeric(valid_series)
    row_map = {r["label"]: r for r in summary["rows"]}

    if show_n:
        row["N"] = _format_weighted_n(valid_n)

    if show_pct:
        for cat in dep_categories:
            source_label = get_value_label(dep_var, cat, metadata)
            label = dep_label_map.get(cat, source_label)
            item = row_map.get(source_label, {"n": 0, "pct": None})
            cat_count = int(item["n"])
            pct = item["pct"]

            if pct_display_mode == "비율만":
                row[f"{label}_%"] = fmt_pct(pct)
            elif pct_display_mode == "응답수+비율(두열)":
                row[f"{label}_N"] = cat_count
                row[f"{label}_%"] = fmt_pct(pct)
            elif pct_display_mode == "응답수(비율)한셀":
                pct_str = "-" if pct is None else format_pct(pct)
                row[label] = f"{cat_count} ({pct_str})"

    if show_subtotal and subtotal_groups:
        for group in subtotal_groups:
            # group is now a dict {'label': ..., 'codes': [...]}
            codes = group.get("codes", [])
            mask = _build_code_membership_mask(valid_series, codes)
            cat_count = _weighted_sum_for_mask(sub_df.loc[valid_series.index], mask, weight_col)

            denom = valid_n if pct_base == "valid" else total_n
            pct = (cat_count / denom * 100) if denom > 0 else None
            
            # Point 6: Improved Subtotal Labels
            col_name = make_subtotal_col_name(group)
            
            if pct_display_mode == "비율만":
                row[f"{col_name}_%"] = fmt_pct(pct)
            elif pct_display_mode == "응답수+비율(두열)":
                row[f"{col_name}_N"] = _format_weighted_n(cat_count)
                row[f"{col_name}_%"] = fmt_pct(pct)
            elif pct_display_mode == "응답수(비율)한셀":
                pct_str = "-" if pct is None else format_pct(pct)
                row[col_name] = f"{_format_weighted_n(cat_count)} ({pct_str})"

    # Point 7: Mean/StdDev (Scale only)
    if dep_var in scale_vars or str(metadata.get(dep_var, {}).get("qtn_type", "")).strip() == "척도형":
        weights = _get_weight_series(sub_df.loc[valid_series.index], weight_col)
        if show_mean:
            mean_val = _weighted_mean(numeric_valid, weights)
            row["평균"] = fmt_num(mean_val) if mean_val is not None else None
        if show_std:
            std_val = _weighted_std(numeric_valid, weights)
            row["표준편차"] = fmt_num(std_val) if std_val is not None else None

    return row


# =========================================================
# 5. 단일/척도 문항 통계표 생성
# =========================================================
def build_block_table(
    df: pd.DataFrame,
    dep_var: str,
    banner_tree: list,
    metadata: dict,
    include_total: bool,
    empty_include: bool,
    show_n: bool,
    show_pct: bool,
    pct_display_mode: str,
    show_subtotal: bool,
    subtotal_groups: list,
    exclude_subtotal_vars: list,
    show_mean: bool,
    show_std: bool,
    pct_base: str,
    scale_vars: list,
    missing_codes: list,
    missing_rules_by_var: dict | None = None,
    question_type: str = "범주형",
    weight_col: str | None = None,
):
    dep_categories = get_categories(
        df[dep_var],
        dep_var,
        metadata,
        empty_include,
        missing_codes,
        missing_rules_by_var=missing_rules_by_var,
    )
    dep_label_map = _build_unique_label_map(
        [(cat, get_value_label(dep_var, cat, metadata)) for cat in dep_categories]
    )

    banner_paths = extract_banner_var_paths(banner_tree)
    normalized_question_type = str(question_type or "").strip().lower()
    subtotal_allowed_types = {"", "범주형", "척도형", "단일", "single", "scale"}
    use_subtotal = (
        show_subtotal
        and (dep_var not in exclude_subtotal_vars)
        and (
            question_type in {"범주형", "척도형", "단일"}
            or normalized_question_type in subtotal_allowed_types
            or dep_var in scale_vars
        )
    )

    if use_subtotal and not subtotal_groups:
        scale_length = len(dep_categories)
        if scale_length == 5:
            subtotal_groups = parse_subtotal_groups(DEFAULT_SUBTOTAL_TEXT)

    show_mean = show_mean and (dep_var in scale_vars or question_type == "척도형")
    show_std = show_std and (dep_var in scale_vars or question_type == "척도형")

    rows = []
    group_cols = build_banner_level_columns(banner_paths)

    if include_total:
        stat_row = summarize_one_group(
            sub_df=df,
            dep_var=dep_var,
            dep_categories=dep_categories,
            metadata=metadata,
            show_n=show_n,
            show_pct=show_pct,
            pct_display_mode=pct_display_mode,
            show_subtotal=use_subtotal,
            subtotal_groups=subtotal_groups,
            show_mean=show_mean,
            show_std=show_std,
            pct_base=pct_base,
            scale_vars=scale_vars,
            missing_codes=missing_codes,
            missing_rules_by_var=missing_rules_by_var,
            dep_label_map=dep_label_map,
            weight_col=weight_col,
        )
        total_prefix = ["전체"] + [""] * (len(group_cols) - 1)
        rows.append({group_cols[i]: total_prefix[i] for i in range(len(group_cols))} | stat_row)

    banner_rows = generate_banner_rows(
        df=df,
        banner_tree=banner_tree,
        metadata=metadata,
        empty_include=empty_include,
        missing_codes=missing_codes,
        missing_rules_by_var=missing_rules_by_var,
    )

    for item in banner_rows:
        labels = item["labels"]
        sub_df = item["sub_df"]

        stat_row = summarize_one_group(
            sub_df=sub_df,
            dep_var=dep_var,
            dep_categories=dep_categories,
            metadata=metadata,
            show_n=show_n,
            show_pct=show_pct,
            pct_display_mode=pct_display_mode,
            show_subtotal=use_subtotal,
            subtotal_groups=subtotal_groups,
            show_mean=show_mean,
            show_std=show_std,
            pct_base=pct_base,
            scale_vars=scale_vars,
            missing_codes=missing_codes,
            missing_rules_by_var=missing_rules_by_var,
            dep_label_map=dep_label_map,
            weight_col=weight_col,
        )

        prefix = labels + [""] * (len(group_cols) - len(labels))
        rows.append({group_cols[i]: prefix[i] for i in range(len(group_cols))} | stat_row)

    result = pd.DataFrame(rows)
    ordered_group_cols = group_cols
    ordered_stat_cols = []

    if show_n and "N" in result.columns:
        ordered_stat_cols.append("N")

    if show_pct:
        if pct_display_mode == "비율만":
            for cat in dep_categories:
                label = dep_label_map[cat]
                col = f"{label}_%"
                if col in result.columns:
                    ordered_stat_cols.append(col)

            if use_subtotal:
                for g in subtotal_groups:
                    col = f"{make_subtotal_col_name(g)}_%"
                    if col in result.columns:
                        ordered_stat_cols.append(col)

        elif pct_display_mode == "응답수+비율(두열)":
            for cat in dep_categories:
                label = dep_label_map[cat]
                col_n = f"{label}_N"
                col_pct = f"{label}_%"
                if col_n in result.columns:
                    ordered_stat_cols.append(col_n)
                if col_pct in result.columns:
                    ordered_stat_cols.append(col_pct)

            if use_subtotal:
                for g in subtotal_groups:
                    base = make_subtotal_col_name(g)
                    col_n = f"{base}_N"
                    col_pct = f"{base}_%"
                    if col_n in result.columns:
                        ordered_stat_cols.append(col_n)
                    if col_pct in result.columns:
                        ordered_stat_cols.append(col_pct)

        elif pct_display_mode == "응답수(비율)한셀":
            for cat in dep_categories:
                label = dep_label_map[cat]
                if label in result.columns:
                    ordered_stat_cols.append(label)

            if use_subtotal:
                for g in subtotal_groups:
                    col = make_subtotal_col_name(g)
                    if col in result.columns:
                        ordered_stat_cols.append(col)

    if dep_var in scale_vars or question_type == "척도형":
        if show_mean and "평균" in result.columns:
            ordered_stat_cols.append("평균")
        if show_std and "표준편차" in result.columns:
            ordered_stat_cols.append("표준편차")

    if result.empty:
        return pd.DataFrame(columns=ordered_group_cols + ordered_stat_cols)

    return result[ordered_group_cols + ordered_stat_cols]


# =========================================================
# 6. 다중응답 유틸
# =========================================================
def get_multiresponse_valid_mask(
    df: pd.DataFrame,
    mr_vars: list,
    missing_codes: list,
    missing_rules_by_var: dict | None = None,
):
    if not mr_vars:
        return pd.Series(False, index=df.index)

    valid_mask = pd.Series(False, index=df.index)

    for var in mr_vars:
        valid_mask = valid_mask | (
            ~df[var].apply(
                lambda x: is_missing_value(
                    x,
                    missing_codes,
                    sav_rule=_get_var_missing_rule(var, missing_rules_by_var),
                )
            )
        )

    return valid_mask


def is_selected_multiresponse(value, selected_mode: str, selected_codes: list):
    if pd.isna(value):
        return False

    if selected_mode == "값 있으면 선택":
        return True

    for code in selected_codes:
        try:
            if float(value) == float(code):
                return True
        except Exception:
            if str(value).strip() == str(code).strip():
                return True
    return False


def compute_multiresponse_distribution(
    df: pd.DataFrame,
    mr_vars: list,
    metadata: dict,
    missing_codes: list,
    pct_base: str,
    mr_selected_mode: str,
    mr_selected_codes: list,
    missing_rules_by_var: dict | None = None,
    weight_col: str | None = None,
):
    """
    Computes distribution for multi-response questions (Point 9).
    Base options: 'valid' (at least one option picked) or 'total' (all respondents).
    """
    # Build mask for respondents who provided any valid response to the set
    valid_mask = get_multiresponse_valid_mask(
        df,
        mr_vars,
        missing_codes,
        missing_rules_by_var=missing_rules_by_var,
    )
    
    stats = _get_n_stats(df, valid_mask, weight_col)
    denom = stats["valid_n"] if pct_base == "valid" else stats["total_n"]
    
    valid_df = df[valid_mask].copy()

    rows = []
    for mr_var in mr_vars:
        label = get_var_label(mr_var, metadata)
        selected_mask = valid_df[mr_var].apply(
            lambda x: is_selected_multiresponse(x, mr_selected_mode, mr_selected_codes)
        )
        
        # Weighted count of picks for this option
        count = _weighted_sum_for_mask(valid_df, selected_mask, weight_col)
        
        pct = (count / denom * 100) if denom > 0 else None

        rows.append({
            "label": label,
            "n": _format_weighted_n(count),
            "pct": pct,  # Return as float
        })

    return {
        "base": _format_weighted_n(denom),
        "total_n": _format_weighted_n(stats["total_n"]),
        "valid_n": _format_weighted_n(stats["valid_n"]),
        "unweighted_total_n": stats["unweighted_total_n"],
        "unweighted_valid_n": stats["unweighted_valid_n"],
        "rows": rows,
    }


def summarize_one_group_multiresponse(
    sub_df: pd.DataFrame,
    mr_vars: list,
    metadata: dict,
    show_n: bool,
    show_pct: bool,
    pct_display_mode: str,
    pct_base: str,
    missing_codes: list,
    mr_selected_mode: str,
    mr_selected_codes: list,
    missing_rules_by_var: dict | None = None,
    mr_label_map: dict | None = None,
    weight_col: str | None = None,
):
    row = {}
    mr_label_map = mr_label_map or _build_unique_label_map([(var, get_var_label(var, metadata)) for var in mr_vars])

    summary = compute_multiresponse_distribution(
        df=sub_df,
        mr_vars=mr_vars,
        metadata=metadata,
        missing_codes=missing_codes,
        pct_base=pct_base,
        mr_selected_mode=mr_selected_mode,
        mr_selected_codes=mr_selected_codes,
        missing_rules_by_var=missing_rules_by_var,
        weight_col=weight_col,
    )
    valid_n = summary["valid_n"]
    row_map = {r["label"]: r for r in summary["rows"]}

    if show_n:
        row["N"] = _format_weighted_n(valid_n)

    if show_pct:
        for mr_var in mr_vars:
            source_label = get_var_label(mr_var, metadata)
            label = mr_label_map.get(mr_var, source_label)
            item = row_map.get(source_label, {"n": 0, "pct": None})
            count = int(item["n"])
            pct = item["pct"]

            if pct_display_mode == "비율만":
                row[f"{label}_%"] = fmt_pct(pct)
            elif pct_display_mode == "응답수+비율(두열)":
                row[f"{label}_N"] = _format_weighted_n(count)
                row[f"{label}_%"] = fmt_pct(pct)
            elif pct_display_mode == "응답수(비율)한셀":
                pct_str = "-" if pct is None else format_pct(pct)
                row[label] = f"{_format_weighted_n(count)} ({pct_str})"

    return row


# =========================================================
# 7. 다중응답 통계표 생성
# =========================================================
def build_multiresponse_table(
    df: pd.DataFrame,
    mr_group_name: str,
    mr_vars: list,
    banner_tree: list,
    metadata: dict,
    include_total: bool,
    show_n: bool,
    show_pct: bool,
    pct_display_mode: str,
    pct_base: str,
    missing_codes: list,
    mr_selected_mode: str,
    mr_selected_codes: list,
    empty_include: bool,
    missing_rules_by_var: dict | None = None,
    weight_col: str | None = None,
):
    rows = []
    banner_paths = extract_banner_var_paths(banner_tree)
    group_cols = build_banner_level_columns(banner_paths)
    mr_label_map = _build_unique_label_map([(mr_var, get_var_label(mr_var, metadata)) for mr_var in mr_vars])

    if include_total:
        stat_row = summarize_one_group_multiresponse(
            sub_df=df,
            mr_vars=mr_vars,
            metadata=metadata,
            show_n=show_n,
            show_pct=show_pct,
            pct_display_mode=pct_display_mode,
            pct_base=pct_base,
            missing_codes=missing_codes,
            mr_selected_mode=mr_selected_mode,
            mr_selected_codes=mr_selected_codes,
            missing_rules_by_var=missing_rules_by_var,
            mr_label_map=mr_label_map,
            weight_col=weight_col,
        )
        total_prefix = ["전체"] + [""] * (len(group_cols) - 1)
        rows.append({group_cols[i]: total_prefix[i] for i in range(len(group_cols))} | stat_row)

    banner_rows = generate_banner_rows(
        df=df,
        banner_tree=banner_tree,
        metadata=metadata,
        empty_include=empty_include,
        missing_codes=missing_codes,
        missing_rules_by_var=missing_rules_by_var,
    )

    for item in banner_rows:
        labels = item["labels"]
        sub_df = item["sub_df"]

        stat_row = summarize_one_group_multiresponse(
            sub_df=sub_df,
            mr_vars=mr_vars,
            metadata=metadata,
            show_n=show_n,
            show_pct=show_pct,
            pct_display_mode=pct_display_mode,
            pct_base=pct_base,
            missing_codes=missing_codes,
            mr_selected_mode=mr_selected_mode,
            mr_selected_codes=mr_selected_codes,
            missing_rules_by_var=missing_rules_by_var,
            mr_label_map=mr_label_map,
            weight_col=weight_col,
        )

        prefix = labels + [""] * (len(group_cols) - len(labels))
        rows.append({group_cols[i]: prefix[i] for i in range(len(group_cols))} | stat_row)

    result = pd.DataFrame(rows)
    ordered_stat_cols = []

    if show_n and "N" in result.columns:
        ordered_stat_cols.append("N")

    if show_pct:
        if pct_display_mode == "비율만":
            for mr_var in mr_vars:
                label = mr_label_map[mr_var]
                col = f"{label}_%"
                if col in result.columns:
                    ordered_stat_cols.append(col)

        elif pct_display_mode == "응답수+비율(두열)":
            for mr_var in mr_vars:
                label = mr_label_map[mr_var]
                col_n = f"{label}_N"
                col_pct = f"{label}_%"
                if col_n in result.columns:
                    ordered_stat_cols.append(col_n)
                if col_pct in result.columns:
                    ordered_stat_cols.append(col_pct)

        elif pct_display_mode == "응답수(비율)한셀":
            for mr_var in mr_vars:
                label = mr_label_map[mr_var]
                if label in result.columns:
                    ordered_stat_cols.append(label)

    if result.empty:
        return pd.DataFrame(columns=group_cols + ordered_stat_cols)

    return result[group_cols + ordered_stat_cols]


# =========================================================
# 8. 응답자 특성표
# =========================================================
def build_profile_table(
    df: pd.DataFrame,
    profile_vars: list,
    metadata: dict,
    empty_include: bool,
    missing_codes: list,
    pct_base: str = "valid",
    missing_rules_by_var: dict | None = None,
    weight_col: str | None = None,
):
    """
    Independent profile table generator (Point 12).
    Always reports N without parentheses and maintains a distinct '구분1', '구분2' layout.
    """
    rows = []

    # Overall Total Row
    # Use a dummy valid_mask (all True) for total N
    stats = _get_n_stats(df, pd.Series(True, index=df.index), weight_col)
    
    rows.append({
        "구분1": "전체",
        "구분2": "",
        "N": _format_weighted_n(stats["total_n"]),
        "%": 100.0,
    })

    for var in profile_vars:
        var_label = get_var_label(var, metadata)
        summary = compute_single_distribution(
            df=df,
            var_name=var,
            metadata=metadata,
            empty_include=empty_include,
            missing_codes=missing_codes,
            pct_base=pct_base,
            missing_rules_by_var=missing_rules_by_var,
            weight_col=weight_col,
        )

        for item in summary["rows"]:
            rows.append({
                "구분1": var_label,
                "구분2": item["label"],
                "N": _format_weighted_n(item["n"]),
                "%": item["pct"],
            })

    return pd.DataFrame(rows, columns=["구분1", "구분2", "N", "%"])


# =========================================================
# 9. 순위형 통계표
# =========================================================
def summarize_one_group_ranking(
    sub_df: pd.DataFrame,
    rank_vars: list[str],
    metadata: dict,
    show_n: bool,
    show_pct: bool,
    pct_display_mode: str,
    empty_include: bool,
    missing_codes: list,
    rank_show_first: bool,
    rank_show_topk: bool,
    rank_show_mean: bool,
    rank_top_k: int,
    missing_rules_by_var: dict | None = None,
    rank_label_map: dict | None = None,
):
    row = {}

    summary = compute_ranking_distribution(
        df=sub_df,
        rank_vars=rank_vars,
        metadata=metadata,
        empty_include=empty_include,
        missing_codes=missing_codes,
        rank_top_k=rank_top_k,
        missing_rules_by_var=missing_rules_by_var,
    )

    rank_label_map = rank_label_map or _build_unique_label_map([(item["code"], item["label"]) for item in summary["rows"]])

    if show_n:
        row["N"] = int(summary["valid_n"])

    if show_pct:
        for item in summary["rows"]:
            label = rank_label_map.get(item["code"], item["label"])

            if rank_show_first:
                if pct_display_mode == "비율만":
                    row[f"{label}_1순위%"] = fmt_pct(item["first_pct"])
                elif pct_display_mode == "응답수+비율(두열)":
                    row[f"{label}_1순위N"] = int(item["first_n"])
                    row[f"{label}_1순위%"] = fmt_pct(item["first_pct"])
                elif pct_display_mode == "응답수(비율)한셀":
                    pct_str = "-" if item["first_pct"] is None else format_pct(item['first_pct'])
                    row[f"{label}_1순위"] = f"{int(item['first_n'])} ({pct_str})"

            if rank_show_topk:
                topk_label = f"{label}_1+{summary['top_k']}순위"
                if pct_display_mode == "비율만":
                    row[f"{topk_label}%"] = fmt_pct(item["top_k_pct"])
                elif pct_display_mode == "응답수+비율(두열)":
                    row[f"{topk_label}N"] = int(item["top_k_n"])
                    row[f"{topk_label}%"] = fmt_pct(item["top_k_pct"])
                elif pct_display_mode == "응답수(비율)한셀":
                    pct_str = "-" if item["top_k_pct"] is None else format_pct(item['top_k_pct'])
                    row[topk_label] = f"{int(item['top_k_n'])} ({pct_str})"

            if rank_show_mean:
                row[f"{label}_평균순위"] = item["mean_rank"]

    return row


def build_ranking_table(
    df: pd.DataFrame,
    dep_var: str,
    banner_tree: list,
    metadata: dict,
    include_total: bool,
    empty_include: bool,
    show_n: bool,
    show_pct: bool,
    pct_display_mode: str,
    pct_base: str,
    missing_codes: list,
    question_type_map: dict | None = None,
    rank_show_first: bool = True,
    rank_show_topk: bool = True,
    rank_show_mean: bool = True,
    rank_top_k: int = 2,
    missing_rules_by_var: dict | None = None,
    weight_col: str | None = None,
):
    rank_vars = _resolve_rank_group_vars(dep_var, df, question_type_map=question_type_map)

    if len(rank_vars) < 2:
        return build_block_table(
            df=df,
            dep_var=dep_var,
            banner_tree=banner_tree,
            metadata=metadata,
            include_total=include_total,
            empty_include=empty_include,
            show_n=show_n,
            show_pct=show_pct,
            pct_display_mode=pct_display_mode,
            show_subtotal=False,
            subtotal_groups=[],
            exclude_subtotal_vars=[],
            show_mean=False,
            show_std=False,
            pct_base=pct_base,
            scale_vars=[],
            missing_codes=missing_codes,
            missing_rules_by_var=missing_rules_by_var,
            question_type="순위형",
            weight_col=weight_col,
        )

    rank_summary_categories = _get_rank_categories_from_group(
        df=df,
        rank_vars=rank_vars,
        metadata=metadata,
        empty_include=empty_include,
        missing_codes=missing_codes,
        missing_rules_by_var=missing_rules_by_var,
    )
    rank_label_map = _build_unique_label_map([(cat, get_value_label(rank_vars[0], cat, metadata)) for cat in rank_summary_categories])

    rows = []
    banner_paths = extract_banner_var_paths(banner_tree)
    group_cols = build_banner_level_columns(banner_paths)

    if include_total:
        stat_row = summarize_one_group_ranking(
            sub_df=df,
            rank_vars=rank_vars,
            metadata=metadata,
            show_n=show_n,
            show_pct=show_pct,
            pct_display_mode=pct_display_mode,
            empty_include=empty_include,
            missing_codes=missing_codes,
            rank_show_first=rank_show_first,
            rank_show_topk=rank_show_topk,
            rank_show_mean=rank_show_mean,
            rank_top_k=rank_top_k,
            missing_rules_by_var=missing_rules_by_var,
            rank_label_map=rank_label_map,
        )
        total_prefix = ["전체"] + [""] * (len(group_cols) - 1)
        rows.append({group_cols[i]: total_prefix[i] for i in range(len(group_cols))} | stat_row)

    banner_rows = generate_banner_rows(
        df=df,
        banner_tree=banner_tree,
        metadata=metadata,
        empty_include=empty_include,
        missing_codes=missing_codes,
        missing_rules_by_var=missing_rules_by_var,
    )

    for item in banner_rows:
        labels = item["labels"]
        sub_df = item["sub_df"]

        stat_row = summarize_one_group_ranking(
            sub_df=sub_df,
            rank_vars=rank_vars,
            metadata=metadata,
            show_n=show_n,
            show_pct=show_pct,
            pct_display_mode=pct_display_mode,
            empty_include=empty_include,
            missing_codes=missing_codes,
            rank_show_first=rank_show_first,
            rank_show_topk=rank_show_topk,
            rank_show_mean=rank_show_mean,
            rank_top_k=rank_top_k,
            missing_rules_by_var=missing_rules_by_var,
            rank_label_map=rank_label_map,
        )
        prefix = labels + [""] * (len(group_cols) - len(labels))
        rows.append({group_cols[i]: prefix[i] for i in range(len(group_cols))} | stat_row)

    result = pd.DataFrame(rows)
    ordered_stat_cols = []

    if show_n and "N" in result.columns:
        ordered_stat_cols.append("N")

    summary_total = compute_ranking_distribution(
        df=df,
        rank_vars=rank_vars,
        metadata=metadata,
        empty_include=empty_include,
        missing_codes=missing_codes,
        rank_top_k=rank_top_k,
        missing_rules_by_var=missing_rules_by_var,
    )

    labels = [r["label"] for r in summary_total["rows"]]
    top_k = summary_total["top_k"]

    if show_pct:
        for label in labels:
            if rank_show_first:
                if pct_display_mode == "비율만":
                    col = f"{label}_1순위%"
                    if col in result.columns:
                        ordered_stat_cols.append(col)
                elif pct_display_mode == "응답수+비율(두열)":
                    col_n = f"{label}_1순위N"
                    col_pct = f"{label}_1순위%"
                    if col_n in result.columns:
                        ordered_stat_cols.append(col_n)
                    if col_pct in result.columns:
                        ordered_stat_cols.append(col_pct)
                elif pct_display_mode == "응답수(비율)한셀":
                    col = f"{label}_1순위"
                    if col in result.columns:
                        ordered_stat_cols.append(col)

            if rank_show_topk:
                topk_base = f"{label}_1+{top_k}순위"
                if pct_display_mode == "비율만":
                    col = f"{topk_base}%"
                    if col in result.columns:
                        ordered_stat_cols.append(col)
                elif pct_display_mode == "응답수+비율(두열)":
                    col_n = f"{topk_base}N"
                    col_pct = f"{topk_base}%"
                    if col_n in result.columns:
                        ordered_stat_cols.append(col_n)
                    if col_pct in result.columns:
                        ordered_stat_cols.append(col_pct)
                elif pct_display_mode == "응답수(비율)한셀":
                    if topk_base in result.columns:
                        ordered_stat_cols.append(topk_base)

            if rank_show_mean:
                col = f"{label}_평균순위"
                if col in result.columns:
                    ordered_stat_cols.append(col)

    if result.empty:
        return pd.DataFrame(columns=group_cols + ordered_stat_cols)

    return result[group_cols + ordered_stat_cols]


# =========================================================
# 10. 통합 진입점
# =========================================================
def build_question_table(
    df: pd.DataFrame,
    dep_var: str,
    question_type: str,
    banner_tree: list,
    metadata: dict,
    include_total: bool,
    empty_include: bool,
    show_n: bool,
    show_pct: bool,
    pct_display_mode: str,
    show_subtotal: bool,
    subtotal_groups: list,
    exclude_subtotal_vars: list,
    show_mean: bool,
    show_std: bool,
    pct_base: str,
    scale_vars: list,
    missing_codes: list,
    question_type_map: dict | None = None,
    rank_show_first: bool = True,
    rank_show_topk: bool = True,
    rank_show_mean: bool = True,
    rank_top_k: int = 2,
    missing_rules_by_var: dict | None = None,
    weight_col: str | None = None,
):
    question_type = str(question_type or "범주형").strip()
    question_type_map = question_type_map or {}

    if question_type == "순위형":
        return build_ranking_table(
            df=df,
            dep_var=dep_var,
            banner_tree=banner_tree,
            metadata=metadata,
            include_total=include_total,
            empty_include=empty_include,
            show_n=show_n,
            show_pct=show_pct,
            pct_display_mode=pct_display_mode,
            pct_base=pct_base,
            missing_codes=missing_codes,
            question_type_map=question_type_map,
            rank_show_first=rank_show_first,
            rank_show_topk=rank_show_topk,
            rank_show_mean=rank_show_mean,
            rank_top_k=rank_top_k,
            missing_rules_by_var=missing_rules_by_var,
            weight_col=weight_col,
        )

    if question_type == "개방형":
        return pd.DataFrame(columns=["구분1", "N"])

    return build_block_table(
        df=df,
        dep_var=dep_var,
        banner_tree=banner_tree,
        metadata=metadata,
        include_total=include_total,
        empty_include=empty_include,
        show_n=show_n,
        show_pct=show_pct,
        pct_display_mode=pct_display_mode,
        show_subtotal=show_subtotal,
        subtotal_groups=subtotal_groups,
        exclude_subtotal_vars=exclude_subtotal_vars,
        show_mean=show_mean,
        show_std=show_std,
        pct_base=pct_base,
        scale_vars=scale_vars,
        missing_codes=missing_codes,
        missing_rules_by_var=missing_rules_by_var,
        question_type=question_type,
        weight_col=weight_col,
    )


# =========================================================
# 11. 고급 기능: 다문항 요약표 (Point 1)
# =========================================================
def build_summary_table(
    df: pd.DataFrame,
    vars: list,
    metadata: dict,
    scale_vars: list,
    missing_codes: list,
    pct_base: str = "valid",
    subtotal_text: str = "4+5",
    show_index_score: bool = True,
    weight_col: str | None = None,
):
    """
    Generates a summary table across multiple variables (Point 1).
    Columns: 문항, N, [Subtotal %], [Mean], [Index Score]
    """
    rows = []
    subtotal_groups = parse_subtotal_groups(subtotal_text)
    
    for var in vars:
        var_label = get_var_label(var, metadata)
        valid_series = filter_valid_series(df[var], missing_codes)
        numeric_valid = safe_numeric(valid_series)
        valid_mask = pd.Series(df.index.isin(valid_series.index), index=df.index)

        stats = _get_n_stats(df, valid_mask, weight_col)
        valid_n = stats["valid_n"]
        total_n = stats["total_n"]
        
        row = {"문항": var_label, "N": _format_weighted_n(valid_n)}
        
        # 1. Subtotal (e.g., Top-2 Box)
        for group in subtotal_groups:
            codes = group["codes"]
            mask = _build_code_membership_mask(valid_series, codes)
            cat_count = _weighted_sum_for_mask(df.loc[valid_series.index], mask, weight_col)
            denom = valid_n if pct_base == "valid" else total_n
            pct = (cat_count / denom * 100) if denom > 0 else None
            row[group["label"]] = fmt_pct(pct)

        # 2. Mean
        weights = _get_weight_series(df.loc[valid_series.index], weight_col)
        mean_val = _weighted_mean(numeric_valid, weights)
        row["평균"] = fmt_num(mean_val)
        
        # 3. Index Score (Point 2)
        if show_index_score:
            # Determine scale type (5 or 10)
            cats = metadata.get(var, {}).get("value_labels", {})
            scale_type = 5 if len(cats) == 5 else (10 if len(cats) == 10 else 5)
            index_val = _calculate_index_score(mean_val, scale_type)
            row["100점 환산"] = fmt_num(index_val)
            
        rows.append(row)
        
    return pd.DataFrame(rows)


# =========================================================
# 12. 분석 감사 기록 (Point 5)
# =========================================================
def generate_analysis_audit_trail(
    df: pd.DataFrame,
    metadata: dict,
    weight_col: str | None,
    missing_codes: list,
    pct_base: str,
    subtotal_text: str,
    analysis_mode: str,
):
    """Generates metadata about the current analysis run for the final report."""
    import datetime
    trail = {
        "생성일시": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "분석기준": analysis_mode,
        "가중치변수": weight_col or "(미적용)",
        "결측코드": ", ".join(map(str, missing_codes)) if missing_codes else "(없음)",
        "백분율기준": "유효응답" if pct_base == "valid" else "전체응답",
        "부분합설정": subtotal_text,
        "전체샘플수(N)": len(df),
    }
    return trail
