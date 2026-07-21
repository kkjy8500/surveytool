import pandas as pd
import json

# ---------------------------------------------------------
# Missing Value Detection and Rules
# ---------------------------------------------------------

def is_missing_by_sav_rules(value, sav_rule: dict | None) -> bool:
    """Check if a value matches SAV user-defined missing codes or ranges."""
    if pd.isna(value):
        return True
    if not sav_rule:
        return False

    user_values = sav_rule.get("user_values", []) or []
    for code in user_values:
        try:
            if float(value) == float(code):
                return True
        except Exception:
            if str(value).strip() == str(code).strip():
                return True

    for item in sav_rule.get("ranges", []) or []:
        try:
            lo = float(item.get("lo"))
            hi = float(item.get("hi"))
            fv = float(value)
            if lo <= fv <= hi:
                return True
        except Exception:
            continue
    return False

def is_missing_value(value, missing_codes=None, sav_rule: dict | None = None):
    """
    Check if a value is considered 'missing' based on standard tokens,
    manual codes, or SAV-defined rules.
    """
    if pd.isna(value):
        return True

    s = str(value).strip().lower()
    missing_tokens = {
        "", "na", "n/a", "nan", "none", "null",
        ".", "-", "--", "모름"
        # "없음" removed to allow valid answers like "지지정당 없음"
    }
    if s in missing_tokens:
        return True

    if is_missing_by_sav_rules(value, sav_rule):
        return True

    for m in missing_codes or []:
        try:
            if float(value) == float(m):
                return True
        except Exception:
            if str(value).strip() == str(m).strip():
                return True

    return False

def filter_valid_series(series: pd.Series, missing_codes: list, sav_rule: dict | None = None):
    """Filter out missing values from a pandas Series."""
    return series[~series.apply(lambda x: is_missing_value(x, missing_codes, sav_rule=sav_rule))]

# ---------------------------------------------------------
# Parsing and Combining Missing Specifications
# ---------------------------------------------------------

def parse_code_list(text: str):
    """Parse a comma-separated string into a list of numbers or strings."""
    if not text.strip():
        return []

    result = []
    parts = [x.strip() for x in text.split(",") if x.strip()]

    for p in parts:
        try:
            result.append(float(p))
        except Exception:
            result.append(p)

    return result

def parse_missing_codes(text: str):
    """Alias for parse_code_list specifically for missing value inputs."""
    return parse_code_list(text)

def combine_missing_specs(
    manual_missing_codes: list,
    sav_meta: dict | None = None,
    use_sav_missing: bool = False,
) -> dict:
    """
    Combine global missing codes with variable-specific rules (e.g., from SAV).
    """
    from sav_utils import extract_sav_missing_rules  # Local import to prevent circular dependency
    
    manual_missing_codes = manual_missing_codes or []
    combined_codes = list(manual_missing_codes)
    missing_rules_by_var = {}

    if use_sav_missing:
        rules = extract_sav_missing_rules(sav_meta)
        for var, values in rules["user_values"].items():
            missing_rules_by_var.setdefault(var, {"user_values": [], "ranges": []})
            missing_rules_by_var[var]["user_values"].extend(values)
        for var, ranges in rules["ranges"].items():
            missing_rules_by_var.setdefault(var, {"user_values": [], "ranges": []})
            missing_rules_by_var[var]["ranges"].extend(ranges)

    deduped_codes = []
    seen = set()
    for item in combined_codes:
        key = json.dumps(item, ensure_ascii=False, default=str)
        if key in seen:
             continue
        seen.add(key)
        deduped_codes.append(item)

    return {
        "global_missing_codes": deduped_codes,
        "missing_rules_by_var": missing_rules_by_var,
    }
