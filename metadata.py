import re
import pandas as pd

from utils import candidate_columns, filter_valid_series

# ---------------------------------------------------------
# Configuration and Constants
# ---------------------------------------------------------

COLUMN_ALIASES = {
    "문항/보기번호": [
        "문항/보기번호", "문항번호", "보기번호", "item_no", "item", "question_no", 
        "question number", "번호", "no", "code", "보기값", "값"
    ],
    "내용": [
        "내용", "문항내용", "보기내용", "text", "label", "content", 
        "question text", "보기", "문항", "라벨"
    ],
    "VALUE LABELS": [
        "VALUE LABELS", "value labels", "value_labels", "value label", "변수명", 
        "var", "variable", "변수", "name", "컬럼명", "column"
    ],
    "QtnType": [
        "QtnType", "qtn type", "question type", "question_type", "type", 
        "문항유형", "문항 타입"
    ],
}

# ---------------------------------------------------------
# Text Normalization Utilities
# ---------------------------------------------------------

def _normalize_key(text) -> str:
    """Normalize text for column mapping (remove special chars, lowercase)."""
    return re.sub(r"[^a-z0-9가-힣]+", "", str(text).strip().lower())


def is_var_token(x) -> str | None:
    """
    Checks if a string represents a new variable start (e.g., /Q1, Q1, SQ1, Q1_1).
    Returns the cleaned variable name if it matches, else None.
    """
    s = str(x).strip()
    if not s:
        return None
    # Pattern: Optional /, then letters, then numbers, then optional underscore and numbers
    # Also handles case like /Q1.
    match = re.match(r"^/?([A-Za-z가-힣]*[A-Za-z]+\d+(_\d+)?)\.?$", s)
    if match:
        return match.group(1)
    return None


def is_code_token(x) -> int | None:
    """Checks if a string represents a numeric code (e.g., 1, 2, -9)."""
    s = str(x).strip()
    if re.fullmatch(r"-?\d+(\.0+)?", s):
        try:
            return int(float(s))
        except:
            return None
    return None

def resolve_column_guide_columns(guide_df: pd.DataFrame) -> dict:
    """Map guide dataframe columns to canonical metadata keys using aliases."""
    resolved = {}
    normalized_columns = {_normalize_key(col): col for col in guide_df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            key = _normalize_key(alias)
            if key in normalized_columns:
                found = normalized_columns[key]
                break
        resolved[canonical] = found

    return resolved

# ---------------------------------------------------------
# Metadata Parsing and Extraction
# ---------------------------------------------------------

def parse_column_guide_df(guide_df: pd.DataFrame, return_diagnostics: bool = False):
    """
    Parse a Column Guide dataframe into a structured metadata dictionary.
    Includes flexible detection for variable starts and numeric codes (Point 2, 3, 4).
    """
    guide_df = guide_df.fillna("")
    colmap = resolve_column_guide_columns(guide_df)

    item_col = colmap.get("문항/보기번호")
    content_col = colmap.get("내용")
    value_label_col = colmap.get("VALUE LABELS")
    qtn_type_col = colmap.get("QtnType")

    # Validate presence of required columns (Only Item No and Content are strictly mandatory)
    required_missing = [
        name
        for name, actual in [("문항/보기번호", item_col), ("내용", content_col)]
        if actual is None
    ]
    if required_missing:
        # Fallback: maybe the first columns are what we want if headers are weird
        if len(guide_df.columns) >= 2:
            item_col = guide_df.columns[0]
            content_col = guide_df.columns[1]
            # Try to guess value label column as 3rd column if it exists
            value_label_col = guide_df.columns[2] if len(guide_df.columns) >= 3 else None
        else:
            raise ValueError(
                "Column Guide 필수 열을 찾지 못했습니다 (최소 2개 열 필요): "
                + ", ".join(required_missing)
                + "\n현재 열 이름을 확인해 주세요."
            )

    metadata = {}
    current_var = None
    diagnostics = {
        "vars_found": 0,
        "vars_with_labels": 0,
        "vars_without_labels": [],
        "errors": []
    }

    # Iterate through rows to extract variables and their value labels
    for idx, row in guide_df.iterrows():
        item_val = str(row.get(item_col, "")).strip()
        content_val = str(row.get(content_col, "")).strip()
        vlabel_val = str(row.get(value_label_col, "")).strip()
        qtn_type = str(row.get(qtn_type_col, "")).strip() if qtn_type_col else ""

        # Step 1: Detect new variable start (Point 2)
        # Check Value Labels column first, then Item No column
        found_var = is_var_token(vlabel_val)
        if not found_var:
            # If Item No column has something like Q1 and Content has a label, it's likely a variable start
            # But only if it doesn't look like a simple numeric code.
            temp_var = is_var_token(item_val)
            if temp_var and not is_code_token(item_val) and content_val:
                found_var = temp_var

        if found_var:
            # Check if this is truly a new variable or just a repeat
            current_var = found_var
            metadata[current_var] = {
                "label": content_val if content_val else current_var,
                "display_label": content_val if content_val else current_var,
                "value_labels": {},
                "display_value_labels": {},
                "qtn_type": qtn_type,
                "source": "guide"
            }
            diagnostics["vars_found"] += 1
            continue

        # Step 2: Detect value labels (Point 3)
        code = is_code_token(item_val)
        if current_var and code is not None:
            metadata[current_var]["value_labels"][code] = content_val

    # Finalize Diagnostics
    for var, meta in metadata.items():
        if meta.get("value_labels"):
            diagnostics["vars_with_labels"] += 1
        else:
            diagnostics["vars_without_labels"].append(var)

    if return_diagnostics:
        return metadata, diagnostics
    return metadata

def build_metadata_from_sav_info(df: pd.DataFrame, sav_meta: dict | None = None) -> dict:
    """Build metadata from SPSS/SAV file information extracted via pyreadstat."""
    sav_meta = sav_meta or {}
    variable_labels = sav_meta.get("variable_labels", {}) or {}
    value_labels = sav_meta.get("value_labels", {}) or {}
    variable_measure = sav_meta.get("variable_measure", {}) or {}

    metadata = {}
    for col in df.columns:
        measure = str(variable_measure.get(col, "")).strip().lower()
        qtn_type = "SCALE" if measure == "scale" else ""
        label = str(variable_labels.get(col, col)).strip() or col
        metadata[col] = {
            "label": label,
            "display_label": label,
            "value_labels": value_labels.get(col, {}) or {},
            "display_value_labels": {},
            "qtn_type": qtn_type,
            "source": "sav"
        }
    return metadata

# ---------------------------------------------------------
# Metadata Merging and Label Lookups
# ---------------------------------------------------------

def merge_metadata(primary: dict, secondary: dict) -> dict:
    """
    Merge two metadata dictionaries, prioritizing the primary source.
    Useful for merging Column Guide info (primary) over SAV info (secondary).
    """
    import copy
    merged = copy.deepcopy(secondary)
    
    for var, meta in primary.items():
        if var in merged:
            # Overwrite metadata with primary source values if present
            if meta.get("label"):
                merged[var]["label"] = meta["label"]
            if meta.get("display_label"):
                merged[var]["display_label"] = meta["display_label"]
            
            # Merge value labels non-destructively
            if meta.get("value_labels"):
                merged[var].setdefault("value_labels", {})
                merged[var]["value_labels"].update(meta["value_labels"])
            
            # Preserve or update user-defined display labels
            if meta.get("display_value_labels"):
                merged[var].setdefault("display_value_labels", {})
                merged[var]["display_value_labels"].update(meta["display_value_labels"])
                
            if meta.get("qtn_type"):
                merged[var]["qtn_type"] = meta["qtn_type"]
            
            merged[var]["source"] = "merged"
        else:
            merged[var] = copy.deepcopy(meta)
            
    return merged

def get_var_label(var_name, metadata):
    """Retrieve the best variable label (Display > Original > Name)."""
    meta = metadata.get(var_name, {})
    display_label = str(meta.get("display_label", "")).strip()
    if display_label:
        return display_label
    label = str(meta.get("label", "")).strip()
    if label:
        return label
    return str(var_name)

def _lookup_label_from_map(label_map: dict, value):
    """Find a label in a map with fuzzy type matching for int/float/str keys."""
    if not label_map:
        return None

    # Build potential key variations
    candidates = []
    raw_text = str(value).strip()
    if raw_text:
        candidates.append(raw_text)

    try:
        fvalue = float(raw_text)
        candidates.append(fvalue)
        if fvalue.is_integer():
            ivalue = int(fvalue)
            candidates.extend([ivalue, str(ivalue), f"{ivalue}.0"])
    except (ValueError, TypeError):
        pass

    # Direct lookup
    for key in candidates:
        if key in label_map and str(label_map.get(key, "")).strip():
            return str(label_map[key])

    # Secondary fuzzy normalization lookup
    def _norm(x):
        text = str(x).strip()
        try:
            num = float(text)
            return str(int(num)) if num.is_integer() else str(num)
        except (ValueError, TypeError):
            return text

    target = _norm(value)
    for key, label in label_map.items():
        if _norm(key) == target and str(label).strip():
            return str(label)

    return None

def get_value_label(var_name, value, metadata):
    """Retrieve the value label (Display > Original > Value)."""
    meta = metadata.get(var_name, {})
    # Check display overrides first
    label = _lookup_label_from_map(meta.get("display_value_labels", {}) or {}, value)
    if label is not None:
        return label
    # Fallback to original metadata labels
    label = _lookup_label_from_map(meta.get("value_labels", {}) or {}, value)
    if label is not None:
        return label
    return str(value)

# ---------------------------------------------------------
# Dynamic Metadata Creation
# ---------------------------------------------------------

def add_metadata_for_new_var(metadata: dict, var_name: str, label: str, value_labels: dict, source: str = "recode"):
    """Safely register metadata for variables created during processing (e.g., Recoding)."""
    metadata[var_name] = {
        "label": label,
        "display_label": label,
        "value_labels": value_labels or {},
        "display_value_labels": (value_labels or {}).copy(),
        "qtn_type": "범주형",
        "source": source,
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return metadata

def build_variable_options_from_metadata(metadata: dict, usable_cols: list | None = None):
    """Generate variable options for UI selection components."""
    if usable_cols is None:
        usable_cols = list(metadata.keys())

    options = {}
    for var in usable_cols:
        label = get_var_label(var, metadata)
        options[var] = f"{label} [{var}]"
    return options

# ---------------------------------------------------------
# Diagnosis and Validation
# ---------------------------------------------------------

def find_metadata_issues(df: pd.DataFrame, metadata: dict, missing_codes: list):
    """Identify metadata gaps such as missing labels or mismatched variable names."""
    usable_cols = candidate_columns(df)
    issues = {
        "missing_var_labels": [],
        "missing_value_labels": [],
        "mismatched_vars": [],
        "duplicate_display_labels": {}
    }
    display_labels_seen = {}

    for col in df.columns:
        if col not in metadata:
            issues["mismatched_vars"].append(col)
            continue
        
        if col not in usable_cols:
            continue

        meta = metadata[col]
        label = str(meta.get("label", "")).strip()
        if not label or label == col:
            issues["missing_var_labels"].append(col)
        
        # Track duplicate display labels which can cause confusion in reports
        d_label = meta.get("display_label", label)
        if d_label:
            display_labels_seen.setdefault(d_label, []).append(col)

        # Check for values in data that lack corresponding labels
        value_map = meta.get("value_labels", {})
        series = filter_valid_series(df[col], missing_codes)
        unique_vals = series.dropna().unique()
        
        unlabeled_codes = [v for v in unique_vals if _lookup_label_from_map(value_map, v) is None]
        if unlabeled_codes:
            issues["missing_value_labels"].append({
                "var": col,
                "missing_codes": sorted(unlabeled_codes, key=lambda x: str(x))
            })

    # Record any detected duplicate display labels
    for label, vars in display_labels_seen.items():
        if len(vars) > 1:
            issues["duplicate_display_labels"][label] = vars

    return issues

def validate_metadata_against_df(df: pd.DataFrame, metadata: dict):
    """Strict integrity check between metadata keys and dataframe columns."""
    df_cols = set(df.columns)
    meta_cols = set(metadata.keys())

    return {
        "ghost_vars": list(df_cols - meta_cols), # In DF but not in Metadata
        "zombie_vars": list(meta_cols - df_cols), # In Metadata but not in DF
        "invalid_value_labels": [v for v, m in metadata.items() if not isinstance(m.get("value_labels"), dict)],
        "empty_labels": [v for v, m in metadata.items() if not str(m.get("label", "")).strip() and not str(m.get("display_label", "")).strip()]
    }

def build_minimal_metadata_from_df(df: pd.DataFrame) -> dict:
    """Generate a placeholder metadata structure when no external metadata is provided."""
    metadata = {}
    if df is None:
        return metadata

    for col in df.columns:
        col_str = str(col)
        metadata[col_str] = {
            "label": col_str,
            "display_label": col_str,
            "value_labels": {},
            "display_value_labels": {},
            "qtn_type": "",
            "source": "minimal"
        }
    return metadata

