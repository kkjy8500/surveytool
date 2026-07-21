import pandas as pd
import json
import re

# ---------------------------------------------------------
# SAV Cell and Data Coercion
# ---------------------------------------------------------

def _normalize_sav_cell(value):
    """Normalize a single cell value for SAV export, handling encoding and complex types."""
    if pd.isna(value):
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("cp949", errors="ignore")
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)

def _coerce_sav_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame for SAV export by coercing types to compatible formats."""
    out = df.copy()
    out.columns = [str(col) for col in out.columns]

    for col in out.columns:
        series = out[col]
        dtype_str = str(series.dtype)

        if pd.api.types.is_bool_dtype(series) or dtype_str == "boolean":
            out[col] = series.map(lambda x: None if pd.isna(x) else int(bool(x))).astype("float64")
            continue

        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series) or dtype_str.startswith("Int") or dtype_str.startswith("Float"):
            out[col] = pd.to_numeric(series, errors="coerce")
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if pd.api.types.is_categorical_dtype(series):
            series = series.astype("object")

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_missing = series.dropna()
            numeric_like = False
            if len(non_missing) > 0:
                converted = pd.to_numeric(non_missing, errors="coerce")
                numeric_like = converted.notna().all()

            if numeric_like:
                out[col] = pd.to_numeric(series, errors="coerce")
            else:
                out[col] = series.map(_normalize_sav_cell).astype("object")

    return out

# ---------------------------------------------------------
# SAV Variable Name Sanitization
# ---------------------------------------------------------

def _sanitize_sav_variable_name(name: str) -> str:
    """Ensure variable names are compatible with SPSS naming rules (max 64 chars, etc.)."""
    text = str(name).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_]", "", text)
    if not text:
        text = "VAR"
    if re.match(r"^\d", text):
        text = f"V_{text}"
    text = text[:64]
    return text

def _sanitize_sav_variable_names(df: pd.DataFrame):
    """Sanitize all column names in a DataFrame for SAV export."""
    renamed = {}
    used = set()
    out = df.copy()

    for col in out.columns:
        original = str(col)
        base = _sanitize_sav_variable_name(original)
        candidate = base
        suffix = 1

        while candidate in used:
            suffix_text = f"_{suffix}"
            candidate = f"{base[:max(1, 64 - len(suffix_text))]}{suffix_text}"
            suffix += 1

        used.add(candidate)
        renamed[original] = candidate

    out = out.rename(columns=renamed)
    return out, renamed

# ---------------------------------------------------------
# SAV Export Operations
# ---------------------------------------------------------

def dataframe_to_sav_bytes(df: pd.DataFrame, metadata: dict | None = None) -> tuple[bytes, dict]:
    """Convert a pandas DataFrame to SAV file bytes with metadata labels."""
    import os
    import tempfile

    try:
        import pyreadstat  # type: ignore
    except Exception as e:
        raise ImportError("SAV 저장을 위해 pyreadstat가 필요합니다.") from e

    metadata = metadata or {}
    export_df = _coerce_sav_export_df(df)
    export_df, rename_map = _sanitize_sav_variable_names(export_df)

    column_labels = {}
    variable_value_labels = {}
    variable_measure = {}

    for original_col, sav_col in rename_map.items():
        meta = metadata.get(original_col, {}) or {}
        label = str(meta.get("label", original_col)).strip() or str(original_col)
        column_labels[str(sav_col)] = label

        value_map = meta.get("value_labels", {}) or {}
        cleaned_value_map = {}
        for key, value in value_map.items():
            try:
                cleaned_key = int(float(key))
            except Exception:
                continue
            cleaned_value_map[cleaned_key] = str(value)

        if cleaned_value_map:
            variable_value_labels[str(sav_col)] = cleaned_value_map

        qtn_type = str(meta.get("qtn_type", "")).strip().upper()
        if qtn_type == "SCALE":
            variable_measure[str(sav_col)] = "scale"
        elif cleaned_value_map:
            variable_measure[str(sav_col)] = "nominal"
        else:
            variable_measure[str(sav_col)] = "unknown"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
        tmp_path = tmp.name

    try:
        try:
            pyreadstat.write_sav(
                export_df,
                tmp_path,
                column_labels=column_labels or None,
                variable_value_labels=variable_value_labels or None,
                variable_measure=variable_measure or None,
            )
        except Exception:
            fallback_df = export_df.copy()
            for col in fallback_df.columns:
                series = fallback_df[col]
                if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                    fallback_df[col] = series.map(_normalize_sav_cell).astype("object")

            pyreadstat.write_sav(
                fallback_df,
                tmp_path,
                column_labels=column_labels or None,
                variable_value_labels=variable_value_labels or None,
                variable_measure=variable_measure or None,
            )

        with open(tmp_path, "rb") as f:
            return f.read(), rename_map
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ---------------------------------------------------------
# SAV Metadata Extraction and Missing Rules
# ---------------------------------------------------------

def _normalize_scalar(value):
    """Normalize numeric scalar values for metadata storage."""
    if pd.isna(value):
        return None
    try:
        fv = float(value)
        if fv.is_integer():
            return int(fv)
        return float(fv)
    except Exception:
        return str(value).strip()

def extract_sav_missing_rules(sav_meta: dict | None) -> dict:
    """Extract and normalize missing value rules (codes and ranges) from SAV metadata."""
    sav_meta = sav_meta or {}
    raw_user_values = sav_meta.get("missing_user_values", {}) or {}
    raw_ranges = sav_meta.get("missing_ranges", {}) or {}

    user_values = {}
    for var, values in raw_user_values.items():
        cleaned = []
        for value in values or []:
            norm = _normalize_scalar(value)
            if norm is not None:
                cleaned.append(norm)
        if cleaned:
            user_values[str(var)] = cleaned

    ranges = {}
    for var, items in raw_ranges.items():
        cleaned_ranges = []
        for item in items or []:
            if isinstance(item, dict):
                lo = _normalize_scalar(item.get("lo"))
                hi = _normalize_scalar(item.get("hi"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                lo = _normalize_scalar(item[0])
                hi = _normalize_scalar(item[1])
            else:
                continue
            if lo is None or hi is None:
                continue
            cleaned_ranges.append({"lo": lo, "hi": hi})
        if cleaned_ranges:
            ranges[str(var)] = cleaned_ranges

    return {"user_values": user_values, "ranges": ranges}

def get_sav_missing_summary_text(sav_meta: dict | None, max_vars: int = 8) -> str:
    """Generate a human-readable summary of missing value rules for UI display."""
    rules = extract_sav_missing_rules(sav_meta)
    touched = sorted(set(rules["user_values"]) | set(rules["ranges"]))
    if not touched:
        return ""

    lines = []
    for var in touched[:max_vars]:
        parts = []
        if var in rules["user_values"]:
            parts.append("값: " + ", ".join(map(str, rules["user_values"][var])))
        if var in rules["ranges"]:
            range_text = ", ".join(f"{r['lo']}~{r['hi']}" for r in rules["ranges"][var])
            parts.append("범위: " + range_text)
        lines.append(f"{var} ({'; '.join(parts)})")

    extra = "" if len(touched) <= max_vars else f" 외 {len(touched) - max_vars}개 변수"
    return " / ".join(lines) + extra

# ---------------------------------------------------------
# SAV File Loading
# ---------------------------------------------------------

def _load_sav_file(uploaded_file):
    """Load a SAV file and extract its metadata using pyreadstat or pandas as fallback."""
    import os
    import tempfile
    from io import BytesIO

    sav_meta = None
    file_name = getattr(uploaded_file, "name", "data.sav")

    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()

    if not raw_bytes:
        raise ValueError("SAV 파일이 비어 있습니다.")

    pyreadstat_error = None
    pandas_error = None

    try:
        import pyreadstat  # type: ignore

        suffix = os.path.splitext(file_name)[1] or ".sav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            df, meta = pyreadstat.read_sav(
                tmp_path,
                apply_value_formats=False,
                user_missing=True,
                disable_datetime_conversion=False,
            )
            sav_meta = {
                "variable_labels": dict(getattr(meta, "column_names_to_labels", {}) or {}),
                "value_labels": dict(getattr(meta, "variable_value_labels", {}) or {}),
                "variable_measure": dict(getattr(meta, "variable_measure", {}) or {}),
                "missing_ranges": dict(getattr(meta, "missing_ranges", {}) or {}),
                "missing_user_values": dict(getattr(meta, "missing_user_values", {}) or {}),
                "reader": "pyreadstat",
            }
            return {"df": df, "sav_meta": sav_meta}
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except Exception as e:
        pyreadstat_error = e

    try:
        bio = BytesIO(raw_bytes)
        df = pd.read_spss(bio, convert_categoricals=False)
        sav_meta = {"reader": "pandas.read_spss"}
        return {"df": df, "sav_meta": sav_meta}
    except Exception as e:
        pandas_error = e

    raise ValueError(
        "SAV 파일을 읽지 못했습니다. 파일이 손상되었거나 pyreadstat 라이브러리가 필요할 수 있습니다.\n"
        f"- pyreadstat 오류: {pyreadstat_error}\n"
        f"- pandas.read_spss 오류: {pandas_error}"
    )
