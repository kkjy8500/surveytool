from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from textwrap import fill

import pandas as pd

from config import TECH_COLS


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", str(name or "export_file"))
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:120] or "export_file"


def slugify_text(text: str, default: str = "dashboard") -> str:
    text = safe_filename(text).lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-._")[:63] or default


def wrap_text(text: str, width: int = 28) -> str:
    return fill(str(text).strip(), width=width)


def natural_sort_key(value):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def sort_var_names(values: list) -> list:
    return sorted(values, key=natural_sort_key)


def safe_numeric(series: pd.Series, preserve_int: bool = False) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if preserve_int and pd.api.types.is_float_dtype(out):
        valid = out.dropna()
        if not valid.empty and (valid % 1 == 0).all():
            return out.astype("Int64")
    return out


def candidate_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if str(c).strip().lower() not in TECH_COLS]


def guess_rank_group_name(var_name: str) -> str:
    text = str(var_name or "").strip()
    match = re.match(r"^(.+?)[_.-](\d+)$", text)
    return match.group(1) if match else text


def parse_subtotal_groups(text: str) -> list[dict]:
    groups = []
    for part in [x.strip() for x in str(text or "").split(",") if x.strip()]:
        label, codes_text = (part.split("=", 1) if "=" in part else ("", part))
        codes = []
        for token in codes_text.split("+"):
            token = token.strip()
            if not re.fullmatch(r"-?\d+", token):
                codes = []
                break
            codes.append(int(token))
        if len(codes) >= 2:
            groups.append({"label": label.strip() or f"부분합({'+'.join(map(str, codes))})", "codes": codes})
    return groups


def make_subtotal_col_name(group: dict) -> str:
    return str(group.get("label") or "부분합")


def clean_sheet_name(name: str) -> str:
    return re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31] or "Sheet"


def dataframe_to_csv_bytes(df: pd.DataFrame, encoding: str = "utf-8-sig") -> bytes:
    return df.to_csv(index=False).encode(encoding)


def dict_to_json_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def build_zip_bytes_from_mapping(file_mapping: dict) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in file_mapping.items():
            zf.writestr(name, content.encode("utf-8") if isinstance(content, str) else content)
    return bio.getvalue()


# Legacy aliases used by the tabulation modules.
from missing_utils import is_missing_value, filter_valid_series, parse_code_list, combine_missing_specs
from display_format import fmt_pct, fmt_num
