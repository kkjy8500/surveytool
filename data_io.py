from __future__ import annotations

from io import BytesIO

import pandas as pd

from sav_utils import _load_sav_file


def _read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    raise ValueError("CSV 파일을 읽지 못했습니다. 인코딩 또는 파일 형식을 확인해 주세요.")


def load_data_file(uploaded_file) -> dict:
    if uploaded_file is None:
        raise ValueError("데이터 파일이 없습니다.")
    name = str(getattr(uploaded_file, "name", "")).lower()
    uploaded_file.seek(0)

    if name.endswith(".csv"):
        df = _read_csv(uploaded_file)
        sav_meta = None
        source_type = "csv"
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        sav_meta = None
        source_type = "xlsx"
    elif name.endswith(".sav"):
        loaded = _load_sav_file(uploaded_file)
        df, sav_meta, source_type = loaded["df"], loaded.get("sav_meta"), "sav"
    else:
        raise ValueError("데이터 파일은 CSV, XLSX, SAV 형식만 지원합니다.")

    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("데이터 파일에 중복된 컬럼명이 있습니다.")
    return {"df": df, "sav_meta": sav_meta, "source_type": source_type, "file_name": getattr(uploaded_file, "name", "")}
