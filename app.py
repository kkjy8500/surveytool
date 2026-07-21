from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from charts import (
    CHART_TYPE_OPTIONS,
    build_all_charts,
    build_chart_results,
    build_chart_settings_df,
    chart_settings_df_to_dict,
    get_available_korean_fonts,
    get_chart_preset_names,
)
from config import (
    DEFAULT_DASHBOARD_APP_SUBTITLE,
    DEFAULT_DASHBOARD_APP_TITLE,
    DEFAULT_DASHBOARD_PACKAGE_NAME,
    DEFAULT_EXPORT_BASENAME,
)
from dashboard import (
    build_dashboard_bundle,
    build_dashboard_config_df,
    build_dashboard_deploy_zip,
    build_dashboard_deploy_files,
    normalize_dashboard_config_df,
    render_dashboard_preview,
)
from data_io import load_data_file
from excel_export import export_to_excel
from questionnaire_parser import questionnaire_text_to_guide_df
from survey_config import build_settings_from_questionnaire, load_survey_settings
from tabulation import build_multiresponse_table, build_profile_table, build_question_table
from utils import build_zip_bytes_from_mapping, safe_filename

ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "templates" / "standard_survey_settings.xlsx"

st.set_page_config(page_title="Survey Tool", page_icon="📊", layout="wide")


def _init_state():
    defaults = {
        "data_file": None,
        "settings_file": None,
        "chart_settings_df": pd.DataFrame(),
        "dashboard_config_df": pd.DataFrame(),
        "chart_files": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _load_context():
    data_file = st.session_state.get("data_file")
    settings_file = st.session_state.get("settings_file")
    if data_file is None or settings_file is None:
        return None
    loaded = load_data_file(data_file)
    settings = load_survey_settings(settings_file, data_df=loaded["df"], sav_meta=loaded.get("sav_meta"))
    return loaded | {"settings": settings}


def _weight_column(df: pd.DataFrame) -> str | None:
    preferred = ["wt", "weight", "weights", "가중치", "표본가중치"]
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        if name.lower() in lower_map:
            series = pd.to_numeric(df[lower_map[name.lower()]], errors="coerce")
            if series.notna().any() and (series.fillna(0) > 0).any():
                return lower_map[name.lower()]
    return None


def _build_tables(ctx: dict) -> tuple[dict, dict]:
    df = ctx["df"]
    cfg = ctx["settings"]
    weight_col = _weight_column(df)
    common = dict(
        include_total=True,
        empty_include=True,
        show_n=True,
        show_pct=True,
        pct_display_mode="비율만",
        pct_base="valid",
        missing_codes=[],
        missing_rules_by_var={},
        weight_col=weight_col,
    )

    tables = {}
    for var in cfg["dep_vars"]:
        qtype = cfg["type_map"].get(var, "범주형")
        banner_key = cfg["banner_by_question"].get(var, "전체값만 출력")
        banner_tree = cfg["banner_groups"].get(banner_key, []) if banner_key != "전체값만 출력" else []
        tables[var] = build_question_table(
            df=df,
            dep_var=var,
            question_type=qtype,
            banner_tree=banner_tree,
            metadata=cfg["metadata"],
            show_subtotal=False,
            subtotal_groups=[],
            exclude_subtotal_vars=[],
            show_mean=qtype == "척도형",
            show_std=qtype == "척도형",
            scale_vars=cfg["scale_vars"],
            question_type_map=cfg["type_map"],
            rank_show_first=True,
            rank_show_topk=True,
            rank_show_mean=True,
            rank_top_k=2,
            **common,
        )

    for group_name, mr_vars in cfg["mr_group_map"].items():
        first_var = mr_vars[0]
        banner_key = cfg["banner_by_question"].get(first_var, "전체값만 출력")
        banner_tree = cfg["banner_groups"].get(banner_key, []) if banner_key != "전체값만 출력" else []
        tables[group_name] = build_multiresponse_table(
            df=df,
            mr_group_name=group_name,
            mr_vars=mr_vars,
            banner_tree=banner_tree,
            metadata=cfg["metadata"],
            mr_selected_mode="선택코드",
            mr_selected_codes=[1],
            **common,
        )

    if cfg["profile_vars"]:
        tables["응답자특성"] = build_profile_table(
            df=df,
            profile_vars=cfg["profile_vars"],
            metadata=cfg["metadata"],
            empty_include=True,
            missing_codes=[],
            pct_base="valid",
            missing_rules_by_var={},
            weight_col=weight_col,
        )

    return tables, {"weight_col": weight_col}


def page_files():
    st.header("1. 파일 준비")

    col1, col2 = st.columns(2)
    with col1:
        data_file = st.file_uploader("데이터 파일", type=["csv", "xlsx", "sav"], key="data_upload")
        if data_file is not None:
            st.session_state["data_file"] = data_file
    with col2:
        settings_file = st.file_uploader("조사설정 파일", type=["xlsx"], key="settings_upload")
        if settings_file is not None:
            st.session_state["settings_file"] = settings_file

    if TEMPLATE_PATH.exists():
        st.download_button(
            "표준 조사설정 템플릿 다운로드",
            TEMPLATE_PATH.read_bytes(),
            file_name="범용_표준_조사설정_템플릿.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with st.expander("콜럼가이드가 없는 경우: 설문지에서 조사설정 초안 만들기"):
        questionnaire_text = st.text_area("설문지 텍스트", height=260, placeholder="Q1. ...\n① ...\n② ...")
        if st.button("조사설정 초안 생성", type="primary"):
            guide_df = questionnaire_text_to_guide_df(questionnaire_text)
            if guide_df.empty:
                st.error("인식된 문항이 없습니다. 문항번호와 보기번호 형식을 확인해 주세요.")
            else:
                generated = build_settings_from_questionnaire(TEMPLATE_PATH, guide_df)
                st.success(f"{guide_df['문항/보기번호'].astype(str).str.match(r'^[A-Za-z]+').sum()}개 문항을 인식했습니다.")
                st.download_button(
                    "생성된 조사설정 파일 다운로드",
                    generated,
                    file_name="조사설정_초안.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    try:
        ctx = _load_context()
    except Exception as exc:
        st.error(str(exc))
        return

    if ctx:
        cfg = ctx["settings"]
        if cfg["errors"]:
            for error in cfg["errors"]:
                st.error(error)
        else:
            st.success(f"연결 완료: 응답 {len(ctx['df']):,}건 · 데이터 컬럼 {len(ctx['df'].columns):,}개 · 분석 문항 {len(cfg['dep_vars']) + len(cfg['mr_group_map']):,}개")
        for warning in cfg["warnings"]:
            st.warning(warning)


def page_tables():
    st.header("2. 통계표 추출")
    try:
        ctx = _load_context()
    except Exception as exc:
        st.error(str(exc))
        return
    if not ctx:
        st.info("먼저 파일 준비 단계에서 데이터와 조사설정 파일을 업로드해 주세요.")
        return
    cfg = ctx["settings"]
    if cfg["errors"]:
        for error in cfg["errors"]:
            st.error(error)
        return

    tables, info = _build_tables(ctx)
    st.metric("생성 대상 통계표", f"{len(tables):,}개")
    preview_name = st.selectbox("미리보기", list(tables.keys()))
    st.dataframe(tables[preview_name], use_container_width=True, hide_index=True)

    excel_bytes = export_to_excel(
        table_dict=tables,
        metadata=cfg["metadata"],
        title_bold=True,
        highlight_total=True,
        stat_fill_name="없음",
        pct_decimals=1,
        stat_decimals=2,
        add_excel_chart=False,
        n_in_parentheses=False,
        audit_trail=None,
        section_map=cfg.get("section_by_question", {}),
    )
    st.download_button(
        "통계표 엑셀 다운로드",
        excel_bytes,
        file_name=f"{DEFAULT_EXPORT_BASENAME}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


def page_visualization():
    st.header("3. 저장 / 시각화")
    try:
        ctx = _load_context()
    except Exception as exc:
        st.error(str(exc))
        return
    if not ctx:
        st.info("먼저 데이터와 조사설정 파일을 업로드해 주세요.")
        return
    cfg = ctx["settings"]
    if cfg["errors"]:
        for error in cfg["errors"]:
            st.error(error)
        return

    weight_col = _weight_column(ctx["df"])
    chart_results = build_chart_results(
        df=ctx["df"],
        dep_vars=cfg["dep_vars"],
        profile_vars=cfg["profile_vars"],
        metadata=cfg["metadata"],
        empty_include=True,
        missing_codes=[],
        pct_base="valid",
        mr_group_map=cfg["mr_group_map"],
        selected_mr_groups=cfg["selected_mr_groups"],
        mr_selected_mode="선택코드",
        mr_selected_codes=[1],
        mr_group_definitions={},
        question_type_map=cfg["type_map"],
        missing_rules_by_var={},
        weight_col=weight_col,
    )

    tab1, tab2 = st.tabs(["그래프", "대시보드"])
    with tab1:
        presets = get_chart_preset_names()
        fonts = get_available_korean_fonts() or ["자동 선택"]
        c1, c2, c3 = st.columns(3)
        preset = c1.selectbox("그래프 프리셋", presets)
        default_type = c2.selectbox("기본 그래프 유형", list(CHART_TYPE_OPTIONS.keys()))
        max_categories = c3.number_input("최대 보기 수", min_value=2, max_value=30, value=10)

        fresh = build_chart_settings_df(
            chart_results,
            default_chart_type=default_type,
            default_font=fonts[0],
        )
        current = st.session_state.get("chart_settings_df")
        if current is None or current.empty or set(current.get("question_id", [])) != set(fresh.get("question_id", [])):
            current = fresh
        edited = st.data_editor(
            current,
            use_container_width=True,
            hide_index=True,
            height=520,
            disabled=["question_id", "question_text", "recommended_chart"],
            column_config={
                "include": st.column_config.CheckboxColumn("포함"),
                "question_id": st.column_config.TextColumn("문항번호"),
                "question_text": st.column_config.TextColumn("문항명"),
                "chart_type": st.column_config.SelectboxColumn("그래프 유형", options=list(CHART_TYPE_OPTIONS.keys())),
                "font": st.column_config.SelectboxColumn("폰트", options=fonts),
            },
            column_order=["include", "question_id", "question_text", "chart_type"],
        )
        st.session_state["chart_settings_df"] = edited

        if st.button("그래프 생성", type="primary"):
            settings_dict = chart_settings_df_to_dict(edited)
            st.session_state["chart_files"] = build_all_charts(chart_results, settings_dict, show_n_label=False, max_categories=int(max_categories), section_map=cfg.get("section_by_question", {}), graph_settings=cfg.get("graph_settings", {}))

        chart_files = st.session_state.get("chart_files", {})
        if chart_files:
            first_name = next(iter(chart_files))
            st.image(chart_files[first_name], caption=first_name)
            st.download_button(
                "그래프 PNG 일괄 다운로드",
                build_zip_bytes_from_mapping({f"graphs/{k}": v for k, v in chart_files.items()}),
                file_name="survey_charts.zip",
                mime="application/zip",
            )

    with tab2:
        fresh_config = build_dashboard_config_df(cfg["dep_vars"], cfg["selected_mr_groups"], cfg["metadata"], question_type_map=cfg["type_map"], section_map=cfg.get("section_by_question", {}))
        dashboard_df = normalize_dashboard_config_df(st.session_state.get("dashboard_config_df"), cfg["dep_vars"], cfg["selected_mr_groups"], cfg["metadata"], question_type_map=cfg["type_map"], section_map=cfg.get("section_by_question", {}))
        st.session_state["dashboard_config_df"] = st.data_editor(
            dashboard_df, use_container_width=True, hide_index=True, height=520,
            disabled=["question_id", "question_type"],
            column_config={
                "include": st.column_config.CheckboxColumn("포함"),
                "order": st.column_config.NumberColumn("순서", min_value=1, step=1),
                "section": st.column_config.TextColumn("문항영역"),
                "question_id": st.column_config.TextColumn("문항번호"),
                "display_label": st.column_config.TextColumn("표시 문항명"),
                "default_chart": st.column_config.SelectboxColumn("기본 그래프", options=["가로 막대", "세로 막대", "도넛", "레이더"]),
            },
            column_order=["include", "order", "section", "question_id", "display_label", "default_chart"],
        )

        render_dashboard_preview(
            config_df=st.session_state["dashboard_config_df"],
            df=ctx["df"],
            metadata=cfg["metadata"],
            profile_vars=cfg["profile_vars"],
            missing_codes=[],
            pct_base="valid",
            mr_group_map=cfg["mr_group_map"],
            mr_selected_mode="선택코드",
            mr_selected_codes=[1],
            missing_rules_by_var={},
        )

        title = st.text_input("대시보드 제목", value=DEFAULT_DASHBOARD_APP_TITLE)
        subtitle = st.text_input("부제", value=DEFAULT_DASHBOARD_APP_SUBTITLE)
        package_name = st.text_input("패키지명", value=DEFAULT_DASHBOARD_PACKAGE_NAME)
        bundle = build_dashboard_bundle(
            config_df=st.session_state["dashboard_config_df"],
            df=ctx["df"],
            metadata=cfg["metadata"],
            profile_vars=cfg["profile_vars"],
            missing_codes=[],
            pct_base="valid",
            mr_group_map=cfg["mr_group_map"],
            mr_selected_mode="선택코드",
            mr_selected_codes=[1],
            app_title=title,
            app_subtitle=subtitle,
            preferred_subdomain=package_name,
            missing_rules_by_var={},
        )
        if st.button("고객용 Dashboard 폴더 생성", type="primary"):
            output_dir = ROOT / "output" / safe_filename(package_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            for relative_path, content in build_dashboard_deploy_files(bundle).items():
                target = output_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            st.success(f"생성 완료: {output_dir}")

        st.download_button(
            "Dashboard 폴더 ZIP 다운로드",
            build_dashboard_deploy_zip(bundle, package_name=package_name),
            file_name=f"{safe_filename(package_name)}.zip",
            mime="application/zip",
        )


def main():
    _init_state()
    st.title("Survey Tool")
    page = st.sidebar.radio("작업 단계", ["1. 파일 준비", "2. 통계표 추출", "3. 저장 / 시각화"])
    if page.startswith("1"):
        page_files()
    elif page.startswith("2"):
        page_tables()
    else:
        page_visualization()


if __name__ == "__main__":
    main()
