# -*- coding: utf-8 -*-
"""
[블록 C] 노코드 통합 예측 에이전트 빌드 - Streamlit 웹 UI
====================================================================
Step 9(통합 파이프라인 검증) + Step 10(위험 신호 자동 정렬 리포트)를
웹 화면 하나로 묶은 버전. 내부 로직(IntegratedPredictionAgent)은
block_c_integrated_agent.py와 완전히 동일하다.

■ 실행 방법
    1) pip install streamlit pandas scikit-learn openpyxl
    2) streamlit run block_c_app.py
    3) 브라우저가 자동으로 안 열리면 http://localhost:8501 접속

■ 필요한 파일 (이 파일과 같은 폴더에 둘 것)
    master_dataset.csv 또는 .xlsx     — 에이전트 학습용 데이터
    블록C_검증서류_샘플.csv            — Step 9 검증용 (화면에서 다른 파일 업로드 가능)
    선적서류_샘플.csv                  — Step 10 리포트용 (화면에서 다른 파일 업로드 가능)
"""

import os

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="블록 C · 통합 예측 에이전트", layout="wide")

NAVY = "#0B2545"
TEAL = "#028090"
RISK_BG = "#FCEBEB"
RISK_TEXT = "#A32D2D"
SAFE_BG = "#E1F5EE"

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER_REQUIRED = ["상품영문명", "사양", "HS코드_정답", "협력사_환경위반건수", "협력사_노동평가점수", "리스크_정답"]
DOC_REQUIRED = ["상품영문명", "사양", "협력사_환경위반건수", "협력사_노동평가점수"]

# 교육용 예시 워치리스트 — 실제 제재·전략물자 리스트가 아님
EXAMPLE_WATCHLIST_HS_CODES = {"853400"}

st.title("블록 C — 노코드 통합 예측 에이전트")
st.caption(
    "HS Code 모델과 ESG 위험도 모델을 문서번호 기준으로 병합해 하나의 파이프라인으로 묶고 "
    "(Step 9), 위험도순 정렬·임계값 필터로 '오늘 검토가 필요한 건'만 자동으로 걸러냅니다 (Step 10)."
)


# ------------------------------------------------------------------
# 에이전트 학습 (마스터 데이터셋으로 한 번만)
# ------------------------------------------------------------------
st.header("에이전트 빌드 — 학습 데이터 확인")

uploaded_master = st.file_uploader(
    "마스터 데이터셋 업로드 (업로드하지 않으면 같은 폴더의 master_dataset.csv/.xlsx를 자동 사용)",
    type=["csv", "xlsx"], key="master_file",
)

if uploaded_master is not None:
    df_master = pd.read_excel(uploaded_master) if uploaded_master.name.lower().endswith("xlsx") \
        else pd.read_csv(uploaded_master, encoding="utf-8-sig")
    master_source = f"업로드한 파일: {uploaded_master.name}"
elif os.path.exists(os.path.join(HERE, "master_dataset.csv")):
    df_master = pd.read_csv(os.path.join(HERE, "master_dataset.csv"), encoding="utf-8-sig")
    master_source = "기본 제공 파일: master_dataset.csv"
elif os.path.exists(os.path.join(HERE, "master_dataset.xlsx")):
    df_master = pd.read_excel(os.path.join(HERE, "master_dataset.xlsx"))
    master_source = "기본 제공 파일: master_dataset.xlsx"
else:
    df_master = None
    master_source = None

if df_master is None:
    st.warning("마스터 데이터셋을 업로드해주세요.")
    st.stop()

missing = [c for c in MASTER_REQUIRED if c not in df_master.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()

st.caption(f"데이터 출처: {master_source} · 총 {len(df_master)}행")


@st.cache_resource(show_spinner=False)
def build_agent(df_master: pd.DataFrame):
    df = df_master.copy()
    df["상품설명"] = df["상품영문명"].astype(str) + " " + df["사양"].astype(str)

    y_hs = df["HS코드_정답"].astype(str)
    Xtr_t, Xte_t, ytr_hs, yte_hs = train_test_split(
        df["상품설명"], y_hs, test_size=0.2, random_state=42, stratify=y_hs
    )
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    Xtr_vec = tfidf.fit_transform(Xtr_t)
    Xte_vec = tfidf.transform(Xte_t)
    hs_model = RandomForestClassifier(n_estimators=200, random_state=42)
    hs_model.fit(Xtr_vec, ytr_hs)
    hs_acc = accuracy_score(yte_hs, hs_model.predict(Xte_vec))

    X_esg = df[["협력사_환경위반건수", "협력사_노동평가점수"]]
    y_esg = (df["리스크_정답"].astype(str).str.strip() == "위험").astype(int)
    esg_model = LogisticRegression(random_state=42)
    esg_model.fit(X_esg, y_esg)

    return {"tfidf": tfidf, "hs_model": hs_model, "hs_acc": hs_acc, "esg_model": esg_model}


def agent_predict(agent, docs_df, id_col):
    """Step 9: 병렬 예측 후 문서번호(id_col) 기준으로 병합."""
    docs = docs_df.copy()
    docs["상품설명"] = docs["상품영문명"].astype(str) + " " + docs["사양"].astype(str)

    vec = agent["tfidf"].transform(docs["상품설명"])
    hs_result = docs[[id_col]].copy()
    hs_result["예측_HS코드"] = agent["hs_model"].predict(vec)

    X_esg = docs[["협력사_환경위반건수", "협력사_노동평가점수"]]
    esg_result = docs[[id_col]].copy()
    esg_result["위험_확률"] = agent["esg_model"].predict_proba(X_esg)[:, 1]

    merged = docs.merge(hs_result, on=id_col).merge(esg_result, on=id_col)
    merged["워치리스트_관찰대상"] = merged["예측_HS코드"].apply(
        lambda c: "예" if c in EXAMPLE_WATCHLIST_HS_CODES else "-"
    )
    return merged


with st.spinner("HS Code 모델과 ESG 위험도 모델을 학습하는 중입니다..."):
    agent = build_agent(df_master)

m1, m2 = st.columns(2)
m1.metric("HS Code 모델 테스트 정확도", f"{agent['hs_acc']:.1%}")
m2.metric("ESG 위험도 모델", "학습 완료")

st.divider()


# ------------------------------------------------------------------
# Step 9. 통합 예측 파이프라인 검증
# ------------------------------------------------------------------
st.header("Step 9. 통합 예측 파이프라인 검증")
st.caption("오늘 사용하지 않은 신규 서류로 HS Code 추천값과 위험 예측 확률이 문서번호 기준으로 정확히 합쳐지는지 확인합니다.")

uploaded_docs = st.file_uploader(
    "검증용 신규 서류 CSV 업로드 (필수 컬럼: 문서번호, 상품영문명, 사양, 협력사_환경위반건수, 협력사_노동평가점수)",
    type=["csv"], key="docs_file",
)

default_docs = os.path.join(HERE, "블록C_검증서류_샘플.csv")

if uploaded_docs is not None:
    docs_df = pd.read_csv(uploaded_docs, encoding="utf-8-sig")
    docs_source = f"업로드한 파일: {uploaded_docs.name}"
elif os.path.exists(default_docs):
    docs_df = pd.read_csv(default_docs, encoding="utf-8-sig")
    docs_source = "기본 제공 파일: 블록C_검증서류_샘플.csv"
else:
    docs_df = None
    docs_source = None

if docs_df is None:
    st.info("검증용 신규 서류 CSV를 업로드해주세요.")
elif "문서번호" not in docs_df.columns:
    st.error("업로드한 파일에 '문서번호' 컬럼이 없습니다. 병합 기준 컬럼이 반드시 필요합니다.")
else:
    missing_doc = [c for c in DOC_REQUIRED if c not in docs_df.columns]
    if missing_doc:
        st.error(f"필수 컬럼이 없습니다: {missing_doc}")
    else:
        st.caption(f"검증 서류 출처: {docs_source} · 총 {len(docs_df)}건")
        merged_docs = agent_predict(agent, docs_df, id_col="문서번호")
        show_cols = ["문서번호", "상품영문명", "예측_HS코드", "위험_확률", "워치리스트_관찰대상"]
        preview = merged_docs[show_cols].copy()
        preview["위험_확률"] = (preview["위험_확률"] * 100).round(1).astype(str) + "%"
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.success("문서번호 기준 병합 완료 — 각 행에 HS Code 추천값과 위험 확률이 함께 표시됩니다.")

st.divider()


# ------------------------------------------------------------------
# Step 10. 위험 신호 자동 정렬 리포트
# ------------------------------------------------------------------
st.header("Step 10. 위험 신호 자동 정렬 리포트")

threshold = st.slider(
    "리포트 임계값 (3.7 Step 7에서 정한 값)",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="이 값 이상인 건만 '오늘 검토가 필요한 건'으로 분류됩니다.",
)

uploaded_ship = st.file_uploader(
    "이번 주 선적 서류 CSV 업로드 (컬럼: 선적ID 또는 문서번호, 상품영문명, 사양, 협력사_환경위반건수, 협력사_노동평가점수)",
    type=["csv"], key="ship_file",
)

default_ship = os.path.join(HERE, "선적서류_샘플.csv")

if uploaded_ship is not None:
    ships_df = pd.read_csv(uploaded_ship, encoding="utf-8-sig")
    ship_source = f"업로드한 파일: {uploaded_ship.name}"
elif os.path.exists(default_ship):
    ships_df = pd.read_csv(default_ship, encoding="utf-8-sig")
    ship_source = "기본 제공 파일: 선적서류_샘플.csv"
else:
    ships_df = None
    ship_source = None

if ships_df is None:
    st.info("이번 주 선적 서류 CSV를 업로드해주세요.")
else:
    if "선적ID" in ships_df.columns and "문서번호" not in ships_df.columns:
        ships_df = ships_df.rename(columns={"선적ID": "문서번호"})

    if "문서번호" not in ships_df.columns:
        st.error("업로드한 파일에 '선적ID' 또는 '문서번호' 컬럼이 없습니다.")
    else:
        missing_ship = [c for c in DOC_REQUIRED if c not in ships_df.columns]
        if missing_ship:
            st.error(f"필수 컬럼이 없습니다: {missing_ship}")
        else:
            st.caption(f"선적 서류 출처: {ship_source} · 총 {len(ships_df)}건")

            merged_ships = agent_predict(agent, ships_df, id_col="문서번호")
            sorted_all = merged_ships.sort_values("위험_확률", ascending=False).reset_index(drop=True)
            sorted_all["판정"] = sorted_all["위험_확률"].apply(lambda p: "위험" if p >= threshold else "정상")
            review_needed = sorted_all[sorted_all["판정"] == "위험"].reset_index(drop=True)

            mc1, mc2 = st.columns(2)
            mc1.metric("오늘 검토가 필요한 건", f"{len(review_needed)}건")
            mc2.metric("전체 건수", f"{len(sorted_all)}건")

            display_cols = ["문서번호", "상품영문명", "예측_HS코드", "위험_확률", "판정", "워치리스트_관찰대상"]
            show_df = sorted_all[display_cols].copy()
            show_df["위험_확률"] = (show_df["위험_확률"] * 100).round(1).astype(str) + "%"

            def highlight(row):
                if row["판정"] == "위험":
                    return [f"background-color: {RISK_BG}; color: {RISK_TEXT}; font-weight: 600"] * len(row)
                return [f"background-color: {SAFE_BG}"] * len(row)

            st.markdown("**전체 정렬 결과 (위험도 높은 순)**")
            st.dataframe(show_df.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)

            st.markdown("**오늘 검토가 필요한 건만 모은 최종 리포트**")
            if len(review_needed) > 0:
                review_display = review_needed[["문서번호", "상품영문명", "예측_HS코드", "위험_확률", "워치리스트_관찰대상"]].copy()
                review_display["위험_확률"] = (review_display["위험_확률"] * 100).round(1).astype(str) + "%"
                st.dataframe(review_display, use_container_width=True, hide_index=True)

                csv_bytes = review_needed.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "오늘의 검토 리포트 CSV 다운로드",
                    data=csv_bytes, file_name="오늘의_검토리포트.csv", mime="text/csv",
                )
            else:
                st.info("임계값을 넘는 위험 건이 없습니다.")

            st.caption(
                "워치리스트_관찰대상은 교육용 예시 HS Code 목록 기준입니다. "
                "실제 스크리닝은 전략물자관리원·OFAC SDN 등 공식 리스트 연동이 필요합니다."
            )
