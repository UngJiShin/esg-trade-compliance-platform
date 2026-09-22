# -*- coding: utf-8 -*-
"""
[블록 B] ESG 위험도 예측 및 선적 서류 스크리닝 - Streamlit 웹 UI
====================================================================
Orange3의 Step 5~8을 웹 화면 하나로 묶은 버전.
[블록 A]와 마찬가지로 내부 로직(LogisticRegression + RandomForest)은
block_b_esg_screening.py와 완전히 동일하다.

■ 실행 방법
    1) pip install streamlit pandas scikit-learn openpyxl
    2) streamlit run block_b_app.py
    3) 브라우저가 자동으로 안 열리면 http://localhost:8501 접속

■ 필요한 파일 (이 파일과 같은 폴더에 둘 것)
    master_dataset.csv 또는 .xlsx   — Step 5 학습용 데이터
    선적서류_샘플.csv                — Step 8에서 쓸 선적 서류 묶음
    (화면에서 다른 파일을 업로드하면 그 파일로 다시 계산합니다)
"""

import os

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="블록 B · ESG 위험도 예측 및 선적서류 스크리닝", layout="wide")

NAVY = "#0B2545"
TEAL = "#028090"
RISK_BG = "#FCEBEB"
RISK_TEXT = "#A32D2D"

MASTER_REQUIRED = ["상품영문명", "사양", "HS코드_정답", "협력사_환경위반건수", "협력사_노동평가점수", "리스크_정답"]
SHIP_REQUIRED = ["선적ID", "상품영문명", "사양", "협력사_환경위반건수", "협력사_노동평가점수"]

# 교육용 예시 워치리스트 — 실제 제재·전략물자 리스트가 아님 (block_b_esg_screening.py와 동일)
EXAMPLE_WATCHLIST_HS_CODES = {"853400"}

HERE = os.path.dirname(os.path.abspath(__file__))

st.title("블록 B — ESG 위험도 예측 및 선적 서류 스크리닝")
st.caption("Step 5~8 (모델 구성 → Confusion Matrix → 임계값 조정 → 선적 서류 스크리닝)을 화면 하나로 실습합니다.")


# ------------------------------------------------------------------
# Step 5. 마스터 데이터셋 불러오기 + 모델 학습
# ------------------------------------------------------------------
st.header("Step 5. 학습 데이터 확인 및 ESG 위험도 모델 구성")

uploaded_master = st.file_uploader(
    "마스터 데이터셋 업로드 (업로드하지 않으면 같은 폴더의 master_dataset.csv/.xlsx를 자동 사용)",
    type=["csv", "xlsx"],
    key="master_file",
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
def train_models(df: pd.DataFrame):
    df = df.copy()

    # HS Code 모델 (블록 A와 동일 로직 — Step 8 스크리닝에서 함께 씀)
    df["상품설명"] = df["상품영문명"].astype(str) + " " + df["사양"].astype(str)
    y_hs = df["HS코드_정답"].astype(str)
    strat_hs = y_hs if y_hs.value_counts().min() >= 2 else None
    Xtr_t, Xte_t, ytr_hs, yte_hs = train_test_split(
        df["상품설명"], y_hs, test_size=0.2, random_state=42, stratify=strat_hs
    )
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    Xtr_vec = tfidf.fit_transform(Xtr_t)
    Xte_vec = tfidf.transform(Xte_t)
    hs_model = RandomForestClassifier(n_estimators=200, random_state=42)
    hs_model.fit(Xtr_vec, ytr_hs)
    hs_acc = accuracy_score(yte_hs, hs_model.predict(Xte_vec))

    # ESG 위험도 모델 (Step 5)
    X_esg = df[["협력사_환경위반건수", "협력사_노동평가점수"]]
    y_esg = (df["리스크_정답"].astype(str).str.strip() == "위험").astype(int)
    strat_esg = y_esg if y_esg.value_counts().min() >= 2 else None
    Xtr_e, Xte_e, ytr_e, yte_e = train_test_split(
        X_esg, y_esg, test_size=0.2, random_state=42, stratify=strat_esg
    )
    esg_model = LogisticRegression(random_state=42)
    esg_model.fit(Xtr_e, ytr_e)
    proba_test = esg_model.predict_proba(Xte_e)[:, 1]

    return {
        "tfidf": tfidf, "hs_model": hs_model, "hs_acc": hs_acc,
        "esg_model": esg_model, "y_test_esg": yte_e.to_numpy(), "esg_proba_test": proba_test,
        "n_train_esg": len(Xtr_e), "n_test_esg": len(Xte_e),
    }


with st.spinner("HS Code 모델과 ESG 위험도 모델을 학습하는 중입니다..."):
    m = train_models(df_master)

c1, c2, c3 = st.columns(3)
c1.metric("HS Code 모델 테스트 정확도", f"{m['hs_acc']:.1%}")
c2.metric("ESG 학습 데이터", f"{m['n_train_esg']}건")
c3.metric("ESG 테스트 데이터", f"{m['n_test_esg']}건")

st.divider()


# ------------------------------------------------------------------
# Step 6~7. 임계값 설정 + Confusion Matrix / Precision·Recall
# ------------------------------------------------------------------
st.header("Step 6~7. 판정 기준(threshold) 설정 및 모델 검증")

threshold = st.slider(
    "ESG 위험 판정 임계값 (threshold)",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="이 값을 움직이면 아래 Confusion Matrix와 Step 8 스크리닝 결과가 실시간으로 바뀝니다.",
)

y_test_esg = m["y_test_esg"]
proba_test = m["esg_proba_test"]
y_pred_test = (proba_test >= threshold).astype(int)

cm = confusion_matrix(y_test_esg, y_pred_test, labels=[0, 1])
cm_df = pd.DataFrame(cm, index=["실제_정상", "실제_위험"], columns=["예측_정상", "예측_위험"])
precision = precision_score(y_test_esg, y_pred_test, zero_division=0)
recall = recall_score(y_test_esg, y_pred_test, zero_division=0)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Confusion Matrix**")
    st.dataframe(cm_df, use_container_width=True)
with col2:
    st.markdown("**Precision · Recall**")
    st.dataframe(
        pd.DataFrame({"지표": ["Precision", "Recall"], "값": [round(precision, 3), round(recall, 3)]}),
        use_container_width=True, hide_index=True,
    )
    if recall < 0.7:
        st.warning("Recall이 낮습니다 — 실제 위험 건을 놓치는 비율이 높아 컴플라이언스 관점에서 주의가 필요합니다.")

# 임계값별 비교표 (Step 7)
with st.expander("임계값별 Precision · Recall 비교표 (0.7 / 0.5 / 0.3)"):
    rows = []
    for th in [0.7, 0.5, 0.3]:
        yp = (proba_test >= th).astype(int)
        rows.append({
            "임계값": th,
            "Precision": round(precision_score(y_test_esg, yp, zero_division=0), 3),
            "Recall": round(recall_score(y_test_esg, yp, zero_division=0), 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()


# ------------------------------------------------------------------
# Step 8. 선적 서류 스크리닝
# ------------------------------------------------------------------
st.header("Step 8. 선적 서류 스크리닝 자동 적용")

uploaded_ship = st.file_uploader(
    "선적 서류 CSV 업로드 (업로드하지 않으면 같은 폴더의 선적서류_샘플.csv를 자동 사용)",
    type=["csv"],
    key="ship_file",
)

default_ship = os.path.join(HERE, "선적서류_샘플.csv")

if uploaded_ship is not None:
    ships = pd.read_csv(uploaded_ship, encoding="utf-8-sig")
    ship_source = f"업로드한 파일: {uploaded_ship.name}"
elif os.path.exists(default_ship):
    ships = pd.read_csv(default_ship, encoding="utf-8-sig")
    ship_source = "기본 제공 파일: 선적서류_샘플.csv"
else:
    ships = None
    ship_source = None

if ships is None:
    st.info("선적 서류 CSV를 업로드해주세요.")
else:
    missing_ship = [c for c in SHIP_REQUIRED if c not in ships.columns]
    if missing_ship:
        st.error(f"선적 서류 CSV에 필수 컬럼이 없습니다: {missing_ship}")
    else:
        st.caption(f"선적 서류 출처: {ship_source} · 총 {len(ships)}건")

        items = ships.copy()
        items["상품설명"] = items["상품영문명"].astype(str) + " " + items["사양"].astype(str)

        vec = m["tfidf"].transform(items["상품설명"])
        items["예측_HS코드"] = m["hs_model"].predict(vec)

        X_new_esg = items[["협력사_환경위반건수", "협력사_노동평가점수"]]
        items["위험_확률"] = m["esg_model"].predict_proba(X_new_esg)[:, 1]
        items["판정"] = items["위험_확률"].apply(lambda p: "위험" if p >= threshold else "정상")
        items["워치리스트_관찰대상"] = items["예측_HS코드"].apply(
            lambda code: "예" if code in EXAMPLE_WATCHLIST_HS_CODES else "-"
        )

        items_sorted = items.sort_values("위험_확률", ascending=False).reset_index(drop=True)
        risky_count = int((items_sorted["판정"] == "위험").sum())

        mc1, mc2 = st.columns(2)
        mc1.metric("이번 주 위험 건수", f"{risky_count}건")
        mc2.metric("전체 건수", f"{len(items_sorted)}건")

        display_cols = ["선적ID", "상품영문명", "예측_HS코드", "위험_확률", "판정", "워치리스트_관찰대상"]
        show_df = items_sorted[display_cols].copy()
        show_df["위험_확률"] = (show_df["위험_확률"] * 100).round(1).astype(str) + "%"

        def highlight_risk(row):
            if row["판정"] == "위험":
                return [f"background-color: {RISK_BG}; color: {RISK_TEXT}; font-weight: 600"] * len(row)
            return [""] * len(row)

        st.dataframe(
            show_df.style.apply(highlight_risk, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "워치리스트_관찰대상은 교육용 예시 HS Code 목록 기준입니다. "
            "실제 스크리닝은 전략물자관리원·OFAC SDN 등 공식 리스트 연동이 필요합니다."
        )

        csv_bytes = items_sorted.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "스크리닝 결과 CSV 다운로드",
            data=csv_bytes,
            file_name="screening_result.csv",
            mime="text/csv",
        )
