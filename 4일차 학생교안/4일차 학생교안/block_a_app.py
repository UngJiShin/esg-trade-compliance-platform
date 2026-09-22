# -*- coding: utf-8 -*-
"""
[블록 A] HS Code 추천 모델 - Streamlit 웹 UI
================================================
Orange3의 Step 1~4(File → Preprocess Text → Bag of Words → Tree/kNN →
Test & Score → Predictions)를 웹 화면 하나로 묶은 버전.

■ 실행 방법
    1) pip install streamlit pandas scikit-learn openpyxl
    2) streamlit run block_a_app.py
    3) 브라우저가 자동으로 안 열리면 http://localhost:8501 접속

■ 필요한 파일
    master_dataset.csv 또는 master_dataset.xlsx — 이 파일과 같은 폴더에 둘 것
    (화면에서 다른 데이터셋을 업로드하면 그 파일로 다시 학습합니다)
"""

import os

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="블록 A · HS Code 추천 모델", layout="wide")

NAVY = "#0B2545"
TEAL = "#028090"
MINT = "#02C39A"

REQUIRED_COLS = ["상품영문명", "사양", "HS코드_정답"]

st.title("블록 A — HS Code 추천 모델")
st.caption("Step 1~4 (File → 텍스트 인코딩 → 분류 모델 학습 → 신규 상품 예측)를 화면 하나로 실습합니다.")


# ------------------------------------------------------------------
# Step 1. 마스터 데이터셋 불러오기
# ------------------------------------------------------------------
st.header("Step 1. 학습 데이터 확인")

uploaded = st.file_uploader(
    "마스터 데이터셋 업로드 (업로드하지 않으면 같은 폴더의 master_dataset.csv/.xlsx를 자동 사용)",
    type=["csv", "xlsx"],
    key="master_file",
)

here = os.path.dirname(os.path.abspath(__file__))
default_csv = os.path.join(here, "master_dataset.csv")
default_xlsx = os.path.join(here, "master_dataset.xlsx")

if uploaded is not None:
    if uploaded.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded, encoding="utf-8-sig")
    source = f"업로드한 파일: {uploaded.name}"
elif os.path.exists(default_csv):
    df = pd.read_csv(default_csv, encoding="utf-8-sig")
    source = "기본 제공 파일: master_dataset.csv"
elif os.path.exists(default_xlsx):
    df = pd.read_excel(default_xlsx)
    source = "기본 제공 파일: master_dataset.xlsx"
else:
    df = None
    source = None

if df is None:
    st.warning("마스터 데이터셋을 업로드해주세요.")
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()

st.caption(f"데이터 출처: {source} · 총 {len(df)}행 · HS Code 종류 {df['HS코드_정답'].nunique()}개")
with st.expander("데이터 미리보기 (상위 10행)"):
    st.dataframe(df[REQUIRED_COLS].head(10), use_container_width=True)

st.divider()


# ------------------------------------------------------------------
# Step 2~3. 텍스트 인코딩 + 분류 모델 학습 (캐시)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_hs_model(df: pd.DataFrame):
    df = df.copy()
    df["상품설명"] = df["상품영문명"].astype(str) + " " + df["사양"].astype(str)
    y = df["HS코드_정답"].astype(str)
    strat = y if y.value_counts().min() >= 2 else None

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["상품설명"], y, test_size=0.2, random_state=42, stratify=strat
    )

    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train_text)
    X_test_vec = tfidf.transform(X_test_text)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_vec, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_vec))
    return tfidf, model, acc, len(y_train), len(y_test)


st.header("Step 2~3. 텍스트 인코딩 및 분류 모델 학습")
with st.spinner("텍스트를 벡터화하고 모델을 학습하는 중입니다..."):
    tfidf, model, acc, n_train, n_test = train_hs_model(df)

m1, m2, m3 = st.columns(3)
m1.metric("학습 데이터", f"{n_train}건")
m2.metric("테스트 데이터", f"{n_test}건")
m3.metric("테스트 정확도", f"{acc:.1%}")

st.divider()


# ------------------------------------------------------------------
# Step 4. 신규 상품 HS Code 예측
# ------------------------------------------------------------------
st.header("Step 4. 신규 상품 HS Code 예측")

input_mode = st.radio("입력 방식", ["직접 입력", "CSV 업로드"], horizontal=True)

new_items = None

if input_mode == "직접 입력":
    col1, col2 = st.columns(2)
    with col1:
        name_in = st.text_input("상품영문명", value="Cotton Graphic T-Shirt")
    with col2:
        spec_in = st.text_input("사양", value="cotton-polyester blend, short sleeve, export packaging standard")
    new_items = pd.DataFrame([{"상품영문명": name_in, "사양": spec_in}])
else:
    new_file = st.file_uploader(
        "신규 상품 CSV 업로드 (필수 컬럼: 상품영문명, 사양)", type=["csv"], key="new_items"
    )
    if new_file is not None:
        new_items = pd.read_csv(new_file, encoding="utf-8-sig")
        missing_new = [c for c in ["상품영문명", "사양"] if c not in new_items.columns]
        if missing_new:
            st.error(f"CSV에 필요한 컬럼이 없습니다: {missing_new}")
            new_items = None
        else:
            st.dataframe(new_items, use_container_width=True)

run = st.button("예측 실행", type="primary")

if run:
    if new_items is None or len(new_items) == 0:
        st.warning("먼저 신규 상품 정보를 입력하거나 CSV를 업로드해주세요.")
    else:
        items = new_items.copy()
        items["상품설명"] = items["상품영문명"].astype(str) + " " + items["사양"].astype(str)
        vec = tfidf.transform(items["상품설명"])
        items["예측_HS코드"] = model.predict(vec)
        st.session_state["block_a_predictions"] = items[["상품영문명", "사양", "예측_HS코드"]]

st.subheader("예측 결과")

if "block_a_predictions" in st.session_state:
    result = st.session_state["block_a_predictions"]
    for _, row in result.iterrows():
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f"""<div style="padding:20px;border-radius:12px;background:{NAVY};
                color:white;height:100%;">
                <div style="font-size:13px;opacity:0.75;">상품명 / 사양</div>
                <div style="font-size:16px;font-weight:700;margin-top:6px;">{row['상품영문명']}</div>
                <div style="font-size:13px;opacity:0.85;margin-top:4px;">{row['사양']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div style="padding:20px;border-radius:12px;background:{TEAL};
                color:white;text-align:center;height:100%;">
                <div style="font-size:13px;opacity:0.85;">추천 HS Code</div>
                <div style="font-size:32px;font-weight:700;margin-top:6px;">{row['예측_HS코드']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.write("")
    st.caption("체크리스트: 위 예측 결과를 실제 관세사 자문 결과(또는 강사 제공 정답)와 비교해 기록하세요.")
else:
    st.info("신규 상품 정보를 입력한 뒤 '예측 실행' 버튼을 눌러주세요.")
