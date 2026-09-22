# -*- coding: utf-8 -*-
"""
[3일차 실습] HS Code 추천 + ESG 위험도 예측 - Streamlit 웹 앱
============================================================
■ 실행 방법
  1) 이 파일(app.py)과 master_dataset.csv를 같은 폴더에 둔다.
  2) 터미널에서 해당 폴더로 이동한 뒤, 필요한 라이브러리를 설치한다 (최초 1회):
       pip install streamlit pandas scikit-learn
  3) 아래 명령어로 실행한다:
       streamlit run app.py
  4) 브라우저가 자동으로 열리지 않으면, 터미널에 표시되는
     http://localhost:8501 주소를 직접 열어 접속한다.

■ 화면 구성
  1. 학습용 마스터 데이터셋(CSV) 업로드
  2. 신규 상품 정보 입력 (직접 입력 / CSV 업로드)
  3. '예측 실행' 버튼 → 추천 HS Code, ESG 위험 확률·판정을 카드로 표시
  4. 임계값(threshold) 슬라이더 → 정상/위험 판정이 실시간으로 갱신
  5. 모델 성능(Confusion Matrix, Precision·Recall) 표
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="HS Code · ESG 위험도 예측", layout="wide")

NAVY = "#0B2545"
TEAL = "#028090"
RISK_COLOR = "#C9432E"
SAFE_COLOR = "#02C39A"

MASTER_REQUIRED_COLS = [
    "상품영문명", "사양", "HS코드_정답", "협력사_환경위반건수", "협력사_노동평가점수", "리스크_정답",
]
NEW_ITEM_REQUIRED_COLS = ["상품영문명", "사양", "협력사_환경위반건수", "협력사_노동평가점수"]

st.title("HS Code 추천 · ESG 위험도 예측")
st.caption("마스터 데이터셋으로 학습한 두 개의 독립 모델 — HS Code 분류 모델 / ESG 위험도 분류 모델")


# ------------------------------------------------------------------
# 1. 학습용 마스터 데이터셋 업로드
# ------------------------------------------------------------------
st.header("1. 학습용 마스터 데이터셋 업로드")

uploaded_master = st.file_uploader(
    "마스터 데이터셋 CSV를 업로드하세요 (업로드하지 않으면 같은 폴더의 master_dataset.csv를 자동으로 사용합니다)",
    type=["csv"],
    key="master_csv",
)

default_master_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_dataset.csv")

if uploaded_master is not None:
    df_master = pd.read_csv(uploaded_master, encoding="utf-8-sig")
    data_source = f"업로드한 파일: {uploaded_master.name}"
elif os.path.exists(default_master_path):
    df_master = pd.read_csv(default_master_path, encoding="utf-8-sig")
    data_source = "기본 제공 파일: master_dataset.csv"
else:
    df_master = None
    data_source = None

if df_master is None:
    st.warning("마스터 데이터셋 CSV를 업로드해야 다음 단계로 진행할 수 있습니다.")
    st.stop()

missing_cols = [c for c in MASTER_REQUIRED_COLS if c not in df_master.columns]
if missing_cols:
    st.error(f"마스터 데이터셋에 필수 컬럼이 없습니다: {missing_cols}")
    st.stop()

st.caption(f"데이터 출처: {data_source}  ·  총 {len(df_master)}행")
with st.expander("마스터 데이터셋 미리보기 (상위 10행)"):
    st.dataframe(df_master.head(10), use_container_width=True)


# ------------------------------------------------------------------
# 모델 학습 (캐시: 같은 데이터로는 한 번만 학습)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    df = df.copy()

    # --- HS Code 추천 모델: 텍스트 → TF-IDF → RandomForest ---
    df["상품설명"] = df["상품영문명"].astype(str) + " " + df["사양"].astype(str)
    y_hs = df["HS코드_정답"].astype(str)
    strat_hs = y_hs if y_hs.value_counts().min() >= 2 else None
    X_train_text, X_test_text, y_train_hs, y_test_hs = train_test_split(
        df["상품설명"], y_hs, test_size=0.2, random_state=42, stratify=strat_hs
    )
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train_text)
    X_test_vec = tfidf.transform(X_test_text)

    hs_model = RandomForestClassifier(n_estimators=200, random_state=42)
    hs_model.fit(X_train_vec, y_train_hs)
    hs_test_acc = accuracy_score(y_test_hs, hs_model.predict(X_test_vec))

    # --- ESG 위험도 예측 모델: 환경위반건수·노동점수 → 로지스틱회귀 ---
    X_esg = df[["협력사_환경위반건수", "협력사_노동평가점수"]]
    y_esg = (df["리스크_정답"].astype(str).str.strip() == "위험").astype(int)
    strat_esg = y_esg if y_esg.value_counts().min() >= 2 else None
    X_train_esg, X_test_esg, y_train_esg, y_test_esg = train_test_split(
        X_esg, y_esg, test_size=0.2, random_state=42, stratify=strat_esg
    )
    esg_model = LogisticRegression(random_state=42)
    esg_model.fit(X_train_esg, y_train_esg)
    esg_test_proba = esg_model.predict_proba(X_test_esg)[:, 1]

    return {
        "tfidf": tfidf,
        "hs_model": hs_model,
        "hs_test_acc": hs_test_acc,
        "esg_model": esg_model,
        "y_test_esg": y_test_esg.to_numpy(),
        "esg_test_proba": esg_test_proba,
    }


with st.spinner("두 모델을 학습하는 중입니다..."):
    models = train_models(df_master)

st.success(f"학습 완료 — HS Code 모델 테스트 정확도: {models['hs_test_acc']:.1%} (테스트셋 기준)")

st.divider()


# ------------------------------------------------------------------
# 2. 신규 상품 정보 입력
# ------------------------------------------------------------------
st.header("2. 신규 상품 정보 입력")

input_mode = st.radio("입력 방식을 선택하세요", ["직접 입력", "CSV 업로드"], horizontal=True)

new_items = None

if input_mode == "직접 입력":
    col1, col2 = st.columns(2)
    with col1:
        name_in = st.text_input("상품영문명", value="Cotton Graphic T-Shirt")
        spec_in = st.text_input(
            "사양", value="cotton-polyester blend, short sleeve, export packaging standard"
        )
    with col2:
        env_in = st.number_input("협력사 환경위반건수", min_value=0, max_value=20, value=1, step=1)
        labor_in = st.number_input("협력사 노동평가점수", min_value=0, max_value=100, value=75, step=1)

    new_items = pd.DataFrame([{
        "상품영문명": name_in,
        "사양": spec_in,
        "협력사_환경위반건수": env_in,
        "협력사_노동평가점수": labor_in,
    }])

else:
    new_file = st.file_uploader(
        "신규 상품 CSV 업로드 (필수 컬럼: 상품영문명, 사양, 협력사_환경위반건수, 협력사_노동평가점수)",
        type=["csv"],
        key="new_items_csv",
    )
    if new_file is not None:
        new_items = pd.read_csv(new_file, encoding="utf-8-sig")
        missing_new = [c for c in NEW_ITEM_REQUIRED_COLS if c not in new_items.columns]
        if missing_new:
            st.error(f"업로드한 CSV에 필수 컬럼이 없습니다: {missing_new}")
            new_items = None
        else:
            st.dataframe(new_items, use_container_width=True)

st.divider()


# ------------------------------------------------------------------
# 3~4. 판정 기준(threshold) 설정 + 예측 실행
# ------------------------------------------------------------------
st.header("3. 판정 기준 설정 및 예측 실행")

threshold = st.slider(
    "ESG 위험 판정 임계값 (threshold)",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="이 값 이상이면 '위험'으로 판정합니다. 슬라이더를 움직이면 아래 판정 결과와 성능 표가 실시간으로 바뀝니다.",
)

run = st.button("예측 실행", type="primary")

if run:
    if new_items is None or len(new_items) == 0:
        st.warning("먼저 신규 상품 정보를 입력하거나 CSV를 업로드해주세요.")
    else:
        items = new_items.copy()
        items["상품설명"] = items["상품영문명"].astype(str) + " " + items["사양"].astype(str)

        # HS Code 예측
        vec = models["tfidf"].transform(items["상품설명"])
        items["예측_HS코드"] = models["hs_model"].predict(vec)

        # ESG 위험 확률 예측 (판정은 슬라이더 값으로 매번 다시 계산됨)
        X_new_esg = items[["협력사_환경위반건수", "협력사_노동평가점수"]]
        items["위험_확률"] = models["esg_model"].predict_proba(X_new_esg)[:, 1]

        st.session_state["predictions"] = items[
            ["상품영문명", "예측_HS코드", "위험_확률"]
        ].reset_index(drop=True)

st.subheader("예측 결과")

if "predictions" in st.session_state:
    result = st.session_state["predictions"].copy()
    result["판정"] = np.where(result["위험_확률"] >= threshold, "위험", "정상")

    for _, row in result.iterrows():
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""<div style="padding:22px;border-radius:12px;background:{NAVY};
                color:white;text-align:center;height:100%;">
                <div style="font-size:13px;opacity:0.75;">상품명</div>
                <div style="font-size:17px;font-weight:700;margin-top:6px;">{row['상품영문명']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div style="padding:22px;border-radius:12px;background:{TEAL};
                color:white;text-align:center;height:100%;">
                <div style="font-size:13px;opacity:0.85;">추천 HS Code</div>
                <div style="font-size:36px;font-weight:700;margin-top:6px;">{row['예측_HS코드']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c3:
            badge_color = RISK_COLOR if row["판정"] == "위험" else SAFE_COLOR
            st.markdown(
                f"""<div style="padding:22px;border-radius:12px;background:{badge_color};
                color:white;text-align:center;height:100%;">
                <div style="font-size:13px;opacity:0.9;">ESG 위험 확률 {row['위험_확률'] * 100:.1f}%</div>
                <div style="font-size:36px;font-weight:700;margin-top:6px;">{row['판정']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.write("")
else:
    st.info("위에서 신규 상품 정보를 입력한 뒤 '예측 실행' 버튼을 눌러주세요.")

st.divider()


# ------------------------------------------------------------------
# 5. 모델 성능 (Confusion Matrix, Precision · Recall)
# ------------------------------------------------------------------
st.header("4. 모델 성능 확인 (ESG 위험도 모델, 테스트셋 기준)")
st.caption(f"현재 임계값 {threshold:.2f} 기준으로 계산됩니다. 위 슬라이더를 움직이면 아래 표도 함께 바뀝니다.")

y_test_esg = models["y_test_esg"]
proba_test = models["esg_test_proba"]
y_pred_test = (proba_test >= threshold).astype(int)

cm = confusion_matrix(y_test_esg, y_pred_test, labels=[0, 1])
cm_df = pd.DataFrame(cm, index=["실제_정상", "실제_위험"], columns=["예측_정상", "예측_위험"])

precision = precision_score(y_test_esg, y_pred_test, zero_division=0)
recall = recall_score(y_test_esg, y_pred_test, zero_division=0)

perf_col1, perf_col2 = st.columns(2)
with perf_col1:
    st.markdown("**Confusion Matrix**")
    st.dataframe(cm_df, use_container_width=True)
with perf_col2:
    st.markdown("**Precision · Recall**")
    st.dataframe(
        pd.DataFrame({"지표": ["Precision", "Recall"], "값": [round(precision, 3), round(recall, 3)]}),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Precision: 위험이라고 예측한 것 중 실제로 위험인 비율 / "
    "Recall: 실제 위험 중 놓치지 않고 잡아낸 비율"
)
