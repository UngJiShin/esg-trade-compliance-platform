# -*- coding: utf-8 -*-
"""
========================================================================================
글로벌 무역 컴플라이언스 & ESG 통합 인텔리전스 플랫폼 (Global Trade Compliance & ESG Platform)
========================================================================================
- 교육과정 1일차~6일차 전체 실습 및 노코드/프롬프트/ML 파이프라인 통합 웹 애플리케이션
- Streamlit 기반 인터랙티브 UI / Render.com 원클릭 클라우드 배포 호환
"""

import os
import sys
import json
import re
import math
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1. 기본 설정 및 디자인
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="무역 컴플라이언스 & ESG 인텔리전스",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-risk {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warn {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 6px 6px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 경로 및 헬퍼 함수
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_data_path(filename):
    """우선 data/ 폴더에서 찾고, 없으면 전체 경로에서 탐색"""
    direct_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path
    # Fallback to local search
    for root_dir, _, files in os.walk(BASE_DIR):
        if filename in files:
            return os.path.join(root_dir, filename)
    return None

@st.cache_data
def load_csv(filename, encoding="utf-8-sig"):
    path = get_data_path(filename)
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            return pd.read_csv(path, encoding="cp949")
    return None

@st.cache_data
def load_excel(filename):
    path = get_data_path(filename)
    if path and os.path.exists(path):
        return pd.read_excel(path)
    return None

@st.cache_data
def load_knowledge():
    path = get_data_path("knowledge.json")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# -----------------------------------------------------------------------------
# 3. 모델 캐싱 (블록 A & B)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_prediction_models():
    df = load_csv("master_dataset.csv")
    if df is None:
        # Fallback dummy data if not found
        df = pd.DataFrame({
            "상품영문명": ["Hot Rolled Steel Sheet", "Aluminum Extruded Profile", "Urea Fertilizer"],
            "사양": ["Thickness 2.0mm Coil", "6063-T5 Alloy", "46% Nitrogen Granular"],
            "HS코드_정답": [7208.39, 7604.29, 3102.10],
            "협력사_환경위반건수": [0, 2, 5],
            "협력사_노동평가점수": [85, 60, 40],
            "리스크_정답": ["정상", "위험", "위험"]
        })
    
    # 1. HS Code 추천 모델 (블록 A)
    df["상품설명"] = df["상품영문명"].astype(str) + " " + df["사양"].astype(str)
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    X_text = tfidf.fit_transform(df["상품설명"])
    y_hs = df["HS코드_정답"].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_hs, test_size=0.25, random_state=42, stratify=y_hs if len(y_hs.unique()) > 1 else None
    )
    hs_model = RandomForestClassifier(n_estimators=100, random_state=42)
    hs_model.fit(X_train, y_train)
    hs_acc = accuracy_score(y_test, hs_model.predict(X_test))
    
    # 2. ESG 위험도 모델 (블록 B)
    X_esg = df[["협력사_환경위반건수", "협력사_노동평가점수"]]
    y_esg = (df["리스크_정답"].astype(str).str.strip() == "위험").astype(int)
    
    X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(
        X_esg, y_esg, test_size=0.25, random_state=42
    )
    esg_model = LogisticRegression(random_state=42)
    esg_model.fit(X_e_train, y_e_train)
    
    y_e_pred = esg_model.predict(X_e_test)
    esg_acc = accuracy_score(y_e_test, y_e_pred)
    esg_prec = precision_score(y_e_test, y_e_pred, zero_division=0)
    esg_rec = recall_score(y_e_test, y_e_pred, zero_division=0)
    cm = confusion_matrix(y_e_test, y_e_pred)
    
    return {
        "df_master": df,
        "tfidf": tfidf,
        "hs_model": hs_model,
        "hs_acc": hs_acc,
        "esg_model": esg_model,
        "esg_acc": esg_acc,
        "esg_prec": esg_prec,
        "esg_rec": esg_rec,
        "cm": cm
    }

WATCHLIST_HS_CODES = ["7208.10", "7208.39", "7604.21", "7604.29", "2804.10", "2716.00"]

# -----------------------------------------------------------------------------
# 4. 상단 네비게이션 및 헤더
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown('<div class="main-title">🌐 글로벌 무역 컴플라이언스 & ESG 통합 인텔리전스</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">EU CBAM · HS Code 자동추천 · 공급망 ESG 위험도 스크리닝 · 서류 교차검증 · RAG 규정 감사 플랫폼</div>', unsafe_allow_html=True)
with col_h2:
    st.caption("시스템 상태: 🟢 정상 가동 중")
    st.caption(f"기준 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 6개 핵심 탭 구성
tabs = st.tabs([
    "📊 종합 모니터링 대시보드",
    "🤖 HS Code & ESG 예측 에이전트",
    "🔍 서류 교차 검증 & 사유서 진단",
    "⚖️ CBAM 배출량 산정 & 규제 스코어링",
    "📚 RAG 규정 지식 검색 & 계약서 감사",
    "🚀 Render 배포 & 시스템 가이드"
])

# =============================================================================
# TAB 1: 종합 모니터링 대시보드 (Executive Dashboard & Alerts - 5·6일차)
# =============================================================================
with tabs[0]:
    st.subheader("📌 수출입 통관 & 공급망 ESG 종합 모니터링 대시보드")
    st.caption("5일차 실습 대시보드 연동 데이터, 국가별 ESG 리스크, 관세율 및 주간 리스크 추이를 한눈에 모니터링합니다.")
    
    # 1.1 데이터 로드
    df_dash = load_csv("dashboard_integrated.csv")
    df_country = load_csv("master_country_esg.csv")
    df_tariff = load_csv("master_tariff_cbam.csv")
    df_weekly = load_csv("weekly_risk_log.csv")
    df_manager = load_csv("manager_mapping.csv")
    
    if df_dash is None:
        st.error("대시보드 연동 데이터를 불러올 수 없습니다.")
    else:
        # 1.2 상단 KPI 카드
        total_cnt = len(df_dash)
        high_risk_cnt = len(df_dash[df_dash["예측_위험확률"] >= 0.5])
        cross_err_cnt = len(df_dash[df_dash["교차검증_위험신호수"] > 0])
        cbam_cnt = len(df_dash[df_dash["배출량_tCO2e"].notnull()])
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("총 점검 건수", f"{total_cnt} 건", delta="전체 선적 문서")
        kpi2.metric("고위험 감지 건수", f"{high_risk_cnt} 건", delta="위험확률 50% 이상", delta_color="inverse")
        kpi3.metric("교차검증 불일치", f"{cross_err_cnt} 건", delta="서류 오류 발생", delta_color="inverse")
        kpi4.metric("CBAM 대상 품목", f"{cbam_cnt} 건", delta="배출량 산정 대상")
        
        st.divider()
        
        # 1.3 차트 및 필터 레이아웃
        c_left, c_right = st.columns([3, 2])
        
        with c_left:
            st.markdown("##### 📈 주간 리스크 감지 추이 (최근 7일)")
            if df_weekly is not None and not df_weekly.empty:
                # 날짜별 / 리스크 유형별 피벗
                pivot_risk = df_weekly.pivot_table(
                    index="날짜", columns="리스크유형", values="문서번호", aggfunc="count", fill_value=0
                )
                st.bar_chart(pivot_risk, height=260)
            else:
                st.info("주간 리스크 로그 데이터가 없습니다.")
                
        with c_right:
            st.markdown("##### 🔍 대시보드 실시간 필터링")
            risk_types = ["전체"] + list(df_dash["리스크유형"].unique())
            selected_type = st.selectbox("리스크 유형 선택", risk_types)
            prob_filter = st.slider("최소 위험 확률 필터", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
            
            # 필터링 적용
            filtered_dash = df_dash.copy()
            if selected_type != "전체":
                filtered_dash = filtered_dash[filtered_dash["리스크유형"] == selected_type]
            filtered_dash = filtered_dash[filtered_dash["예측_위험확률"] >= prob_filter]
            st.caption(f"필터 결과: {len(filtered_dash)}건 (전체 {total_cnt}건 중)")
            
        # 1.4 통합 데이터 테이블
        st.markdown("##### 📋 통합 리스크 상세 현황")
        
        display_df = filtered_dash[[
            "문서번호", "세트ID", "품목명", "HS_Code", "원산지", "목적지", 
            "인보이스금액_USD", "순중량_kg", "배출량_tCO2e", "교차검증_내용", "예측_위험확률", "리스크유형"
        ]].copy()
        
        # 스타일링
        def highlight_dash(row):
            if row["예측_위험확률"] >= 0.7 or row["리스크유형"] in ["서류불일치", "ESG"]:
                return ["background-color: #FEE2E2; color: #991B1B; font-weight: 600;"] * len(row)
            elif row["예측_위험확률"] >= 0.4:
                return ["background-color: #FEF3C7; color: #92400E;"] * len(row)
            return ["background-color: #F0FDF4;"] * len(row)
            
        st.dataframe(display_df.style.apply(highlight_dash, axis=1), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 1.5 자동 알림 및 담당자 자동 매핑 발송 시뮬레이션
        st.markdown("##### 📢 리스크 감지 자동 알림 발송 시뮬레이터 (5일차 핵심 기능)")
        st.caption("위험 건을 선택하면 담당자 매핑표(김서류, 박원산, 이품목, 정환경)와 연동되어 공식 경고 메일 템플릿이 자동 작성됩니다.")
        
        col_sel, col_mail = st.columns([1, 2])
        
        with col_sel:
            selected_doc_id = st.selectbox("알림을 보낼 문서번호 선택", df_dash["문서번호"].tolist())
            selected_row = df_dash[df_dash["문서번호"] == selected_doc_id].iloc[0]
            
            # 담당자 매핑
            risk_t = selected_row["리스크유형"]
            assigned_manager = "미지정"
            assigned_email = "compliance@company.com"
            assigned_dept = "컴플라이언스팀"
            
            if df_manager is not None:
                m_match = df_manager[df_manager["리스크유형"] == risk_t]
                if not m_match.empty:
                    assigned_manager = m_match.iloc[0]["담당자"]
                    assigned_email = m_match.iloc[0]["이메일"]
                    assigned_dept = m_match.iloc[0]["부서"]
                    
            st.info(f"""
            **문서 정보:**
            - 세트ID: `{selected_row['세트ID']}`
            - 품목명: **{selected_row['품목명']}** (HS {selected_row['HS_Code']})
            - 리스크유형: **{risk_t}**
            - 예측 위험확률: **{selected_row['예측_위험확률']*100:.1f}%**
            
            **자동 배정 담당자:**
            - 담당자: **{assigned_manager}** ({assigned_dept})
            - 이메일: `{assigned_email}`
            """)
            
        with col_mail:
            # 템플릿 자동 포맷팅
            default_template = f"""Subject: [Risk Alert] {risk_t} detected - {selected_doc_id}

Dear {assigned_manager} ({assigned_dept}),

A risk item has been detected in the export compliance dashboard.

- Document No.: {selected_doc_id} (Set: {selected_row['세트ID']})
- Item: {selected_row['품목명']} (HS Code {selected_row['HS_Code']})
- Risk type: {risk_t}
- Risk probability: {selected_row['예측_위험확률']*100:.1f}%
- Cross-check detail: {selected_row['교차검증_내용']}
- Action Required: Please review the discrepancy and confirm corrective actions before shipment release.

Best regards,
Export Compliance Intelligence System (Automated Alert)
"""
            mail_content = st.text_area("생성된 알림 메일 초안 (수정 가능)", value=default_template, height=220)
            if st.button("🚀 담당자에게 경고 이메일 발송 시뮬레이션", key="send_alert"):
                st.success(f"✅ [{assigned_email}] 담당자에게 성공적으로 알림이 전달되었습니다! (로그 기록 완료: {datetime.now().strftime('%H:%M:%S')})")

        # 1.6 마스터 데이터 참조 섹션
        with st.expander("📂 국가별 ESG 리스크 & 품목별 관세율 마스터 데이터 참조"):
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                st.markdown("**마스터: 국가별 ESG 리스크 기준**")
                if df_country is not None:
                    st.dataframe(df_country, use_container_width=True, hide_index=True)
            with m_c2:
                st.markdown("**마스터: 품목별 관세율 및 CBAM 적용 여부**")
                if df_tariff is not None:
                    st.dataframe(df_tariff, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: HS Code & ESG 예측 에이전트 (3·4일차 블록 A, B, C)
# =============================================================================
with tabs[1]:
    st.subheader("🤖 AI 기반 HS Code 추천 & 공급망 ESG 위험도 통합 예측 에이전트")
    st.caption("3·4일차 학생 교안 블록 A(HS Code 모델), 블록 B(ESG 스크리닝), 블록 C(통합 에이전트)를 실시간 실행합니다.")
    
    # 모델 학습
    models = train_prediction_models()
    
    # 2.1 모델 성능 요약 배너
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("블록 A (HS코드) 정확도", f"{models['hs_acc']:.1%}")
    m2.metric("블록 B (ESG위험) 정확도", f"{models['esg_acc']:.1%}")
    m3.metric("블록 B 정밀도 (Precision)", f"{models['esg_prec']:.1%}")
    m4.metric("블록 B 재현율 (Recall)", f"{models['esg_rec']:.1%}")
    
    st.divider()
    
    # 2.2 블록 A & B 개별 테스트
    b_col1, b_col2 = st.columns(2)
    
    with b_col1:
        st.markdown("#### [블록 A] 신규 품목 HS Code 자동 추천")
        st.caption("영문 상품명과 규격 사양을 입력하면 TF-IDF와 Random Forest가 적합한 HS Code를 추천합니다.")
        
        sample_items = [
            ("Custom Input", ""),
            ("Aluminum Profile", "Alloy 6063-T5 Extruded"),
            ("Hot Rolled Steel", "Coil Thickness 3.0mm Width 1200mm"),
            ("Urea Fertilizer", "Prilled 46% Nitrogen"),
            ("Compressed Hydrogen", "High Pressure Gas Cylinder")
        ]
        sel_sample = st.selectbox("테스트 샘플 선택", [s[0] for s in sample_items])
        
        if sel_sample == "Custom Input":
            in_name = st.text_input("상품 영문명", "Aluminum Alloy Bar")
            in_spec = st.text_input("상품 사양", "Diameter 50mm Round 6061-T6")
        else:
            match = [s for s in sample_items if s[0] == sel_sample][0]
            in_name = st.text_input("상품 영문명", match[0])
            in_spec = st.text_input("상품 사양", match[1])
            
        if st.button("🎯 HS Code 예측 실행", key="btn_hs_predict"):
            combined = f"{in_name} {in_spec}"
            vec = models["tfidf"].transform([combined])
            pred_hs = models["hs_model"].predict(vec)[0]
            proba = models["hs_model"].predict_proba(vec).max()
            
            is_watchlist = "⚠️ 전략물자 / 워치리스트 대상" if str(pred_hs) in WATCHLIST_HS_CODES else "일반 품목"
            
            st.success(f"추천 HS Code: **{pred_hs}** (예측 확신도: {proba*100:.1f}%)")
            st.info(f"품목 분류 태그: **{is_watchlist}**")
            
    with b_col2:
        st.markdown("#### [블록 B] 협력사 ESG 위험도 스크리닝")
        st.caption("협력사의 환경 위반 건수와 노동평가 점수를 기반으로 부실/위험 협력사를 조기 탐지합니다.")
        
        in_env = st.slider("협력사 환경 위반 건수 (최근 3년)", 0, 10, 2)
        in_labor = st.slider("협력사 노동/인권 평가 점수 (100점 만점)", 0, 100, 58)
        
        threshold_b = st.slider("분류 위험 임계값 (Threshold)", 0.1, 0.9, 0.5, 0.05, 
                               help="이 값 이상이면 위험 협력사로 판정합니다.")
        
        prob_risk = models["esg_model"].predict_proba([[in_env, in_labor]])[0][1]
        is_risk = prob_risk >= threshold_b
        
        st.markdown(f"**산출된 위험 확률:** `{prob_risk*100:.1f}%` (기준 임계값: `{threshold_b*100:.0f}%`)")
        if is_risk:
            st.error("🚨 판정 결과: **[위험] 공급망 실사 및 시정조치계획 제출 대상**")
        else:
            st.success("✅ 판정 결과: **[정상] 적격 공급사 (지속 모니터링)**")
            
        # 혼동행렬 표시
        st.caption(f"테스트 데이터 Confusion Matrix: TN={models['cm'][0,0]}, FP={models['cm'][0,1]}, FN={models['cm'][1,0]}, TP={models['cm'][1,1]}")

    st.divider()

    # 2.3 블록 C: 노코드 통합 예측 에이전트 (일괄 서류 스크리닝)
    st.markdown("#### [블록 C] 노코드 통합 예측 에이전트 빌드 (일괄 스크리닝)")
    st.caption("문서번호 기준으로 HS추천값과 ESG위험도를 안전 병합하고, 오늘 검토가 필요한 고위험 건을 자동 상단 정렬합니다.")
    
    c_threshold = st.slider("리포트 필터 임계값 (오늘 검토할 위험 확률 기준)", 0.1, 0.9, 0.5, 0.05, key="c_thresh")
    
    # 기본 서류 로드
    df_shipping = load_csv("shipping_docs_sample.csv")
    if df_shipping is not None:
        if "선적ID" in df_shipping.columns and "문서번호" not in df_shipping.columns:
            df_shipping = df_shipping.rename(columns={"선적ID": "문서번호"})
            
        # 통합 에이전트 실행
        docs = df_shipping.copy()
        docs["상품설명"] = docs["상품영문명"].astype(str) + " " + docs["사양"].astype(str)
        
        vec_c = models["tfidf"].transform(docs["상품설명"])
        docs["예측_HS코드"] = models["hs_model"].predict(vec_c)
        
        X_esg_c = docs[["협력사_환경위반건수", "협력사_노동평가점수"]]
        docs["위험_확률"] = models["esg_model"].predict_proba(X_esg_c)[:, 1]
        docs["판정"] = docs["위험_확률"].apply(lambda p: "위험" if p >= c_threshold else "정상")
        docs["워치리스트_관찰대상"] = docs["예측_HS코드"].apply(lambda c: "예 (주의)" if str(c) in WATCHLIST_HS_CODES else "-")
        
        # 정렬
        sorted_docs = docs.sort_values("위험_확률", ascending=False).reset_index(drop=True)
        review_targets = sorted_docs[sorted_docs["판정"] == "위험"].reset_index(drop=True)
        
        r_col1, r_col2 = st.columns(2)
        r_col1.metric("오늘 검토가 필요한 건", f"{len(review_targets)} 건", delta="우선 조치 대상")
        r_col2.metric("전체 점검 서류 건수", f"{len(sorted_docs)} 건")
        
        st.markdown("**전체 선적 서류 스크리닝 결과 (위험도 내림차순 정렬)**")
        show_cols = ["문서번호", "상품영문명", "사양", "협력사_환경위반건수", "협력사_노동평가점수", "예측_HS코드", "위험_확률", "판정", "워치리스트_관찰대상"]
        
        disp_c = sorted_docs[show_cols].copy()
        disp_c["위험_확률"] = (disp_c["위험_확률"] * 100).round(1).astype(str) + "%"
        
        def highlight_c(row):
            if row["판정"] == "위험":
                return ["background-color: #FEE2E2; color: #991B1B; font-weight: 600;"] * len(row)
            return ["background-color: #F0FDF4;"] * len(row)
            
        st.dataframe(disp_c.style.apply(highlight_c, axis=1), use_container_width=True, hide_index=True)
        
        if not review_targets.empty:
            st.markdown("##### 📥 오늘의 검토 리포트 다운로드")
            csv_data = review_targets.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "오늘의 검토 리포트 CSV 다운로드",
                data=csv_data,
                file_name="오늘의_검토리포트.csv",
                mime="text/csv"
            )

# =============================================================================
# TAB 3: 서류 교차 검증 & 사유서 진단 (2일차)
# =============================================================================
with tabs[2]:
    st.subheader("🔍 비정형 서류 교차 검증 & 통관 거부 사유서 자동 진단")
    st.caption("2일차 실습: 상업송장(Invoice)과 수출신고필증을 라인 단위로 대조하여 단위 환산, 수량, 금액, HS Code, 원산지 불일치를 자동 적출합니다.")
    
    c_sub1, c_sub2 = st.tabs(["📑 인보이스 vs 수출신고필증 교차 검증", "🛑 통관 거부 사유서(Rejection Docs) 분석"])
    
    with c_sub1:
        st.markdown("#### 상업송장 vs 수출신고필증 자동 교차 검증")
        st.caption("단위 환산 규칙(1 ton = 1,000 kg), HS Code 일치 여부, 원산지 누락을 자동으로 검증합니다.")
        
        df_inv = load_excel("sample_invoice.xlsx")
        df_exp = load_excel("sample_export_declaration.xlsx")
        
        if df_inv is not None and df_exp is not None:
            # 병합 및 검증 로직
            merged_check = []
            for i in range(min(len(df_inv), len(df_exp))):
                inv_row = df_inv.iloc[i]
                exp_row = df_exp.iloc[i]
                
                doc_inv = str(inv_row.get("문서번호", f"INV-{i+1:03d}"))
                doc_exp = str(exp_row.get("신고번호", f"EXP-{i+1:03d}"))
                item_name = str(inv_row.get("품명", ""))
                
                # 수량 및 단위 검증
                qty_inv = float(inv_row.get("수량", 0))
                unit_inv = str(inv_row.get("단위", "kg")).strip().lower()
                qty_exp = float(exp_row.get("수량", 0))
                unit_exp = str(exp_row.get("단위", "kg")).strip().lower()
                
                # 단위 환산: ton -> kg
                norm_qty_inv = qty_inv * 1000 if unit_inv == "ton" else qty_inv
                norm_qty_exp = qty_exp * 1000 if unit_exp == "ton" else qty_exp
                
                qty_match = abs(norm_qty_inv - norm_qty_exp) < 1e-3
                is_unit_converted = (unit_inv != unit_exp) and qty_match
                
                # 금액 검증
                amt_inv = float(inv_row.get("금액(USD)", 0))
                amt_exp = float(exp_row.get("신고금액(USD)", 0))
                amt_match = abs(amt_inv - amt_exp) < 1.0
                
                # HS Code 검증
                hs_inv = str(inv_row.get("HS코드", "")).strip()
                hs_exp = str(exp_row.get("HS코드", "")).strip()
                hs_match = (hs_inv != "nan" and hs_exp != "nan" and hs_inv == hs_exp)
                
                # 원산지 검증
                orig_inv = str(inv_row.get("원산지", "")).strip()
                orig_exp = str(exp_row.get("원산지", "")).strip()
                orig_missing = (orig_inv in ["nan", ""] or orig_exp in ["nan", ""])
                orig_match = (not orig_missing) and (orig_inv == orig_exp)
                
                # 불일치 사유 정리
                discrepancies = []
                if not qty_match:
                    discrepancies.append(f"수량 불일치 (송장 {qty_inv}{unit_inv} vs 신고 {qty_exp}{unit_exp})")
                elif is_unit_converted:
                    discrepancies.append(f"단위 환산 적용됨 ({unit_exp} -> {unit_inv})")
                    
                if not amt_match:
                    discrepancies.append(f"금액 불일치 (송장 ${amt_inv:,.0f} vs 신고 ${amt_exp:,.0f})")
                if not hs_match:
                    discrepancies.append(f"HS Code 불일치 또는 누락 (송장 {hs_inv} vs 신고 {hs_exp})")
                if orig_missing:
                    discrepancies.append("원산지 표기 누락")
                elif not orig_match:
                    discrepancies.append(f"원산지 불일치 (송장 {orig_inv} vs 신고 {orig_exp})")
                    
                status = "정상" if (qty_match and amt_match and hs_match and orig_match) else "불일치 발생"
                
                merged_check.append({
                    "송장번호": doc_inv,
                    "신고번호": doc_exp,
                    "품명": item_name,
                    "송장수량": f"{qty_inv} {unit_inv}",
                    "신고수량": f"{qty_exp} {unit_exp}",
                    "송장HS": hs_inv,
                    "신고HS": hs_exp,
                    "송장원산지": orig_inv if orig_inv != "nan" else "❌누락",
                    "신고원산지": orig_exp if orig_exp != "nan" else "❌누락",
                    "판정상태": status,
                    "상세 진단 내용": "; ".join(discrepancies) if discrepancies else "전 항목 일치 (정상)"
                })
                
            res_df = pd.DataFrame(merged_check)
            
            # 통계 요약
            err_count = len(res_df[res_df["판정상태"] == "불일치 발생"])
            s1, s2, s3 = st.columns(3)
            s1.metric("총 검증 쌍", f"{len(res_df)} 세트")
            s2.metric("정상 세트", f"{len(res_df) - err_count} 세트")
            s3.metric("오류/불일치 적출", f"{err_count} 세트", delta_color="inverse")
            
            def highlight_check(row):
                if row["판정상태"] == "불일치 발생":
                    return ["background-color: #FEE2E2; color: #991B1B; font-weight: 500;"] * len(row)
                return ["background-color: #F0FDF4;"] * len(row)
                
            st.dataframe(res_df.style.apply(highlight_check, axis=1), use_container_width=True, hide_index=True)
        else:
            st.warning("인보이스 및 수출신고필증 샘플 파일을 찾을 수 없습니다.")

    with c_sub2:
        st.markdown("#### 통관 거부 사유서 (Rejection Notice) 자동 분석 & 카테고리 매핑")
        st.caption("비정형 사유서 텍스트에서 키워드를 추출하여 5대 컴플라이언스 리스크 카테고리로 자동 분류합니다.")
        
        df_reason = load_csv("reason_docs.csv")
        df_cat_ref = load_csv("risk_category_ref.csv")
        
        case_options = ["직접 텍스트 입력"]
        if df_reason is not None:
            case_options = df_reason["문서번호"].tolist() + ["직접 텍스트 입력"]
            
        sel_case = st.selectbox("분석할 통관 거부 사유서 선택", case_options)
        
        if sel_case != "직접 텍스트 입력" and df_reason is not None:
            sample_text = df_reason[df_reason["문서번호"] == sel_case].iloc[0]["사유서원문텍스트"]
        else:
            sample_text = "통관 거부 사유서\n\n문서번호: CD-2025-9999\n발급기관: 세관 검사과\n거부 사유: 수입신고서상 품목분류 코드가 7616.99로 신고되었으나 현품 확인 결과 7604.29로 판정되어 HS Code 불일치로 통관이 보류됩니다."
            
        input_reason = st.text_area("사유서 원문 텍스트", value=sample_text, height=180)
        
        if st.button("🔎 사유서 정밀 진단 실행", key="btn_diag_reason"):
            # 키워드 매칭
            detected_cats = set()
            detected_kws = []
            
            if df_cat_ref is not None:
                for _, r in df_cat_ref.iterrows():
                    kw = str(r["키워드"]).strip()
                    cat = str(r["리스크카테고리"]).strip()
                    if kw in input_reason:
                        detected_cats.add(cat)
                        detected_kws.append(f"{kw} ➔ [{cat}]")
                        
            # 보완 fallback
            if not detected_cats:
                if "카드뮴" in input_reason or "허용기준" in input_reason or "중금속" in input_reason:
                    detected_cats.add("환경/유해물질 기준 초과")
                elif "UN번호" in input_reason or "라벨" in input_reason or "GHS" in input_reason:
                    detected_cats.add("라벨링/표시사항 부적합")
                elif "전략물자" in input_reason or "수출허가" in input_reason:
                    detected_cats.add("수출 제한 품목")
                elif "원산지" in input_reason:
                    detected_cats.add("원산지 증빙 누락")
                elif "HS" in input_reason or "품목분류" in input_reason:
                    detected_cats.add("HS Code 불일치")
                else:
                    detected_cats.add("기타 규정 위반")
                    
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("##### 📌 진단 결과 요약")
                for c in detected_cats:
                    st.error(f"🚨 적출된 리스크 카테고리: **{c}**")
                st.markdown("**검출된 핵심 키워드:**")
                st.write(", ".join(detected_kws) if detected_kws else "특정 매핑 키워드 없음")
                
            with r_col2:
                st.markdown("##### 💡 필수 대응 조치 가이드")
                if "HS Code 불일치" in detected_cats:
                    st.info("• 관세청 사전심사제도 활용 및 정정신고서 제출\n• 송장 및 신고서 세번 일치 검증 강화")
                if "원산지 증빙 누락" in detected_cats:
                    st.info("• 원산지증명서(C/O) 재발급 및 세번변경/부가가치 기준 소명서 준비")
                if "환경/유해물질 기준 초과" in detected_cats:
                    st.info("• ISO/IEC 17025 공인시험기관 성분분석 성적서 첨부\n• 원료 배치별 중금속 허용치 재검증")
                if "라벨링/표시사항 부적합" in detected_cats:
                    st.info("• UN 규격 라벨 재인쇄 및 GHS 픽토그램 크기 규격(10cm x 10cm) 확인\n• 수입국 공용어 안전문구 동봉")
                if "수출 제한 품목" in detected_cats:
                    st.info("• 산업통상자원부 전략물자관리원 사전판정서 발급\n• 최종수요자(End-User) 확인서 확보")

# =============================================================================
# TAB 4: CBAM 배출량 산정 & 규제 스코어링 (1일차 & Round 5)
# =============================================================================
with tabs[3]:
    st.subheader("⚖️ EU CBAM 내재배출량 산정 & 노코드 리스크 스코어링")
    st.caption("1일차 CBAM 가이드라인 산정 공식과 라운드5 노코드 에이전트의 가중합(Bag of Words + Feature Construction) 스코어링을 실시간 실행합니다.")
    
    cbam_tab1, cbam_tab2 = st.tabs(["🏭 CBAM 내재배출량 산정 계산기", "🏷️ 노코드 에이전트 리스크 스코어러 (Round 5)"])
    
    with cbam_tab1:
        st.markdown("#### EU 탄소국경조정제도(CBAM) 품목별 내재배출량 계산기")
        st.caption("Scope 1(직접 연료 연소)과 Scope 2(간접 전력 소비)를 합산해 제품 1톤당 내재배출량과 예상 인증서 비용을 산출합니다.")
        
        c_prod, c_vol, c_country = st.columns(3)
        with c_prod:
            prod_type = st.selectbox("대상 품목군", ["철강 (Steel)", "알루미늄 (Aluminum)", "시멘트 (Cement)", "비료 (Fertilizer)", "수소 (Hydrogen)"])
        with c_vol:
            prod_tons = st.number_input("생산/수출 물량 (톤, Ton)", min_value=1.0, value=200.0, step=10.0)
        with c_country:
            grid_factor = st.selectbox("생산국 전력망 배출계수 (tCO2e/MWh)", [
                ("대한민국 (KR) - 0.4781", 0.4781),
                ("독일 (DE) - 0.3850", 0.3850),
                ("중국 (CN) - 0.6100", 0.6100),
                ("인도 (IN) - 0.7200", 0.7200),
                ("EU 평균 (EU) - 0.2500", 0.2500)
            ], index=0)[1]
            
        c_s1, c_s2, c_cost = st.columns(3)
        with c_s1:
            fuel_direct = st.number_input("Scope 1 직접 배출량 (연료/공정, tCO2e)", min_value=0.0, value=120.0, step=5.0)
        with c_s2:
            mwh_used = st.number_input("생산 투입 전력량 (MWh)", min_value=0.0, value=350.0, step=10.0)
        with c_cost:
            ets_price = st.number_input("EU ETS 탄소 인증서 예상 가격 (€/tCO2e)", min_value=10.0, value=68.0, step=1.0)
            
        # 배출량 산출
        scope2_indirect = mwh_used * grid_factor
        total_emissions = fuel_direct + scope2_indirect
        intensity = total_emissions / prod_tons
        est_cost_eur = total_emissions * ets_price
        
        st.markdown("##### 📊 산정 결과 요약")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("총 내재배출량", f"{total_emissions:,.2f} tCO2e")
        e2.metric("톤당 배출집약도", f"{intensity:.3f} tCO2e / t")
        e3.metric("간접배출(Scope 2) 비중", f"{(scope2_indirect/total_emissions)*100:.1f}%")
        e4.metric("예상 CBAM 인증서 비용", f"€ {est_cost_eur:,.0f}")
        
        # 시각화
        chart_df = pd.DataFrame({
            "배출 유형": ["Scope 1 (직접연료)", "Scope 2 (간접전력)"],
            "배출량(tCO2e)": [fuel_direct, scope2_indirect]
        }).set_index("배출 유형")
        st.bar_chart(chart_df, height=200)

    with cbam_tab2:
        st.markdown("#### 라운드 5 노코드 에이전트: 리스크 키워드 가중합 스코어링")
        st.caption("PDF 교안의 Orange3 Bag of Words + Feature Constructor 가중합 수식을 파이썬 알고리즘으로 완벽 구현했습니다.")
        
        df_kws = load_csv("risk_keywords.csv")
        df_thresh = load_csv("risk_threshold.csv")
        df_sample_docs = load_csv("sample_docs.csv")
        
        # 기본 텍스트 선택
        sample_doc_list = ["사용자 직접 입력"]
        if df_sample_docs is not None:
            sample_doc_list = [f"샘플 문서 {i+1}" for i in range(len(df_sample_docs))] + ["사용자 직접 입력"]
            
        sel_doc_idx = st.selectbox("테스트용 문서 텍스트 선택", sample_doc_list)
        
        if sel_doc_idx != "사용자 직접 입력" and df_sample_docs is not None:
            idx = int(sel_doc_idx.replace("샘플 문서 ", "")) - 1
            default_text = df_sample_docs.iloc[idx]["텍스트"]
        else:
            default_text = "신고 물품의 내재배출량 산정 시 실측값 없음으로 인하여 기본값 적용되었으며, 검증 실패로 인해 과징금 부과 및 소급 추징 대상이 될 수 있음을 경고합니다. 서류 불일치 항목에 대한 소명 요구를 발송합니다."
            
        doc_text = st.text_area("분석 대상 텍스트", value=default_text, height=130)
        
        if st.button("⚡ 리스크 스코어 계산", key="btn_calc_score"):
            score = 0
            detected = []
            
            if df_kws is not None:
                for _, row in df_kws.iterrows():
                    kw = str(row["키워드"]).strip()
                    wt = int(row["가중치"])
                    count = len(re.findall(re.escape(kw), doc_text))
                    if count > 0:
                        kw_score = count * wt
                        score += kw_score
                        detected.append({"키워드": kw, "등장횟수": count, "가중치": wt, "기여점수": kw_score})
                        
            # 등급 판정
            if score <= 5:
                level = "초록 (안전 / 경미)"
                badge_class = "badge-safe"
                action_text = "정상 처리 가능. 통관 절차를 속행합니다."
            elif score <= 15:
                level = "노랑 (주의 / 보완 필요)"
                badge_class = "badge-warn"
                action_text = "주의 대상. 담당자 서류 재검토 및 소명 자료 사전 준비 권고."
            else:
                level = "빨강 (위험 / 심각 경고)"
                badge_class = "badge-risk"
                action_text = "고위험 경보! 즉시 출하 보류 및 법무/컴플라이언스팀 보고 필수."
                
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>최종 리스크 스코어</h4>
                    <h1 style="color: #1E3A8A;">{score} 점</h1>
                    <span class="{badge_class}">{level}</span>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"**대응 조치:** {action_text}")
                
            with sc2:
                st.markdown("##### 📋 검출된 리스크 키워드 기여도")
                if detected:
                    det_df = pd.DataFrame(detected).sort_values("기여점수", ascending=False)
                    st.dataframe(det_df, use_container_width=True, hide_index=True)
                else:
                    st.info("검출된 리스크 키워드가 없습니다. (점수: 0점)")

# =============================================================================
# TAB 5: RAG 규정 지식 검색 & 계약서 감사 (5·6일차)
# =============================================================================
with tabs[4]:
    st.subheader("📚 RAG 규정 지식 검색 엔진 & 실전 무역 서류 실시간 감사")
    st.caption("5·6일차 실습: 통관 규정집, CBAM 가이드라인 기반 지능형 질의응답(RAG)과 신규 수출계약서/바이어 문의 메일의 독소조항을 자동 감사합니다.")
    
    rag_sub1, rag_sub2, rag_sub3 = st.tabs(["📖 규정집 RAG 지식 검색", "📑 신규 수출계약서 자동 감사", "✉️ 바이어 문의 메일 자동 분석"])
    
    knowledge_base = load_knowledge()
    
    # 5.1 RAG 지식 검색
    with rag_sub1:
        st.markdown("#### 수출입 통관 & CBAM 이행 규정집 지능형 검색 (RAG)")
        st.caption("10대 표준 검증 질의(Q01~Q10)를 선택하거나 직접 질문을 입력하면 정확한 근거 조항을 발췌합니다.")
        
        df_q = load_csv("rag_questions.csv")
        q_options = ["직접 질문 입력"]
        if df_q is not None:
            q_options = [f"[{r['질의번호']}] {r['질의']}" for _, r in df_q.iterrows()] + ["직접 질문 입력"]
            
        selected_q = st.selectbox("질문 선택 또는 직접 입력", q_options)
        
        if selected_q != "직접 질문 입력" and df_q is not None:
            q_code = selected_q.split("] ")[0].replace("[", "")
            match_row = df_q[df_q["질의번호"] == q_code].iloc[0]
            current_query = match_row["질의"]
            expected_clause = str(match_row["정답_조항번호"])
        else:
            current_query = "철강 제품의 CBAM 적용 여부와 근거 조항을 알려줘"
            expected_clause = "14"
            
        user_query = st.text_input("질의 내용", value=current_query)
        
        if st.button("🔎 RAG 지식 검색 실행", key="btn_rag_search"):
            # 간단 TF-IDF 코사인 유사도 검색
            corpus = []
            meta = []
            for doc in knowledge_base:
                content = doc.get("content", "")
                # 조항 단위 분할
                clauses = re.split(r'(제\s*\d+\s*조)', content)
                if len(clauses) > 1:
                    for i in range(1, len(clauses), 2):
                        c_title = clauses[i]
                        c_body = clauses[i+1] if i+1 < len(clauses) else ""
                        full_clause = c_title + c_body
                        corpus.append(full_clause)
                        meta.append({"title": doc.get("title", ""), "clause": c_title, "source": doc.get("source", "")})
                else:
                    corpus.append(content[:2000])
                    meta.append({"title": doc.get("title", ""), "clause": "전체", "source": doc.get("source", "")})
                    
            if corpus:
                vec_rag = TfidfVectorizer().fit(corpus + [user_query])
                corpus_v = vec_rag.transform(corpus)
                q_v = vec_rag.transform([user_query])
                sims = cosine_similarity(q_v, corpus_v)[0]
                
                top_indices = np.argsort(sims)[::-1][:3]
                
                st.markdown("##### 💡 RAG 검색 결과 (상위 매칭 조항)")
                for rank, idx in enumerate(top_indices):
                    sim_score = sims[idx]
                    clause_info = meta[idx]
                    clause_text = corpus[idx].strip()
                    
                    with st.expander(f"Top {rank+1}: [{clause_info['source']}] {clause_info['clause']} (유사도: {sim_score*100:.1f}%)", expanded=(rank==0)):
                        st.markdown(f"**출처 문서:** `{clause_info['title']}`")
                        st.markdown(f"**발췌 본문:**\n```text\n{clause_text[:800]}...\n```")
                        if expected_clause != "None" and str(expected_clause) in clause_info['clause']:
                            st.success(f"✅ 정답 조항(제{expected_clause}조)과 완벽 일치합니다!")
            else:
                st.warning("규정 지식 베이스가 비어 있습니다.")

    # 5.2 신규 수출계약서 자동 감사
    with rag_sub2:
        st.markdown("#### 신규 수출 물품 매매계약서 (Nordic Metals) 자동 감사")
        st.caption("5일차 실습 과제: 계약서 발췌본의 규정 위반 독소조항(배출량 정보 거부, 원산지 미정 등)을 실시간 적출합니다.")
        
        default_contract = """수출 물품 매매계약서 (발췌본)
매도인: 한빛금속 주식회사 (대한민국)
매수인: Nordic Metals GmbH (독일)
품목: Aluminum Extruded Profile 6063-T5 (HS Code: 7604.29) / 수량: 200 BUNDLES / USD 185,000

제1조 (계약 물품) 원산지는 대한민국 또는 중국 중 선적 시점에 매도인이 정한다.
제2조 (인도 조건) 인도 조건은 FOB Incheon으로 한다.
제3조 (선적 서류) 상업송장과 패킹리스트의 수량, 금액, 중량 차이가 5퍼센트 이내인 경우에는 서류 불일치로 보지 않는다.
제4조 (배출량 정보) 매도인은 본 계약 물품의 내재 배출량 정보를 매수인에게 제공할 의무를 지지 않는다.
제5조 (서류 정정) 서류 불일치가 발견되면 매도인은 수출신고 수리일부터 90일 이내에 정정할 수 있다."""
        
        contract_text = st.text_area("계약서 원문", value=default_contract, height=180)
        
        if st.button("📑 계약서 컴플라이언스 감사 실행", key="btn_audit_contract"):
            st.markdown("##### 🚨 계약서 위험 조항 감사 리포트")
            
            # Rule 1: 제4조 배출량
            if "배출량 정보" in contract_text and "의무를 지지 않는다" in contract_text:
                st.error("""
                **[치명적 위험] 제4조 (배출량 정보 미제공 조항):**
                - **관련 규정:** EU CBAM 이행규정 제15조 및 통관규정 제24조
                - **진단:** EU 수입자(Nordic Metals)는 CBAM 분기별 보고서에 제품 실측 배출량을 신고해야 할 법적 의무가 있습니다. 매도인이 배출량 제공을 거부할 경우, EU 세관 통관 거부 및 과징금 분쟁이 발생합니다.
                - **수정 권고:** '매도인은 CBAM 표준 산정 가이드라인에 부합하는 실측 내재배출량 확인서를 선적 후 15일 이내에 제공한다'로 수정 필수.
                """)
                
            # Rule 2: 제1조 원산지
            if "선적 시점에 매도인이 정한다" in contract_text or "중국" in contract_text:
                st.warning("""
                **[주의/고위험] 제1조 (원산지 미확정 조항):**
                - **관련 규정:** 통관규정 제23조 및 FTA 원산지 규정
                - **진단:** 한-EU FTA 특혜관세(0%) 적용을 위해서는 선적 전 한국산 원산지 판정이 확정되어야 합니다. 중국산 혼용 시 일반관세(6.0%)가 부과되며 원산지증명서 불일치로 통관이 지연됩니다.
                - **수정 권고:** 원산지를 '대한민국(Republic of Korea)'으로 특정하고 한-EU FTA 원산지증명서 발급 조건을 명시할 것.
                """)
                
            # Rule 3: 제3조 오차
            if "5퍼센트" in contract_text or "5%" in contract_text:
                st.info("""
                **[양호] 제3조 (5% 허용 오차 조항):**
                - **관련 규정:** 통관규정집 제10조
                - **진단:** 순중량 및 수량 차이 5% 이내 허용 조항은 규정집 상의 표준 허용공차 범위에 부합합니다.
                """)

    # 5.3 바이어 문의 메일 분석
    with rag_sub3:
        st.markdown("#### 바이어 문의 메일 분석 및 공식 답신 초안 자동 생성")
        st.caption("독일 바이어(Nordic Metals Anna Weber)의 서명 전 3대 확인 요청에 대해 규정 근거를 포함한 공식 영문 답신을 작성합니다.")
        
        default_buyer_mail = """From: Anna Weber <anna.weber@nordicmetals.example>
To: export@hanbit-metal.example
Subject: Questions before signing - aluminium profile contract (200 bundles)

Dear Sales Team,
Before we sign, our team has three questions:
1) Origin: Could you confirm that we may receive either Korean or Chinese origin material, depending on availability at shipment?
2) Tolerance: Our finance team prefers a 3% difference between invoice and packing list without treating it as discrepancy. Is this acceptable?
3) Shipping terms: Please confirm FOB Incheon and whether CBAM emissions data will be included."""

        st.text_area("수신된 바이어 메일 원문", value=default_buyer_mail, height=140)
        
        if st.button("✉️ AI 법무/무역 답신 이메일 생성", key="btn_reply_mail"):
            reply_draft = """Subject: Re: Questions before signing - aluminium profile contract (200 bundles)

Dear Ms. Anna Weber,

Thank you for your inquiry. Regarding your three questions before contract signing, please find our official positions below:

1. Origin Clause (Origin Determination):
In accordance with EU Customs Regulation and EU-Korea FTA compliance, the origin cannot be left undecided at shipment. To guarantee preferential tariff treatment (0%) and compliant CBAM documentation, the origin will be strictly fixed as "Republic of Korea (KR)".

2. Document Tolerance (3% Discrepancy):
Yes, a 3% tolerance between the commercial invoice and packing list is fully acceptable under our Customs Guidelines Article 10 (which permits tolerances within 5%).

3. CBAM Embedded Emissions Data:
We confirm FOB Incheon terms. In addition, Section 4 will be revised so that Hanbit Metal will provide fully verified Scope 1 and Scope 2 embedded emissions reports (tCO2e/ton) prior to customs clearance.

We have attached the updated contract amendment for your countersignature.

Sincerely,
Export Compliance Team
Hanbit Metal Co., Ltd.
"""
            st.markdown("##### 📝 생성된 공식 답신 이메일 (영문)")
            st.code(reply_draft, language="markdown")
            st.success("✅ 바이어 규정 리스크 해소 및 협정관세 보호 답신이 작성되었습니다.")

# =============================================================================
# TAB 6: Render 배포 & 시스템 가이드
# =============================================================================
with tabs[5]:
    st.subheader("🚀 웹 Render.com 배포 가이드 & 전체 교육과정 아키텍처")
    st.caption("본 웹 애플리케이션은 클라우드 PaaS(Render.com)에 원클릭으로 무료 배포할 수 있도록 완벽히 패키징되어 있습니다.")
    
    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        st.markdown("""
        #### 🌐 Render.com 원클릭 무료 배포 절차
        1. **GitHub 저장소 푸시:**
           - 현재 폴더의 모든 파일(`app.py`, `requirements.txt`, `Procfile`, `render.yaml`, `data/`)을 GitHub 레포지토리에 푸시합니다.
        2. **Render.com 접속 및 새 웹 서비스 생성:**
           - [dashboard.render.com](https://dashboard.render.com) 로그인
           - **New +** ➔ **Web Service** 클릭 ➔ GitHub 저장소 연결
        3. **배포 설정값 입력:**
           - **Name:** `esg-trade-compliance-platform`
           - **Environment:** `Python 3`
           - **Build Command:** `pip install -r requirements.txt`
           - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
           - **Plan:** `Free`
        4. **Deploy Web Service 클릭:**
           - 빌드 완료 후 제공되는 공개 URL(예: `https://esg-trade-compliance.onrender.com`)로 전 세계 어디서나 접속 가능!
        """)
        
    with col_rep2:
        st.markdown("""
        #### 💻 로컬 PC 실행 방법 (Windows / Mac / Linux)
        ```bash
        # 1. 필수 라이브러리 설치
        pip install -r requirements.txt

        # 2. Streamlit 웹 앱 실행
        streamlit run app.py
        ```
        
        #### 📚 교육과정 대응 모듈 맵
        - **1일차:** CBAM 품목별 가이드라인, 노코드 리스크 스코어링 (탭 4)
        - **2일차:** 서류 클리닝, 인보이스-신고필증 교차검증, 거부사유서 진단 (탭 3)
        - **3·4일차:** 블록 A(HS Code), 블록 B(ESG 스크리닝), 블록 C(통합 에이전트) (탭 2)
        - **5일차:** 종합 모니터링 대시보드, 자동 알림 발송 시뮬레이터 (탭 1)
        - **6일차:** 통관규정 RAG 지식 검색, 신규 계약서/바이어 메일 감사 (탭 5)
        """)
        
    st.info("💡 배포 파일 체크: `requirements.txt`, `Procfile`, `render.yaml` 및 `data/` 디렉토리가 모두 포함되어 클라우드 환경에서 의존성 없이 즉시 작동합니다.")
