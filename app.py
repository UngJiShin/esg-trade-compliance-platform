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
import base64
import io
import zipfile
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

# -----------------------------------------------------------------------------
# 2. 데이터 경로 및 헬퍼 함수
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(BASE_DIR, "img")

@st.cache_data
def get_image_base64(filename):
    """최적화된 이미지를 우선 로드하고 base64로 반환"""
    opt_name = filename.replace(".jpg", "-opt.jpg")
    p_opt = os.path.join(IMG_DIR, opt_name)
    p_orig = os.path.join(IMG_DIR, filename)
    target = p_opt if os.path.exists(p_opt) else p_orig
    if os.path.exists(target):
        with open(target, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

# 이미지 Base64 프리로드
b64_ship = get_image_base64("hero-ship.jpg")
b64_globe = get_image_base64("dream-data-globe.jpg")
b64_harbor = get_image_base64("ceo-vision-harbor.jpg")

# Custom CSS
st.markdown("""
<style>
    /* 전체 폰트 및 모던 스타일링 */
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
    }
    
    /* Hero Banner */
    .hero-banner {
        position: relative;
        border-radius: 16px;
        padding: 36px 32px;
        margin-bottom: 24px;
        color: #FFFFFF;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 16px 36px -8px rgba(15, 23, 42, 0.35);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #E2E8F0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 12px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.25;
        margin: 0 0 10px 0;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #E2E8F0;
        margin: 0 0 20px 0;
        max-width: 850px;
        line-height: 1.55;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.5);
    }
    .hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .hero-chip {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(8px);
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #F1F5F9;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Feature Banner (Tabs) */
    .feature-banner {
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 22px;
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.15);
    }
    .feature-tag {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .feature-heading {
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.01em;
    }
    .feature-desc {
        font-size: 0.92rem;
        color: #E2E8F0;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -3px rgba(0, 0, 0, 0.08);
    }

    /* Badges */
    .badge-risk {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid #FECACA;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid #BBF7D0;
    }
    .badge-warn {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid #FDE68A;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 8px 8px 0px 0px;
        color: #475569;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1E3A8A;
        background-color: #F1F5F9;
    }
    .stTabs [aria-selected="true"] {
        color: #1E3A8A !important;
        border-bottom: 3px solid #2563EB !important;
        background-color: #EFF6FF !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 경로 및 헬퍼 함수
# -----------------------------------------------------------------------------
def get_data_path(filename):
    """우선 data/ 폴더에서 찾고, 없으면 전체 경로에서 탐색"""
    direct_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path
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
# 2.1 블록 C 템플릿 및 통합 에이전트 엔진
# -----------------------------------------------------------------------------
REPORT_TEMPLATE_BLOCKC = """당신은 EU CBAM 대응 ESG 실사 보고서를 작성하는 무역 컴플라이언스 실무 보조자입니다.
아래 [입력 자료]를 [표준 서식 항목]에 맞춰 정리해 보고서 초안을 작성해 주세요.

[표준 서식 항목]
1. 보고 개요 (세트 ID, 인보이스 번호, 품목군)
2. 수출 건 정보 (수출자·수입자, 품목·HS Code, 수량·중량·금액)
3. 내재 배출량 산정 (적용 방식, 입력값, 산식, 결과 tCO2e)
4. 서류 교차 검증 결과 (비교 항목별 판정, 불일치 목록)
5. 리스크 요약
6. 권고 조치
7. 근거 및 한계 (참조 문서, 산정 가정, 확인 필요 사항)

[작성 규칙]
- 7개 항목의 번호와 제목을 그대로 사용하고 순서를 바꾸지 않는다.
- [입력 자료]에 있는 사실만 사용한다. 없는 값은 추정하지 말고 '해당 자료 없음'이라고 적는다.
- 배출량과 합계 숫자는 입력 자료 그대로 옮긴다. 다시 계산해서 바꾸지 않는다.
- (3) 교차 검증 결과는 규칙이 '값이 다르면 모두' 표시한 1차 후보이다. 약어·표기·단위 환산·국가 표기 차이로 설명되는 것은 오류가 아닌 '표기 차이'로 분류하여 4번의 불일치 목록에서 빼고 '참고'로 따로 적는다. 합계 차이가 허용 오차 이내이면 오류로 쓰지 않는다.
- 판정이 애매하면 '확인 요망'으로 적고 이유를 한 줄 덧붙인다.
- 내용이 없는 항목도 삭제하지 말고 '해당 없음'과 그 이유를 적는다.

[입력 자료]
(1) 서류 정보: {doc_info}
(2) 배출량 산정 결과: {emission_info}
(3) 교차 검증 결과(규칙 1차): {check_info}
(4) 내가 정한 가정: {assumption_info}
"""

DIAG_TEMPLATE_BLOCKC = """당신은 수출 규제 리스크를 진단하는 무역 컴플라이언스 실무 보조자입니다.
[입력 자료]와 첨부한 문서(CBAM 가이드라인, 배출량 산정 규정표)를 바탕으로 이 수출 건의 '수출 규제 리스크 진단서'를 작성해 주세요.

[진단서 구성]
1. 진단 대상 (세트 ID, 품목, 수출자 → 수입자)
2. 종합 리스크 등급 (상 / 중 / 하 / 이상 없음 중 하나) 및 판단 이유 한 줄
3. 발견된 오류 (오류마다: 위치, 인보이스 값, 패킹리스트 값, 오류 유형)
4. 리스크 상세 (오류마다 통관·CBAM 신고·FTA 원산지 관점에서 생길 수 있는 문제)
5. 근거 조항 (첨부 문서에 실제로 있는 조항 번호와 요지)
6. 권고 조치 (오류마다: 누가, 무엇을 정정하거나 확인해야 하는지)
7. 배출량 산정 결과의 신뢰도 (확정값 / 잠정값과 그 이유)

[등급 기준 - 예시]
- 상: 품목이나 원산지가 서로 다른 등 신고 정정과 재확인이 필요한 오류
- 중: 수량·금액·중량 불일치로 서류 정정이 필요한 오류
- 하: 허용 오차 이내 차이나 표기 차이만 있는 경우
- 이상 없음: 실제 오류가 없는 경우

[작성 규칙]
- (3) 교차 검증 결과에 적힌 후보만 검토 대상으로 삼고, 오류를 새로 만들지 않는다.
- (3)은 규칙이 '값이 다르면 모두' 표시한 1차 후보이다. 약어·표기·단위 환산·국가 표기 차이로 설명되는 것은 오류로 쓰지 않고 '표기 차이(오류 아님)'로 따로 적는다. 실제 오류가 하나도 없으면 '발견된 오류 없음', 종합 등급은 '이상 없음'으로 한다.
- 근거 조항은 첨부 문서에서 실제로 찾을 수 있는 조항 번호와 문구만 인용한다. 찾을 수 없으면 조항 번호를 만들지 말고 '근거 조항 확인 필요'라고 적는다.
- 배출량 숫자는 입력 자료 그대로 쓰고 다시 계산하지 않는다.

[입력 자료]
(1) 서류 정보: {doc_info}
(2) 배출량 산정 결과: {emission_info}
(3) 교차 검증 결과(규칙 1차): {check_info}
(4) 내가 정한 가정: {assumption_info}
"""

def parse_orange_csv(file_obj_or_path, skip_orange=True):
    try:
        df_raw = pd.read_csv(file_obj_or_path, encoding='utf-8')
    except Exception:
        if hasattr(file_obj_or_path, 'seek'):
            file_obj_or_path.seek(0)
        df_raw = pd.read_csv(file_obj_or_path, encoding='cp949')
        
    if skip_orange and len(df_raw) > 2 and str(df_raw.iloc[0, 0]).strip().lower() in ['d', 'c', 's', 'm']:
        return df_raw.iloc[2:].copy().reset_index(drop=True)
    return df_raw

def run_block_c_pipeline(df_lines, df_coef, tol=0.005, convert_mt=True, korea_aliases=None, base_weight="인보이스 순중량"):
    if korea_aliases is None:
        korea_aliases = ["REPUBLIC OF KOREA", "KOREA", "KR", "ROK"]
        
    df = df_lines.copy()
    coef_df = df_coef.copy()
    
    num_cols = ['line_no', 'inv_qty', 'pl_qty', 'inv_amount', 'pl_amount', 'inv_net_kg', 'pl_net', 'energy_gj', 'power_mwh']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    coef_num_cols = ['gj_coef', 'default_t', 'direct_t', 'power_mwh_coef']
    for c in coef_num_cols:
        if c in coef_df.columns:
            coef_df[c] = pd.to_numeric(coef_df[c], errors='coerce').fillna(0)

    if convert_mt and 'pl_net_unit' in df.columns:
        df['pl_net_kg'] = np.where(df['pl_net_unit'].astype(str).str.upper() == 'MT', df['pl_net'] * 1000.0, df['pl_net'])
    else:
        df['pl_net_kg'] = df['pl_net'] if 'pl_net' in df.columns else df.get('inv_net_kg', 0)

    korea_upper = [a.upper().strip() for a in korea_aliases]
    df['inv_origin_n'] = df['inv_origin'].astype(str).str.strip().apply(
        lambda x: "KOREA" if x.upper() in korea_upper else x
    )
    df['pl_origin_n'] = df['pl_origin'].astype(str).str.strip().apply(
        lambda x: "KOREA" if x.upper() in korea_upper else x
    )

    df['f_qty'] = np.where(df['inv_qty'] != df['pl_qty'], 1, 0)
    df['f_amt'] = np.where(df['inv_amount'] != df['pl_amount'], 1, 0)
    df['f_net'] = np.where(df['inv_net_kg'] != df['pl_net_kg'], 1, 0)
    df['f_origin'] = np.where(df['inv_origin_n'] != df['pl_origin_n'], 1, 0)
    df['f_desc'] = np.where(df['inv_desc'].astype(str).str.strip() != df['pl_desc'].astype(str).str.strip(), 1, 0)

    def make_issue_text(row):
        issues = []
        if row['f_qty']:
            issues.append(f"수량 {int(row['inv_qty'])}→{int(row['pl_qty'])} (인보이스 {row.get('inv_unit','')} / PL {row.get('pl_qty_text','')})")
        if row['f_amt']:
            issues.append(f"금액 {row['inv_amount']:,.2f}→{row['pl_amount']:,.2f}")
        if row['f_net']:
            issues.append(f"순중량 {row['inv_net_kg']:,.1f}kg→{row['pl_net']:,.1f}{row.get('pl_net_unit','')}")
        if row['f_origin']:
            issues.append(f"원산지 {row['inv_origin']}→{row['pl_origin']}")
        if row['f_desc']:
            issues.append(f"품목명 [{row['inv_desc']}]→[{row['pl_desc']}]")
        if issues:
            return f"[{row['set_id']} {int(row['line_no'])}행] " + "; ".join(issues) + ";"
        return ""

    df['issue_text'] = df.apply(make_issue_text, axis=1)

    coef_dict = coef_df.set_index('item_group').to_dict(orient='index')
    results = []
    prompts = {}
    grouped = df.groupby('set_id', sort=False)

    for set_id, group in grouped:
        item_group = group['item_group'].iloc[0]
        inv_amt_sum = group['inv_amount'].sum()
        pl_amt_sum = group['pl_amount'].sum()
        inv_net_sum = group['inv_net_kg'].sum()
        pl_net_sum = group['pl_net_kg'].sum()

        f_qty_sum = int(group['f_qty'].sum())
        f_origin_sum = int(group['f_origin'].sum())
        f_desc_sum = int(group['f_desc'].sum())

        energy_gj_mean = group['energy_gj'].mean() if 'energy_gj' in group.columns else 0.0
        power_mwh_mean = group['power_mwh'].mean() if 'power_mwh' in group.columns else 0.0

        issues = [t for t in group['issue_text'].tolist() if t]
        issue_combined = "\n".join(issues) if issues else "규칙 1차 검증에서 불일치 후보 없음"

        amt_pct = (pl_amt_sum - inv_amt_sum) / inv_amt_sum if inv_amt_sum != 0 else 0.0
        net_pct = (pl_net_sum - inv_net_sum) / inv_net_sum if inv_net_sum != 0 else 0.0

        flag_amt = 1 if abs(amt_pct) > tol else 0
        flag_net = 1 if abs(net_pct) > tol else 0

        risk_flags = int(f_qty_sum + f_origin_sum + flag_amt + flag_net)

        c = coef_dict.get(item_group, {})
        t_weight = inv_net_sum / 1000.0

        if item_group == "비료":
            direct_t = c.get('direct_t', 0.85)
            power_coef = c.get('power_mwh_coef', 0.45)
            emission = (t_weight * direct_t) + (power_mwh_mean * power_coef)
            method = "직접+간접 합산"
            calc_inputs = f"순중량 {t_weight:,.1f} t × 직접배출계수 {direct_t} + 전력 {power_mwh_mean:,.0f} MWh × 전력배출계수 {power_coef}"
        elif item_group == "철강" and energy_gj_mean > 0:
            gj_coef = c.get('gj_coef', 0.056)
            emission = energy_gj_mean * gj_coef
            method = "직접배출(실측)"
            calc_inputs = f"실측 에너지 {energy_gj_mean:,.0f} GJ × 직접배출계수 {gj_coef}"
        elif item_group == "철강":
            default_t = c.get('default_t', 2.1)
            emission = t_weight * default_t
            method = "기본값법(실측 없음→전환)"
            calc_inputs = f"순중량 {t_weight:,.1f} t × 기본값 {default_t}"
        else:
            default_t = c.get('default_t', 6.5)
            emission = t_weight * default_t
            method = "기본값법"
            calc_inputs = f"순중량 {t_weight:,.1f} t × 기본값 {default_t}"

        routing = "위험 건 (보고서+진단서)" if risk_flags > 0 else "정상 건 (보고서 전용)"

        doc_info = (f"세트 {set_id} / 품목군 {item_group} / 인보이스 합계: 금액 {inv_amt_sum:,.0f} USD, 순중량 {inv_net_sum:,.0f} kg / "
                    f"패킹리스트 합계: 금액 {pl_amt_sum:,.0f} USD, 순중량 {pl_net_sum:,.0f} kg(규칙이 계산한 값)")
        emission_info = f"{method} 적용. 결과 {emission:,.2f} tCO2e. 입력값: {calc_inputs}"
        check_info = f"{issue_combined}\n합계 차이: 금액 {amt_pct:+.2%}, 순중량 {net_pct:+.2%}"
        assumption_info = f"허용 오차: 합계 기준 ±{tol*100:.1f}% / 배출량 산정 기준 중량: {base_weight}"

        report_txt = REPORT_TEMPLATE_BLOCKC.format(
            doc_info=doc_info,
            emission_info=emission_info,
            check_info=check_info,
            assumption_info=assumption_info
        )
        prompts[f"{set_id}_보고서_프롬프트.txt"] = report_txt

        diag_txt = ""
        if risk_flags > 0:
            diag_txt = DIAG_TEMPLATE_BLOCKC.format(
                doc_info=doc_info,
                emission_info=emission_info,
                check_info=check_info,
                assumption_info=assumption_info
            )
            prompts[f"{set_id}_진단서_프롬프트.txt"] = diag_txt

        results.append({
            'set_id': set_id,
            'item_group': item_group,
            'method': method,
            'emission_tCO2e': round(emission, 2),
            'inv_amount': inv_amt_sum,
            'pl_amount': pl_amt_sum,
            'amt_diff_pct': amt_pct,
            'inv_net_kg': inv_net_sum,
            'pl_net_kg': pl_net_sum,
            'net_diff_pct': net_pct,
            'f_qty_sum': f_qty_sum,
            'f_origin_sum': f_origin_sum,
            'f_desc_sum': f_desc_sum,
            'flag_amt': flag_amt,
            'flag_net': flag_net,
            'risk_flags': risk_flags,
            'routing': routing,
            'report_prompt': report_txt,
            'diag_prompt': diag_txt
        })

    summary_df = pd.DataFrame(results)
    diff_df = df[df['issue_text'] != ''][['set_id', 'line_no', 'issue_text']].copy().reset_index(drop=True)
    return summary_df, diff_df, prompts

def create_block_c_zip(summary_df, diff_df, prompts_dict):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr("세트별_결과.csv", summary_df.to_csv(index=False, encoding='utf-8-sig'))
        z.writestr("라인별_불일치_후보.csv", diff_df.to_csv(index=False, encoding='utf-8-sig'))
        for filename, content in prompts_dict.items():
            z.writestr(filename, content)
    buffer.seek(0)
    return buffer.getvalue()
# -----------------------------------------------------------------------------
@st.cache_resource
def train_prediction_models():
    df = load_csv("master_dataset.csv")
    if df is None:
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
# 4. 상단 네비게이션 및 프리미엄 히어로 배너 (hero-ship.jpg 배경 적용)
# -----------------------------------------------------------------------------
hero_bg_style = f"background: linear-gradient(135deg, rgba(10, 25, 47, 0.88) 0%, rgba(30, 58, 138, 0.78) 55%, rgba(15, 23, 42, 0.92) 100%), url('data:image/jpeg;base64,{b64_ship}') center/cover no-repeat;" if b64_ship else "background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0F172A 100%);"

st.markdown(f"""
<div class="hero-banner" style="{hero_bg_style}">
    <div class="hero-badge">
        <span class="pulse-dot"></span>
        GLOBAL MARITIME TRADE & ESG INTELLIGENCE
    </div>
    <h1 class="hero-title">
        글로벌 무역 컴플라이언스 & ESG 통합 인텔리전스
    </h1>
    <p class="hero-subtitle">
        EU CBAM 탄소국경조정제도 · HS Code 자동추천 · 공급망 ESG 위험도 스크리닝 · 서류 교차검증 · RAG 규정 감사 플랫폼
    </p>
    <div class="hero-chips">
        <span class="hero-chip">🚢 글로벌 해상물류 실시간 연동</span>
        <span class="hero-chip">🤖 노코드 에이전트 ML 파이프라인</span>
        <span class="hero-chip">⚖️ EU 2026 CBAM 규정집 RAG 탑재</span>
        <span class="hero-chip">⚡ Render Cloud Production</span>
        <span class="hero-chip" style="background: rgba(16, 185, 129, 0.2); border-color: #10B981; color: #6EE7B7;">🟢 시스템 상태: 정상 가동 중</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 18px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.18); font-size: 0.82rem;">
        <span style="font-weight: 700; color: #93C5FD; margin-right: 4px;">📍 활성 실습 트랙:</span>
        <span style="background: rgba(234, 179, 8, 0.25); color: #FDE047; padding: 4px 12px; border-radius: 6px; border: 1px solid #EAB308; font-weight: 800;">⭐ [3·4일차] 블록 C: CBAM 배출량 & 서류 교차검증 에이전트</span>
        <span style="color: #94A3B8;">|</span>
        <span style="background: rgba(30, 58, 138, 0.7); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15);"><b>2일차</b> 송장 교차검증 & 거부사유서</span>
        <span style="color: #94A3B8;">|</span>
        <span style="background: rgba(15, 23, 42, 0.7); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15);"><b>5일차</b> 종합 대시보드 & 자동알림</span>
        <span style="color: #94A3B8;">|</span>
        <span style="background: rgba(13, 148, 136, 0.7); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15);"><b>가이드</b> 블록 C 배포 매뉴얼</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4개 핵심 탭 구성 (1일차, 6일차 제외 / 블록 C 전용 star ⭐ 집중)
tabs = st.tabs([
    "⭐ [3·4일차] 블록 C: CBAM 배출량 & 서류 교차검증 에이전트",
    "🔍 [2일차] 서류 교차검증 & 사유서 진단",
    "📊 [5일차] 종합 대시보드 & 자동 알림 관제",
    "🚀 [가이드] 블록 C 앱 배포 & Render 매뉴얼"
])

# =============================================================================
# TAB 2: 5일차 종합 모니터링 대시보드 & 자동 알림 (5일차)
# =============================================================================
with tabs[2]:
    harbor_bg = f"background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 58, 138, 0.72) 60%, rgba(15, 23, 42, 0.9) 100%), url('data:image/jpeg;base64,{b64_harbor}') center/cover no-repeat;" if b64_harbor else "background: linear-gradient(135deg, #1E293B 0%, #1E3A8A 100%);"
    st.markdown(f"""
    <div class="feature-banner" style="{harbor_bg}">
        <div class="feature-tag" style="color: #38BDF8;">[5일차 실습] EXECUTIVE DASHBOARD & DISPATCH SYSTEM</div>
        <h3 class="feature-heading">5일차. 수출입 통관 종합 대시보드 & 리스크 자동 알림 관제탑</h3>
        <p class="feature-desc">5일차 교안: 대시보드 연동 데이터, 국가별 ESG 리스크, 관세율 매핑 및 리스크 감지 시 담당자 자동 매핑 발송</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 1.1 데이터 로드 및 파일 선택기
    with st.expander("📂 대시보드 연동 데이터 선택 및 직접 업로드 (기본 데이터 자동 로드)", expanded=False):
        d_f1, d_f2 = st.columns(2)
        with d_f1:
            dash_opt = st.selectbox("모니터링 데이터셋 선택", ["기본 연동 데이터 (dashboard_integrated.csv)", "테스트 샘플 (dashboard_test_sample.csv)", "📂 직접 CSV 파일 업로드"], key="sel_dash_file")
        with d_f2:
            up_dash = st.file_uploader("모니터링 CSV 파일 업로드", type=["csv"], key="up_dash_file") if "직접" in dash_opt else None

    if up_dash is not None:
        try:
            df_dash = pd.read_csv(up_dash, encoding="utf-8")
        except Exception:
            df_dash = pd.read_csv(up_dash, encoding="cp949")
        dash_name = up_dash.name
    elif "테스트 샘플" in dash_opt:
        df_dash = load_csv("dashboard_test_sample.csv")
        dash_name = "dashboard_test_sample.csv"
    else:
        df_dash = load_csv("dashboard_integrated.csv")
        dash_name = "dashboard_integrated.csv"
        
    df_country = load_csv("master_country_esg.csv")
    df_tariff = load_csv("master_tariff_cbam.csv")
    df_weekly = load_csv("weekly_risk_log.csv")
    df_manager = load_csv("manager_mapping.csv")
    
    st.info(f"📂 **관제 모니터링 데이터:** `{dash_name}` ({len(df_dash) if df_dash is not None else 0}건 로드 완료)")
    
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
                
                # 범례
                st.markdown("""
                <div style="display: flex; gap: 12px; margin-bottom: 8px; font-size: 0.85rem; font-weight: 600;">
                    <span style="color: #EF4444;">■ 서류불일치</span>
                    <span style="color: #F97316;">■ 원산지</span>
                    <span style="color: #3B82F6;">■ HS Code</span>
                    <span style="color: #10B981;">■ ESG</span>
                </div>
                """, unsafe_allow_html=True)
                
                # 순수 HTML/CSS 반응형 누적 막대 차트 (Vega-Lite JS 청크 의존성 제거)
                color_map = {
                    "서류불일치": "#EF4444",
                    "원산지": "#F97316",
                    "HS Code": "#3B82F6",
                    "ESG": "#10B981"
                }
                max_total = max(pivot_risk.sum(axis=1).max(), 1)
                
                html_bars = ['<div style="background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">']
                for date_val, row in pivot_risk.iterrows():
                    total_row = row.sum()
                    pct_width = min((total_row / max_total) * 100, 100)
                    detail_str = ", ".join([f"{k}: {int(v)}" for k, v in row.items() if v > 0])
                    
                    segments = []
                    for k, v in row.items():
                        if v > 0:
                            seg_pct = (v / total_row) * 100
                            c = color_map.get(k, "#64748B")
                            segments.append(f'<div style="width: {seg_pct:.1f}%; background-color: {c}; height: 16px;" title="{k}: {int(v)}건"></div>')
                    
                    seg_html = "".join(segments)
                    bar_block = f"""
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 2px;">
                            <span style="font-weight: 600; color: #334155;">📅 {date_val}</span>
                            <span style="color: #64748B;">총 {int(total_row)}건 ({detail_str})</span>
                        </div>
                        <div style="width: 100%; background: #e2e8f0; border-radius: 4px; overflow: hidden; height: 16px;">
                            <div style="width: {pct_width:.1f}%; display: flex; height: 100%;">
                                {seg_html}
                            </div>
                        </div>
                    </div>
                    """
                    html_bars.append(bar_block)
                html_bars.append('</div>')
                st.markdown("".join(html_bars), unsafe_allow_html=True)
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
# TAB 0: ⭐ [3·4일차] 블록 C: CBAM 배출량 & 서류 교차검증 에이전트
# =============================================================================
with tabs[0]:
    globe_bg = f"background: linear-gradient(135deg, rgba(10, 15, 30, 0.88) 0%, rgba(49, 46, 129, 0.75) 60%, rgba(15, 23, 42, 0.9) 100%), url('data:image/jpeg;base64,{b64_globe}') center/cover no-repeat;" if b64_globe else "background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);"
    st.markdown(f"""
    <div class="feature-banner" style="{globe_bg}">
        <div class="feature-tag" style="color: #FBBF24;">⭐ [핵심 집중 실습] BLOCK C : CBAM & CROSS-VALIDATION AGENT</div>
        <h3 class="feature-heading">⭐ 3·4일차. 블록 C: CBAM 배출량 산정 및 서류 교차검증 통합 에이전트</h3>
        <p class="feature-desc">인보이스 ↔ 패킹리스트 라인 교차 검증, 품목별 CBAM 배출량 자동 산출, 합계 오차율 분석, AI 보고서/진단서 프롬프트 자동 조립 및 전체 결과 다운로드</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 모델 학습
    models = train_prediction_models()
    
    # ⭐ [핵심 집중 실습] 블록 C: CBAM 배출량 산정 및 서류 교차검증 통합 에이전트
    st.markdown("#### ⭐ [핵심 집중 실습] 블록 C: CBAM 배출량 산정 및 서류 교차검증 통합 에이전트")
    st.caption("인보이스와 패킹리스트 라인 데이터를 교차 검증하고, 품목별 배출량 산정, 오차 분석, AI 보고서/진단서 프롬프트를 자동 생성합니다.")
    
    # [1] 파일 선택 / 업로드 구역
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        st.markdown("**📄 라인쌍 서류 데이터 선택 (인보이스 ↔ 패킹리스트)**")
        lines_source_opt = st.selectbox(
            "라인쌍 서류 파일 선택",
            [
                "블록C_라인쌍_기본5세트.csv (기본 5개 세트)",
                "블록C_라인쌍_확인용3세트.csv (추가 확인용 3세트)",
                "📂 새로운 CSV 파일 직접 업로드"
            ],
            index=0,
            key="lines_source_opt"
        )
        uploaded_lines_file = None
        if "직접 업로드" in lines_source_opt:
            uploaded_lines_file = st.file_uploader("라인쌍 CSV 파일 업로드", type=["csv"], key="upload_lines_csv")
            
    with c_f2:
        st.markdown("**📑 배출계수 마스터 데이터 선택**")
        coef_source_opt = st.selectbox(
            "배출계수 파일 선택",
            [
                "블록C_배출계수.csv (기본 마스터 계수)",
                "📂 새로운 CSV 파일 직접 업로드"
            ],
            index=0,
            key="coef_source_opt"
        )
        uploaded_coef_file = None
        if "직접 업로드" in coef_source_opt:
            uploaded_coef_file = st.file_uploader("배출계수 CSV 파일 업로드", type=["csv"], key="upload_coef_csv")
            
    # [2] v1 / v2 프리셋 및 파라미터 제어 구역
    st.markdown("##### ⚙️ 검증 규칙 및 프리셋 설정 (v1 / v2)")
    
    if "block_c_preset" not in st.session_state:
        st.session_state["block_c_preset"] = "v2"
        st.session_state["bc_tol"] = 0.5
        st.session_state["bc_mt_kg"] = True
        st.session_state["bc_skip_orange"] = True
        st.session_state["bc_aliases"] = "REPUBLIC OF KOREA, KOREA, KR, ROK"
        st.session_state["bc_weight_str"] = "인보이스 순중량"
        
    p_btn1, p_btn2, p_msg = st.columns([1.5, 1.5, 3])
    with p_btn1:
        if st.button("🔄 v1 프리셋 (초기/엄격)", help="MT환산 OFF, 기본 별칭만 적용 (교안 초기 버전)"):
            st.session_state["block_c_preset"] = "v1"
            st.session_state["bc_mt_kg"] = False
            st.session_state["bc_aliases"] = "REPUBLIC OF KOREA, KOREA"
            st.rerun()
    with p_btn2:
        if st.button("✨ v2 프리셋 (실전/유연)", help="MT환산 ON (1 MT=1,000kg), KR/ROK 별칭 확장 (교안 권장 버전)"):
            st.session_state["block_c_preset"] = "v2"
            st.session_state["bc_mt_kg"] = True
            st.session_state["bc_aliases"] = "REPUBLIC OF KOREA, KOREA, KR, ROK"
            st.rerun()
    with p_msg:
        cur_preset = st.session_state.get("block_c_preset", "v2")
        preset_badge = "🟢 v2 (실전 유연 모드: MT환산 ON, 확장 별칭)" if cur_preset == "v2" else "🟡 v1 (초기 엄격 모드: MT환산 OFF)"
        st.markdown(f"<div style='padding: 8px 12px; background: rgba(30,58,138,0.2); border-radius: 8px; border: 1px solid rgba(59,130,246,0.3); font-size: 0.9rem; margin-top: 2px;'>현재 모드: <b>{preset_badge}</b></div>", unsafe_allow_html=True)

    with st.expander("🛠️ 세부 파라미터 직접 조정 (허용 오차, 환산 규칙, Orange 메타스킵)"):
        par_col1, par_col2, par_col3 = st.columns(3)
        with par_col1:
            tol_val = st.slider("합계 허용 오차 (TOL, %)", 0.0, 2.0, float(st.session_state.get("bc_tol", 0.5)), 0.1, format="%.1f%%") / 100.0
            skip_orange = st.checkbox("Orange 3 메타 행 스킵 (2~3번째 줄 건너뛰기)", value=st.session_state.get("bc_skip_orange", True))
        with par_col2:
            convert_mt = st.checkbox("패킹리스트 MT ➔ kg 자동 환산 (1 MT = 1,000 kg)", value=st.session_state.get("bc_mt_kg", True))
            base_wt = st.text_input("프롬프트 기준 중량 문구", value=st.session_state.get("bc_weight_str", "인보이스 순중량"))
        with par_col3:
            alias_str = st.text_input("한국 원산지 통일 별칭 (쉼표 구분)", value=st.session_state.get("bc_aliases", "REPUBLIC OF KOREA, KOREA, KR, ROK"))
            alias_list = [a.strip() for a in alias_str.split(",") if a.strip()]

    # [3] 데이터 로드 실행
    actual_lines_filename = "블록C_라인쌍_기본5세트.csv"
    if "확인용 3세트" in lines_source_opt:
        actual_lines_filename = "블록C_라인쌍_확인용3세트.csv"
        
    if uploaded_lines_file is not None:
        df_lines_raw = parse_orange_csv(uploaded_lines_file, skip_orange=skip_orange)
        loaded_lines_name = uploaded_lines_file.name
    else:
        lines_path = get_data_path(actual_lines_filename)
        df_lines_raw = parse_orange_csv(lines_path, skip_orange=skip_orange) if lines_path else None
        loaded_lines_name = actual_lines_filename
        
    actual_coef_filename = "블록C_배출계수.csv"
    if uploaded_coef_file is not None:
        df_coef_raw = parse_orange_csv(uploaded_coef_file, skip_orange=skip_orange)
        loaded_coef_name = uploaded_coef_file.name
    else:
        coef_path = get_data_path(actual_coef_filename)
        df_coef_raw = parse_orange_csv(coef_path, skip_orange=skip_orange) if coef_path else None
        loaded_coef_name = actual_coef_filename

    if df_lines_raw is None or df_coef_raw is None:
        st.error("❌ 블록 C 서류 데이터 또는 배출계수 파일을 로드할 수 없습니다.")
    else:
        st.info(f"📂 **선택된 서류 파일:** `{loaded_lines_name}` ({len(df_lines_raw)} 라인) | **배출계수 파일:** `{loaded_coef_name}` ({len(df_coef_raw)} 품목군)")
        
        summary_df, diff_df, prompts_dict = run_block_c_pipeline(
            df_lines=df_lines_raw,
            df_coef=df_coef_raw,
            tol=tol_val,
            convert_mt=convert_mt,
            korea_aliases=alias_list,
            base_weight=base_wt
        )
        
        total_sets = len(summary_df)
        danger_sets = len(summary_df[summary_df['risk_flags'] > 0])
        safe_sets = total_sets - danger_sets
        total_issues = len(diff_df)
        
        m_bc1, m_bc2, m_bc3, m_bc4 = st.columns(4)
        m_bc1.metric("총 검증 세트", f"{total_sets} 세트")
        m_bc2.metric("정상 세트 (위험=0)", f"{safe_sets} 세트", delta="통관 적격")
        m_bc3.metric("위험 세트 (진단 대상)", f"{danger_sets} 세트", delta=f"-{danger_sets} 건 주의", delta_color="inverse")
        m_bc4.metric("적출된 불일치 라인", f"{total_issues} 건", delta="1차 후보")
        
        bc_t1, bc_t2, bc_t3, bc_t4 = st.tabs([
            "📊 ① 세트별 집계 및 배출량 결과",
            "🔍 ② 라인별 불일치 후보 목록",
            "🤖 ③ AI 보고서·진단서 프롬프트 생성기",
            "📥 ④ 결과 CSV & ZIP 다운로드 센터"
        ])
        
        with bc_t1:
            st.markdown("##### 📋 세트별 집계, CBAM 배출량 산정 및 리스크 플래그")
            st.caption("품목군별 CBAM 공식 적용 결과와 인보이스-패킹리스트 합계 오차율, 위험 신호(risk_flags) 집계 현황입니다.")
            
            disp_summary = summary_df[[
                'set_id', 'item_group', 'method', 'emission_tCO2e',
                'inv_amount', 'pl_amount', 'amt_diff_pct',
                'inv_net_kg', 'pl_net_kg', 'net_diff_pct',
                'risk_flags', 'routing'
            ]].copy()
            
            disp_summary.columns = [
                '세트 ID', '품목군', '배출량 산정 방식', '배출량 (tCO2e)',
                '인보이스 금액($)', 'PL 금액($)', '금액 오차율',
                '인보이스 중량(kg)', 'PL 중량(kg)', '중량 오차율',
                '위험 신호 수', '처리 경로'
            ]
            
            disp_summary['금액 오차율'] = disp_summary['금액 오차율'].apply(lambda x: f"{x:+.2%}")
            disp_summary['중량 오차율'] = disp_summary['중량 오차율'].apply(lambda x: f"{x:+.2%}")
            disp_summary['인보이스 금액($)'] = disp_summary['인보이스 금액($)'].apply(lambda x: f"{x:,.0f}")
            disp_summary['PL 금액($)'] = disp_summary['PL 금액($)'].apply(lambda x: f"{x:,.0f}")
            disp_summary['인보이스 중량(kg)'] = disp_summary['인보이스 중량(kg)'].apply(lambda x: f"{x:,.0f}")
            disp_summary['PL 중량(kg)'] = disp_summary['PL 중량(kg)'].apply(lambda x: f"{x:,.0f}")
            disp_summary['배출량 (tCO2e)'] = disp_summary['배출량 (tCO2e)'].apply(lambda x: f"{x:,.2f}")
            
            def highlight_set(row):
                if "위험" in str(row['처리 경로']):
                    return ['background-color: #FEF2F2; color: #991B1B; font-weight: 600;'] * len(row)
                return ['background-color: #F0FDF4; color: #166534; font-weight: 500;'] * len(row)
                
            st.dataframe(disp_summary.style.apply(highlight_set, axis=1), use_container_width=True, hide_index=True)
            
            st.download_button(
                "📥 세트별_결과.csv 다운로드",
                data=summary_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                file_name="세트별_결과.csv",
                mime="text/csv",
                key="btn_dl_set_res"
            )

        with bc_t2:
            st.markdown("##### 🔍 1차 규칙 기반 라인별 불일치 후보 목록")
            st.caption("수량, 금액, 순중량, 원산지, 품목명 중 인보이스와 패킹리스트 간 값이 다른 모든 라인을 1차 후보로 표시합니다. (표기 차이 여부는 AI가 최종 판정)")
            
            if diff_df.empty:
                st.success("✅ 검증된 모든 라인이 완벽히 일치합니다! (불일치 후보 0건)")
            else:
                disp_diff = diff_df.copy()
                disp_diff.columns = ['세트 ID', '라인 번호', '불일치 상세 이슈 내역 (Issue Text)']
                st.dataframe(disp_diff, use_container_width=True, hide_index=True)
                
                st.download_button(
                    "📥 라인별_불일치_후보.csv 다운로드",
                    data=diff_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                    file_name="라인별_불일치_후보.csv",
                    mime="text/csv",
                    key="btn_dl_diff_res"
                )

        with bc_t3:
            st.markdown("##### 🤖 세트별 AI 프롬프트 생성기 (ChatGPT / Claude 투입용)")
            st.caption("세트를 선택하면 표준 7대 항목 서식의 보고서 프롬프트 및 위험 건 진단서 프롬프트가 실시간 조립됩니다.")
            
            set_options = summary_df['set_id'].tolist()
            sel_set = st.selectbox("조회할 세트 ID 선택", set_options, key="sel_set_prompt")
            
            set_row = summary_df[summary_df['set_id'] == sel_set].iloc[0]
            st.markdown(f"**선택된 세트 상태:** `{sel_set}` ({set_row['item_group']}) | 위험 신호: `{set_row['risk_flags']}개` ➔ **{set_row['routing']}**")
            
            pr_col1, pr_col2 = st.columns(2)
            with pr_col1:
                st.markdown(f"**📄 [{sel_set}] ESG 실사 보고서 프롬프트** (전체 세트 공통)")
                report_content = set_row['report_prompt']
                st.text_area("보고서 프롬프트 전문 (복사 가능)", value=report_content, height=320, key=f"txt_rep_{sel_set}")
                st.download_button(
                    f"💾 {sel_set}_보고서_프롬프트.txt 저장",
                    data=report_content.encode('utf-8'),
                    file_name=f"{sel_set}_보고서_프롬프트.txt",
                    mime="text/plain",
                    key=f"dl_rep_{sel_set}"
                )
                
            with pr_col2:
                st.markdown(f"**🚨 [{sel_set}] 수출 규제 리스크 진단서 프롬프트** (위험 신호 > 0인 경우)")
                if set_row['risk_flags'] > 0:
                    diag_content = set_row['diag_prompt']
                    st.text_area("진단서 프롬프트 전문 (복사 가능)", value=diag_content, height=320, key=f"txt_diag_{sel_set}")
                    st.download_button(
                        f"💾 {sel_set}_진단서_프롬프트.txt 저장",
                        data=diag_content.encode('utf-8'),
                        file_name=f"{sel_set}_진단서_프롬프트.txt",
                        mime="text/plain",
                        key=f"dl_diag_{sel_set}"
                    )
                else:
                    st.info(f"✨ `{sel_set}`은 위험 신호가 0이므로 진단서 생성이 생략됩니다. (정상 건 ➔ 보고서만 생성)")

            st.markdown("""
            > 💡 **AI 대화창 투입 팁**:
            > 1. 위 프롬프트를 복사하여 **ChatGPT** 또는 **Claude** 대화창에 그대로 붙여넣으세요.
            > 2. 진단서 프롬프트를 넣을 때는 `CBAM 가이드라인` 및 `배출량 산정 규정표` 문서를 함께 첨부하면 조항 번호가 정확히 인용됩니다.
            """)

        with bc_t4:
            st.markdown("##### 📦 전체 결과 패키지 다운로드 센터 (CSV & ZIP)")
            st.caption("생성된 CSV 2종과 모든 세트의 보고서·진단서 프롬프트 텍스트 파일들을 한 번에 다운로드할 수 있습니다.")
            
            dl_c1, dl_c2, dl_c3 = st.columns(3)
            with dl_c1:
                st.markdown("**1. 세트별 집계 결과**")
                st.download_button(
                    "📥 세트별_결과.csv 다운로드",
                    data=summary_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                    file_name="세트별_결과.csv",
                    mime="text/csv",
                    key="btn_dl_all_csv1"
                )
            with dl_c2:
                st.markdown("**2. 라인별 불일치 후보**")
                st.download_button(
                    "📥 라인별_불일치_후보.csv 다운로드",
                    data=diff_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                    file_name="라인별_불일치_후보.csv",
                    mime="text/csv",
                    key="btn_dl_all_csv2"
                )
            with dl_c3:
                st.markdown("**3. 전체 프롬프트 & CSV 압축팩**")
                zip_data = create_block_c_zip(summary_df, diff_df, prompts_dict)
                st.download_button(
                    "📦 전체_결과_프롬프트.zip 다운로드",
                    data=zip_data,
                    file_name="BlockC_결과_전체_패키지.zip",
                    mime="application/zip",
                    key="btn_dl_all_zip"
                )

    # 2.4 부가: 선적 서류 일괄 머신러닝 스크리닝 (기존 모델 병합)
    with st.expander("📦 부가 실습: 선적 서류 일괄 머신러닝 스크리닝 (블록 A + 블록 B 결합)"):
        c_threshold = st.slider("리포트 필터 임계값 (오늘 검토할 위험 확률 기준)", 0.1, 0.9, 0.5, 0.05, key="c_thresh_sub")
        df_shipping = load_csv("shipping_docs_sample.csv")
        if df_shipping is not None:
            if "선적ID" in df_shipping.columns and "문서번호" not in df_shipping.columns:
                df_shipping = df_shipping.rename(columns={"선적ID": "문서번호"})
            docs = df_shipping.copy()
            docs["상품설명"] = docs["상품영문명"].astype(str) + " " + docs["사양"].astype(str)
            vec_c = models["tfidf"].transform(docs["상품설명"])
            docs["예측_HS코드"] = models["hs_model"].predict(vec_c)
            X_esg_c = docs[["협력사_환경위반건수", "협력사_노동평가점수"]]
            docs["위험_확률"] = models["esg_model"].predict_proba(X_esg_c)[:, 1]
            docs["판정"] = docs["위험_확률"].apply(lambda p: "위험" if p >= c_threshold else "정상")
            docs["워치리스트_관찰대상"] = docs["예측_HS코드"].apply(lambda c: "예 (주의)" if str(c) in WATCHLIST_HS_CODES else "-")
            sorted_docs = docs.sort_values("위험_확률", ascending=False).reset_index(drop=True)
            review_targets = sorted_docs[sorted_docs["판정"] == "위험"].reset_index(drop=True)
            
            st.markdown(f"**전체 점검 서류:** `{len(sorted_docs)}건` | **오늘 검토 필요:** `{len(review_targets)}건`")
            show_cols = ["문서번호", "상품영문명", "사양", "협력사_환경위반건수", "협력사_노동평가점수", "예측_HS코드", "위험_확률", "판정", "워치리스트_관찰대상"]
            disp_c = sorted_docs[show_cols].copy()
            disp_c["위험_확률"] = (disp_c["위험_확률"] * 100).round(1).astype(str) + "%"
            st.dataframe(disp_c, use_container_width=True, hide_index=True)
            if not review_targets.empty:
                st.download_button(
                    "오늘의 검토 리포트 CSV 다운로드",
                    data=review_targets.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="오늘의_검토리포트.csv",
                    mime="text/csv",
                    key="dl_ml_review"
                )

    # 2.5 보조 실습: 블록 A & 블록 B 개별 테스트
    with st.expander("📂 [보조 실습] 블록 A (HS Code 자동 추천) & 블록 B (협력사 ESG 위험도 스크리닝)", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("블록 A (HS코드) 정확도", f"{models['hs_acc']:.1%}")
        m2.metric("블록 B (ESG위험) 정확도", f"{models['esg_acc']:.1%}")
        m3.metric("블록 B 정밀도 (Precision)", f"{models['esg_prec']:.1%}")
        m4.metric("블록 B 재현율 (Recall)", f"{models['esg_rec']:.1%}")
        
        st.divider()
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
            sel_sample = st.selectbox("테스트 샘플 선택", [s[0] for s in sample_items], key="sel_sample_exp")
            if sel_sample == "Custom Input":
                in_name = st.text_input("상품 영문명", "Aluminum Alloy Bar", key="in_name_exp")
                in_spec = st.text_input("상품 사양", "Diameter 50mm Round 6061-T6", key="in_spec_exp")
            else:
                match = [s for s in sample_items if s[0] == sel_sample][0]
                in_name = st.text_input("상품 영문명", match[0], key="in_name_exp")
                in_spec = st.text_input("상품 사양", match[1], key="in_spec_exp")
                
            if st.button("🎯 HS Code 예측 실행", key="btn_hs_predict_exp"):
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
            in_env = st.slider("협력사 환경 위반 건수 (최근 3년)", 0, 10, 2, key="in_env_exp")
            in_labor = st.slider("협력사 노동/인권 평가 점수 (100점 만점)", 0, 100, 58, key="in_labor_exp")
            threshold_b = st.slider("분류 위험 임계값 (Threshold)", 0.1, 0.9, 0.5, 0.05, key="thresh_b_exp")
            prob_risk = models["esg_model"].predict_proba([[in_env, in_labor]])[0][1]
            is_risk = prob_risk >= threshold_b
            st.markdown(f"**산출된 위험 확률:** `{prob_risk*100:.1f}%` (기준 임계값: `{threshold_b*100:.0f}%`)")
            if is_risk:
                st.error("🚨 판정 결과: **[위험] 공급망 실사 및 시정조치계획 제출 대상**")
            else:
                st.success("✅ 판정 결과: **[정상] 적격 공급사 (지속 모니터링)**")
            st.caption(f"테스트 데이터 Confusion Matrix: TN={models['cm'][0,0]}, FP={models['cm'][0,1]}, FN={models['cm'][1,0]}, TP={models['cm'][1,1]}")

# =============================================================================
# TAB 1 [UI 순서 2번째]: 서류 교차 검증 & 사유서 진단 (2일차)
# =============================================================================
with tabs[1]:
    doc_bg = f"background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(180, 83, 9, 0.75) 60%, rgba(15, 23, 42, 0.92) 100%), url('data:image/jpeg;base64,{b64_harbor}') center/cover no-repeat;" if b64_harbor else "background: linear-gradient(135deg, #78350F 0%, #B45309 100%);"
    st.markdown(f"""
    <div class="feature-banner" style="{doc_bg}">
        <div class="feature-tag" style="color: #FBBF24;">[2일차 실습] DOCUMENT VALIDATION & REJECTION DIAGNOSIS</div>
        <h3 class="feature-heading">2일차. 비정형 서류 교차 검증 & 통관 거부 사유서 자동 진단</h3>
        <p class="feature-desc">2일차 교안: 상업송장(Invoice)과 수출신고필증 라인 단위 교차 대조 및 거부 사유서(Rejection Notice) 핵심 리스크 자동 적출</p>
    </div>
    """, unsafe_allow_html=True)
    
    c_sub1, c_sub2 = st.tabs(["📑 인보이스 vs 수출신고필증 교차 검증", "🛑 통관 거부 사유서(Rejection Docs) 분석"])
    
    with c_sub1:
        st.markdown("#### 상업송장 vs 수출신고필증 자동 교차 검증")
        st.caption("단위 환산 규칙(1 ton = 1,000 kg), HS Code 일치 여부, 원산지 누락을 자동으로 검증합니다.")
        
        # 서류 파일 선택 및 직접 업로드 옵션
        with st.expander("📂 서류 파일 선택 및 직접 업로드 (기본 샘플 자동 로드)", expanded=False):
            f_inv_col, f_exp_col = st.columns(2)
            with f_inv_col:
                inv_sel = st.selectbox("상업송장(Invoice) 파일 선택", ["기본 샘플 (sample_invoice.xlsx)", "📂 직접 엑셀 파일 업로드"], key="sel_inv_file")
                up_inv = st.file_uploader("송장 엑셀 업로드 (.xlsx)", type=["xlsx"], key="up_inv_file") if "직접" in inv_sel else None
            with f_exp_col:
                exp_sel = st.selectbox("수출신고필증 파일 선택", ["기본 샘플 (sample_export_declaration.xlsx)", "📂 직접 엑셀 파일 업로드"], key="sel_exp_file")
                up_exp = st.file_uploader("신고필증 엑셀 업로드 (.xlsx)", type=["xlsx"], key="up_exp_file") if "직접" in exp_sel else None
                
        df_inv = pd.read_excel(up_inv) if up_inv is not None else load_excel("sample_invoice.xlsx")
        df_exp = pd.read_excel(up_exp) if up_exp is not None else load_excel("sample_export_declaration.xlsx")
        inv_name = up_inv.name if up_inv else "sample_invoice.xlsx"
        exp_name = up_exp.name if up_exp else "sample_export_declaration.xlsx"
        st.info(f"📂 **적용 중인 서류:** 송장 `{inv_name}` ↔ 신고필증 `{exp_name}`")
        
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
# TAB 3: 블록 C 앱 배포 가이드 & Render 매뉴얼
# =============================================================================
with tabs[3]:
    st.subheader("🚀 블록 C 앱 배포 가이드 & Render.com 클라우드 배포 매뉴얼")
    st.caption("Windows 무설치 exe 패키징부터 PaaS(Render.com) 무료 배포, 학생 실습용 3단계 프롬프트와 버전 트러블슈팅 완벽 가이드")
    
    guide_t1, guide_t2 = st.tabs([
        "📑 블록 C 앱 배포 가이드 & 버전/프롬프트 매뉴얼 (2026-09-21)",
        "🌐 Render.com 원클릭 무료 배포 & 교육과정 모듈 맵"
    ])
    
    with guide_t1:
        # 0. 핵심 요약
        st.markdown(r"""
        <div style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.15) 0%, rgba(15, 23, 42, 0.25) 100%); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 10px; padding: 18px; margin-bottom: 20px;">
            <h4 style="color: #60A5FA; margin-top: 0; margin-bottom: 8px;">📌 [핵심 요약] 버전 문제와 프롬프트 정리 (2026-09-21)</h4>
            <ul style="margin-bottom: 0; line-height: 1.7; font-size: 0.95rem;">
                <li><b>빌드 실패 원인:</b> Python 3.14 자체가 아니라, pip이 없는 <b>MSYS2 Python</b>이 기본 실행자로 잡혀서 패키지 설치 단계에서 중단된 것이었으며, <b>python.org 공식 Python 3.13</b>으로 빌드하여 완벽히 성공함.</li>
                <li><b>권장 버전:</b> 빌드는 Windows에서 <b>공식 Python 3.13</b>으로 진행 (빌드, 실행, Python 없는 타 PC 무설치 실행까지 교차 검증 완료).</li>
                <li><b>배포 원칙:</b> <code>dist\BlockC</code> <b>폴더 전체를 통째로(zip) 압축하여 전달</b>해야 함. <code>BlockC.exe</code> 단독으로는 <code>_internal</code> 런타임이 없어 실행되지 않음.</li>
                <li><b>실습 순서:</b> 학생에게는 <b>프롬프트 ①(에이전트 만들기) ➔ ②(Streamlit 앱 만들기) ➔ ③(exe 패키징)</b>을 순서대로 부여.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Python 버전 문제
        with st.expander("1. Python 버전 문제 및 해결 (공식 3.13 vs MSYS2)", expanded=True):
            st.markdown(r"""
            #### 🔍 Python 종류별 빌드 가능 여부 및 권장 상태
            빌드 당시 `python` 명령어가 MSYS2에 딸려 온 Python(3.14.5)을 가리켰고, 해당 환경에는 pip이 없었습니다.
            """)
            
            ver_data = [
                {"Python 종류": "공식 3.13 (python.org)", "경로 예시": "C:\\Users\\...\\Python313", "빌드 사용": "가능 (적극 권장)", "확인된 상태": "Windows에서 빌드, 실행, Python 없는 PC 실행 성공"},
                {"Python 종류": "공식 3.12 (python.org)", "경로 예시": "C:\\Users\\...\\Python312", "빌드 사용": "가능 (안정적)", "확인된 상태": "권장 후보 (대다수 라이브러리 완벽 호환)"},
                {"Python 종류": "공식 3.14 (python.org)", "경로 예시": "C:\\Users\\...\\Python314", "빌드 사용": "가능성 있음", "확인된 상태": "리눅스 3.14에서는 빌드/실행 성공, 윈도우는 검증 필요"},
                {"Python 종류": "MSYS2 Python", "경로 예시": "C:\\msys64\\ucrt64\\bin\\python.exe", "빌드 사용": "❌ 불가", "확인된 상태": "pip 없음. 삭제 시 MSYS2 환경 깨지므로 그대로 두고 우회"}
            ]
            st.table(pd.DataFrame(ver_data))
            
            st.markdown(r"""
            **겪은 순서 & 조치 내용:**
            1. `build_exe.bat` 실행 시 `C:\msys64\ucrt64\bin\python.exe: No module named pip` 오류로 패키지 설치 단계에서 중단됨.
            2. 배치파일이 공식 Python(`py` 런처)을 우선 쓰도록 수정. `py -3`은 최상위 버전(3.14)을 고르기 때문에, `3.13 ➔ 3.12 ➔ 3.14` 순서로 직접 지정.
            3. 공식 Python 3.13 설치 후 빌드에 성공하여 `dist\BlockC` 패키지 생성 완료.
            
            **확인 명령어 (cmd에서 실행):**
            ```cmd
            py -0p          :: 공식 설치본 목록과 경로 확인 (MSYS2는 표시 안 됨)
            where python    :: 맨 위 경로가 C:\msys64\...이면 MSYS2 Python이 우선순위임
            python --version:: 버전 번호 없이 'Python'만 나오면 MS Store 가짜 바로가기임
            ```
            - **삭제/유지 팁:** 공식 Python은 `설정 → 앱 → 설치된 앱`에서 제거 가능. MSYS2 Python은 건드리지 않음. 이미 빌드된 `BlockC.exe`는 Python을 지워도 독립 실행됨.
            """)
            
        # 2. 배포 순서 및 폴더 구성
        with st.expander("2. 배포 순서 및 폴더 구성 (PyInstaller onedir)", expanded=True):
            st.markdown(r"""
            #### 📦 배포 순서 (Python 없는 다른 PC에서 실행하기)
            1. **빌드 (한 번):** 공식 Python 3.13 설치(설치 시 `Add python.exe to PATH` 체크) ➔ `block_c_app.py`, `run_app.py`, `build_exe.bat`을 같은 폴더에 두고 `build_exe.bat` 더블클릭 ➔ 5~10분 뒤 `[3/3] 성공!` 메시지와 함께 `dist\BlockC\BlockC.exe` 생성.
            2. **시험:** `BlockC.exe`를 실행하여 샘플 데이터로 기능 확인 ➔ 종료 시 콘솔 창 닫기.
            3. **배포 폴더 구성:** `dist` 안의 `BlockC` 폴더를 복사하고, `BlockC.exe` 옆에 CSV 2개를 동봉.
            4. **압축과 전달:** `BlockC` 폴더를 zip으로 압축(약 400MB)하여 USB, 구글 드라이브 등으로 전달.
            5. **사용 (받는 PC):** zip을 `C:\BlockC`처럼 짧은 영문 경로에 풀고 `BlockC.exe` 더블클릭.
            
            ```text
            📂 배포 폴더 구조:
            BlockC/
            ├── BlockC.exe                  (실행 파일)
            ├── _internal/                  (필수 런타임: Python, Streamlit, pandas 내장)
            ├── 블록C_라인쌍_기본5세트.csv     (기본 샘플 CSV)
            └── 블록C_배출계수.csv            (기본 배출계수 마스터 CSV)
            ```
            > ⚠️ **주의사항:** 폴더 전체를 옮겨야 합니다. `BlockC.exe` 단독으로는 실행되지 않습니다! (화면에서 업로드 기능이 있으므로 CSV가 없어도 무방)
            """)
            
        # 3. 학생용 프롬프트 ① 에이전트 만들기
        with st.expander("3. 학생용 프롬프트 ①: 파이썬 에이전트 만들기 (block_c_agent.py)", expanded=False):
            prompt_1 = """[목표]
서류 데이터(CSV)를 넣으면 배출량 산정, 인보이스–패킹리스트 교차 검증, 보고서·진단서 프롬프트 생성까지 한 번에 되는 파이썬 스크립트 block_c_agent.py 를 만들어 주세요.
실행: python block_c_agent.py                      (기본 5세트)
      python block_c_agent.py 다른파일.csv          (다른 서류 세트)
필요: Python 3.x, pandas
 
[역할 나눔]
- 스크립트(규칙)는 값이 다르면 모두 '불일치 후보'로 표시하고 배출량을 계산합니다.
- 표기 차이인지 진짜 오류인지의 최종 판단은 AI가 합니다. 스크립트는 AI에 붙여넣을 프롬프트 파일만 만듭니다.
- 위험 판정에는 수량, 원산지, 금액(합계), 순중량(합계) 네 가지만 씁니다. 품목명은 후보에는 표시하지만 위험 판정에는 넣지 않습니다.
 
[입력 파일 1] 블록C_라인쌍_기본5세트.csv  (UTF-8, 한 줄 = 인보이스와 패킹리스트의 같은 품목 한 쌍)
열: set_id, line_no, item_group, inv_desc, pl_desc, hs_code, inv_qty, pl_qty, inv_unit, pl_qty_text, inv_amount, pl_amount, inv_net_kg, pl_net, pl_net_unit, inv_origin, pl_origin, energy_gj, power_mwh
- 2~3번째 줄은 열 종류 표시(Orange용)이므로 pd.read_csv(..., skiprows=[1, 2]) 로 건너뜁니다.
- energy_gj, power_mwh 는 비어 있을 수 있고, 같은 세트 안에서는 값이 반복됩니다.
- pl_net 은 패킹리스트 순중량이고 단위는 pl_net_unit (kg 또는 MT) 입니다.
 
[입력 파일 2] 블록C_배출계수.csv  (같은 형식, 2~3번째 줄 건너뜀)
열: item_group, gj_coef, default_t, direct_t, power_mwh_coef
계수 값은 파일에서 읽고 코드에 직접 적지 마세요.
 
[스크립트 위쪽 '내가 정하는 값' 구역]
LINES_CSV = 실행 인자가 있으면 그 파일, 없으면 "블록C_라인쌍_기본5세트.csv"
COEF_CSV = "블록C_배출계수.csv",  OUT_DIR = "결과"
TOL = 0.005                 # 허용 오차(합계 기준) 0.5%
BASE_WEIGHT = "인보이스 순중량"   # 프롬프트에 표시되는 기준 중량 문구
KOREA_ALIASES = ["REPUBLIC OF KOREA", "KOREA"]   # 주석: v2에서 "KR", "ROK" 추가
CONVERT_MT_TO_KG = False                         # 주석: v2에서 True
 
[Step 9-1 라인별 교차 검증]
1. 패킹리스트 순중량 pl_net_kg: CONVERT_MT_TO_KG 가 True 이고 pl_net_unit 이 "MT" 이면 pl_net × 1000, 아니면 pl_net 그대로.
2. 원산지 비교용 값: KOREA_ALIASES 에 있는 표기는 모두 "KOREA" 로 통일한 뒤 비교.
3. 아래가 서로 다르면 각각 불일치 표시(f_qty, f_amt, f_net, f_origin, f_desc). 값이 다르면 무조건 후보이며, 표기 차이인지 판단하지 않습니다.
   수량(inv_qty ≠ pl_qty), 금액(inv_amount ≠ pl_amount), 순중량(inv_net_kg ≠ pl_net_kg), 원산지(통일 후), 품목명(inv_desc ≠ pl_desc)
4. 불일치가 있는 행은 issue_text 를 만듭니다. 아래 순서와 형식으로 이어 붙이고, 숫자는 f"{x:,.10g}" 로 씁니다.
   시작: "[SET-02 2행] "
   수량:   "수량 30→32 (인보이스 {inv_unit} / PL {pl_qty_text}); "
   금액:   "금액 20,000→25,000; "
   순중량: "순중량 12,000 kg→12,800 kg; "   (뒤쪽 단위는 pl_net_unit, 앞쪽은 kg)
   원산지: "원산지 {inv_origin}→{pl_origin}; "   (통일 전 원래 표기)
   품목명: "품목명 [{inv_desc}]→[{pl_desc}]; "
 
[Step 9-2 세트별 집계, 배출량, 위험 신호]
- set_id 별로 묶어서: item_group 첫 값 / 금액·순중량 합계(인보이스, 패킹리스트) / 수량·원산지·품목명 불일치 건수 합 / energy_gj, power_mwh 는 평균(라인마다 같은 값이 반복되므로 합계가 아님. 비어 있으면 실측값 없음) / issue_text 는 공백으로 이어 붙임.
- 배출계수는 item_group 기준으로 붙입니다.
- 합계 차이율: amt_pct = (패킹리스트 합 − 인보이스 합) / 인보이스 합, net_pct 도 같은 방식. 절대값이 TOL 을 넘으면 flag_amt, flag_net = 1, 이내면 0.
- 배출량(tCO2e): t = 인보이스 순중량 합계(kg) / 1000
   · 비료: t × direct_t + power_mwh × power_mwh_coef  → 방식 문구 "직접+간접 합산"
   · 철강, energy_gj > 0: energy_gj × gj_coef  → "직접배출(실측)"
   · 철강, 실측 없음: t × default_t  → "기본값법(실측 없음→전환)"
   · 그 외: t × default_t  → "기본값법"
- 위험 신호 수 risk_flags = 수량 불일치 건수 + 원산지 불일치 건수 + flag_amt + flag_net  (품목명은 제외)
 
[Step 10 / 10-1 프롬프트 조립]
세트마다 아래 네 문장을 만들어 템플릿의 {doc_info} {emission_info} {check_info} {assumption_info} 자리에 채웁니다 (str.format 사용).
- doc_info: "세트 SET-02 / 품목군 알루미늄 / 인보이스 합계: 금액 143,500 USD, 순중량 37,000 kg / 패킹리스트 합계: 금액 143,500 USD, 순중량 37,800 kg(규칙이 계산한 값)"
- emission_info: "기본값법 적용. 결과 240.50 tCO2e. 입력값: 순중량 37.0 t × 기본값 6.5"
   입력값 문구는 방식별로 다릅니다.
   비료 "순중량 150.0 t × 직접배출계수 0.85 + 전력 120 MWh × 전력배출계수 0.45"
   철강 실측 "실측 에너지 5,200 GJ × 직접배출계수 0.056"
- check_info: 세트의 issue_text 를 이어 붙인 것 (없으면 "규칙 1차 검증에서 불일치 후보 없음") + 줄바꿈 + "합계 차이: 금액 +0.00%, 순중량 +2.16%"  (f"{x:+.2%}")
- assumption_info: "허용 오차: 합계 기준 ±0.5% / 배출량 산정 기준 중량: 인보이스 순중량"
템플릿 2개는 아래 텍스트를 한 글자도 바꾸지 말고 REPORT_TEMPLATE, DIAG_TEMPLATE 문자열로 넣어 주세요. ({ } 는 위 네 자리 외에는 없습니다.)
 
[출력]
- '결과' 폴더(없으면 생성). 실행을 시작할 때 이전 *_프롬프트.txt 는 지웁니다.
- 세트별_결과.csv: 세트별 집계 전체 열 (utf-8-sig)
- 라인별_불일치_후보.csv: 불일치가 있는 행만, 열은 set_id, line_no, issue_text (utf-8-sig)
- 콘솔: 입력 파일명과 허용 오차를 먼저 출력하고, 세트 / 배출량(tCO2e) / 위험신호 / 처리를 표로 출력합니다.
  처리 문구는 "위험 건 → 보고서 + 진단서" 또는 "정상 건 → 보고서만" 입니다."""
            st.code(prompt_1, language="markdown")
            
            st.markdown("""
            **검증 기대값 (기본 5세트 실행 결과):**
            - `SET-01 철강`: 291.20 tCO2e / 위험 0 / 정상 건 (보고서만)
            - `SET-02 알루미늄`: 240.50 tCO2e / 위험 2 / 위험 건 (보고서 + 진단서)
            - `SET-03 비료`: 181.50 tCO2e / 위험 1 / 위험 건 (보고서 + 진단서)
            - `SET-04 알루미늄`: 198.25 tCO2e / 위험 1 / 위험 건 (보고서 + 진단서)
            - `SET-05 철강`: 94.92 tCO2e / 위험 1 / 위험 건 (보고서 + 진단서)
            """)

        # 4. 학생용 프롬프트 ② 앱 만들기
        with st.expander("4. 학생용 프롬프트 ②: Streamlit 앱 만들기 (block_c_app.py)", expanded=False):
            prompt_2 = """[첨부] block_c_agent.py (파이썬 원본 스크립트)
 
이 스크립트를 Streamlit 웹 화면으로 바꿔 주세요. 파일명은 block_c_app.py 로 해 주세요.
 
[원칙]
- 검증 규칙, 배출량 계산, 프롬프트 템플릿(REPORT_TEMPLATE, DIAG_TEMPLATE)은 원본 로직을 그대로 유지하고 바꾸지 마세요.
- 원본의 설정값(TOL, KOREA_ALIASES, CONVERT_MT_TO_KG, BASE_WEIGHT)은 화면에서 조정할 수 있게 하고, 계산 로직은 화면 코드와 분리된 함수로 만드세요.
 
[화면 요구사항]
1. 사이드바: 라인쌍 CSV·배출계수 CSV 업로드(업로드가 없으면 같은 폴더의 기본 CSV 자동 사용), "2~3번째 줄 건너뛰기" 체크박스, 허용 오차(%), 한국 표기 목록, MT→kg 환산 체크박스, 기준 중량 문구, v1/v2 설정 전환
2. 결과 탭: ① 세트별 결과(위험 세트 강조) ② 불일치 후보 ③ 프롬프트(세트 선택, 복사 버튼) ④ 결과 전체 zip 다운로드
3. 상단에 검증 세트 수·위험 세트 수 요약 표시
4. 필수 열이 없거나 파일이 잘못되면 원인을 알려 주는 오류 메시지
5. 샘플 데이터로 체험하는 기능
 
[기타]
- pip install streamlit pandas 만으로 Windows에서 실행되고, 한글 CSV(UTF-8, CP949)를 읽을 수 있게 해 주세요.
- 만든 뒤 직접 실행해서 오류가 없는지 확인해 주세요.
- 마지막에 실행 방법(streamlit run block_c_app.py)을 알려 주세요."""
            st.code(prompt_2, language="markdown")

        # 5. 학생용 프롬프트 ③ exe 만들기
        with st.expander("5. 학생용 프롬프트 ③: 무설치 실행파일 만들기 (build_exe.bat)", expanded=False):
            prompt_3 = r"""[첨부] block_c_app.py
 
이 Streamlit 앱을 Python이 없는 Windows PC에서도 실행되는 프로그램으로 만들고 싶습니다.
 
- PyInstaller onedir 방식으로, 런처 run_app.py 와 빌드용 build_exe.bat 을 만들어 주세요.
- 런처는 exe와 같은 폴더의 CSV를 읽게 하고(os.chdir), 브라우저를 자동으로 열게 해 주세요.
- 아래 문제를 미리 피해 주세요.
  · 첫 실행 때 Streamlit이 콘솔에서 이메일을 물어 멈추는 문제 (--server.headless=true 로 두고 브라우저는 webbrowser 로 직접 열기)
  · --collect-all streamlit, --copy-metadata streamlit, --add-data 옵션 필요
  · global.developmentMode=false 필요
  · PC에 MSYS2처럼 pip 없는 Python이 있을 수 있으니 py 런처(py -3.13)를 우선 쓰고, 오류가 나면 멈춰서 원인과 build_log.txt 를 보여 줄 것
- 빌드 후 dist\\BlockC 폴더를 통째로 배포하는 방법과 실행 순서를 알려 주세요."""
            st.code(prompt_3, language="markdown")

        # 6. AI 프롬프트 투입 방법 & 7. 오류별 문제 해결
        with st.expander("6. AI 프롬프트 대화창 투입 요령 & 7. 오류별 해결 FAQ", expanded=False):
            st.markdown("""
            #### 💬 6. 앱이 만든 프롬프트를 AI에 넣는 방법
            1. 앱의 프롬프트 탭에서 세트를 고르고 복사 버튼을 누릅니다.
            2. ChatGPT나 Claude의 새 대화창에 그대로 붙여넣습니다.
            3. 진단서 프롬프트는 `CBAM 가이드라인`과 `배출량 산정 규정표` 문서를 함께 첨부합니다. (프롬프트가 첨부 문서의 조항만 인용하도록 설계됨)
            4. 위험 신호가 없는 세트는 보고서 프롬프트만 생성되며, 위험 신호가 있는 세트만 진단서 프롬프트가 추가됩니다.
            
            **결과 튜닝 후속 질의 예시:**
            - `4번 불일치 목록을 표로 다시 정리해 줘`
            - `표기 차이로 뺀 항목과 그 이유를 따로 보여 줘`
            
            ---
            #### 🛠️ 7. 오류별 문제 해결 (Troubleshooting)
            """)
            
            faq_data = [
                {"증상": "No module named pip (경로에 C:\\msys64)", "원인": "pip 없는 MSYS2 Python이 기본 인터프리터로 선택됨", "해결 조치": "공식 Python 3.13 설치 후 새 cmd 창에서 build_exe.bat 재실행"},
                {"증상": "배치가 몇 초 만에 끝나고 '아무 키나 누르세요'", "원인": "예전 배치파일이 오류 시 pause로 넘어감", "해결 조치": "새 build_exe.bat 사용 (오류 발생 시 [오류] 메시지에서 정지)"},
                {"증상": "3.13을 설치했는데 3.14로 표시됨", "원인": "py -3 명령어가 시스템 내 최상위 버전을 우선 선택함", "해결 조치": "3.13을 직접 지정하는 새 배치파일 사용, 'py -0p'로 목록 확인"},
                {"증상": "python --version에 Python만 나옴", "원인": "Microsoft Store의 빈 바로가기 앱", "해결 조치": "'py -0p'로 공식 설치 확인. 윈도우 앱 실행 별칭에서 python.exe 끄기"},
                {"증상": "BlockC.exe 실행 시 콘솔 창이 바로 닫힘", "원인": "경로 한글 문제 또는 필수 파일 누락", "해결 조치": "cmd 창에서 BlockC.exe를 직접 실행하여 에러 메시지 확인"},
                {"증상": "브라우저가 자동으로 안 열림", "원인": "자동 열기 지연 또는 팝업 차단", "해결 조치": "주소창에 http://localhost:8501 직접 입력"}
            ]
            st.table(pd.DataFrame(faq_data))

    with guide_t2:
        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            st.markdown("""
            #### 🌐 Render.com 원클릭 무료 배포 절차
            1. **GitHub 저장소 푸시:**
               - 모든 소스코드와 데이터셋이 포함된 리포지토리를 GitHub에 푸시합니다.
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
               - 빌드 완료 후 제공되는 공개 URL(예: `https://esg-trade-compliance-platform.onrender.com`)로 전 세계 어디서나 즉시 접속 가능!
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
            - **1일차:** CBAM 품목별 가이드라인, 노코드 리스크 스코어링 (1일차 탭)
            - **2일차:** 서류 클리닝, 인보이스-신고필증 교차검증, 거부사유서 진단 (2일차 탭)
            - **3·4일차:** 블록 A(HS Code), 블록 B(ESG 스크리닝), 블록 C(통합 에이전트) (3·4일차 탭)
            - **5일차:** 종합 모니터링 대시보드, 자동 알림 발송 시뮬레이터 (5일차 탭)
            - **6일차:** 통관규정 RAG 지식 검색, 신규 계약서/바이어 메일 감사 (6일차 탭)
            """)
        st.info("💡 배포 파일 체크: `requirements.txt`, `Procfile`, `render.yaml` 및 `data/` 디렉토리가 모두 포함되어 클라우드 환경에서 의존성 없이 즉시 작동합니다.")
