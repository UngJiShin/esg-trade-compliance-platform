# 글로벌 무역 컴플라이언스 & ESG 통합 인텔리전스 플랫폼
> Global Trade Compliance & Supply Chain ESG Intelligence Platform

생성형 AI 및 머신러닝 기반 무역 컴플라이언스, 관세 최적화, EU CBAM 내재배출량 산정, 공급망 ESG 위험도 스크리닝, 서류 교차검증 및 RAG 규정 감사 통합 웹 애플리케이션입니다.

---

## 🌟 주요 기능 (탭별 구성)

### 1. 📊 종합 모니터링 대시보드 (Executive Dashboard & Alerts)
- **KPI 메트릭 요약**: 총 점검 건수, 고위험 건수(위험확률 50% 이상), 교차검증 불일치 건수, CBAM 규제 대상 품목 수
- **주간 리스크 추이 차트**: 최근 7일간 리스크 유형별(서류불일치, 원산지, HS Code, ESG) 발생 빈도 시각화
- **실시간 필터링**: 리스크 유형 및 위험 확률 임계값별 동적 필터링 및 컬러 배지 테이블
- **자동 알림 발송 시뮬레이터**: 위험 건 선택 시 담당자 매핑표(`김서류`, `박원산`, `이품목`, `정환경`) 연동 및 맞춤형 경고 메일 자동 생성 및 발송 시뮬레이션

### 2. 🤖 HS Code & ESG 예측 에이전트 (3·4일차 블록 A·B·C)
- **블록 A (HS Code 추천 모델)**: 영문 상품명 및 규격 텍스트 기반 TF-IDF + RandomForest 머신러닝 모델, 전략물자/워치리스트 자동 태깅
- **블록 B (ESG 위험도 스크리닝)**: 협력사 환경 위반 건수 및 노동평가 점수 기반 LogisticRegression 모델, 임계값(Threshold 0.1~0.9) 슬라이더 반응형 스크리닝 및 혼동행렬(Confusion Matrix)
- **블록 C (노코드 통합 예측 에이전트)**: 문서번호 기준 안전 병합, HS 추천과 위험확률 일괄 산출, "오늘 검토가 필요한 건" 자동 상단 정렬 및 CSV 다운로드

### 3. 🔍 서류 교차 검증 & 사유서 진단 (2일차)
- **인보이스 vs 수출신고필증 라인 대조**: 단위 환산(1 ton = 1,000 kg 자동 환산), 수량, 금액, HS Code 불일치 및 원산지 결측/불일치 자동 탐지
- **통관 거부 사유서(Rejection Notice) 자동 진단**: 비정형 거부 사유서(RJ-001 ~ RJ-005) 분석, 5대 리스크 카테고리 매핑 및 필수 보완 서류 가이드

### 4. ⚖️ CBAM 배출량 산정 & 규제 스코어링 (1일차 & Round 5)
- **EU CBAM 내재배출량 산정 계산기**: Scope 1(직접 연료 연소) + Scope 2(간접 전력 소비량 × 전력망 배출계수) 산출, 톤당 배출집약도 및 예상 탄소 인증서 비용(€) 시뮬레이션
- **노코드 에이전트 리스크 스코어러 (Round 5)**: 30대 리스크 키워드 가중합($\text{Score} = \sum \text{Count} \times \text{Weight}$), 초록(0~5)/노랑(6~15)/빨강(16+) 등급 및 대응 가이드

### 5. 📚 RAG 규정 지식 검색 & 계약서 감사 (5·6일차)
- **RAG 규정 지식 검색 엔진**: 통관 규정집 및 CBAM 가이드라인 대상 TF-IDF 코사인 유사도 검색, 10대 표준 질의(Q01~Q10) 정답 조항 자동 대조
- **신규 수출계약서(Nordic Metals) 실시간 감사**: 제4조(배출량 정보 미제공 - CBAM 위반 독소조항), 제1조(원산지 미확정 리스크) 자동 적출 및 수정 권고안 제시
- **바이어 문의 메일 분석**: 독일 바이어(Anna Weber) 3대 사전 질의에 대한 공식 영문/국문 컴플라이언스 답신 초안 자동 생성

### 6. 🚀 Render 배포 & 시스템 가이드
- Render.com 클라우드 원클릭 배포 설정 및 로컬 실행 가이드 제공

---

## 🛠️ 기술 스택
- **Language**: Python 3.10+ / 3.11 / 3.13
- **Framework**: Streamlit
- **Data & ML**: pandas, numpy, scikit-learn, openpyxl, pypdf, python-docx

---

## 💻 로컬 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 웹 앱 실행
streamlit run app.py
```
브라우저에서 `http://localhost:8501` 접속

---

## 🌐 Render.com 원클릭 무료 배포 절차

1. 본 폴더(`app.py`, `requirements.txt`, `Procfile`, `render.yaml`, `data/`)를 GitHub 저장소에 푸시합니다.
2. [Render Dashboard](https://dashboard.render.com/) 접속 후 **New +** ➔ **Web Service** 선택
3. GitHub 저장소 선택 후 아래 설정 입력:
   - **Name**: `esg-trade-compliance-platform`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Plan**: `Free`
4. **Deploy Web Service**를 클릭하면 몇 분 내로 무료 공용 URL이 발급됩니다.
