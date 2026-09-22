# -*- coding: utf-8 -*-
"""
[블록 C] 노코드 통합 예측 에이전트 빌드 — 파이썬 버전
======================================================================
Step 1~8은 HS Code 모델과 ESG 위험도 모델을 "따로" 실행해 결과를 확인하는
방식이었다. 블록 C는 이 둘을 하나의 파이프라인("에이전트")으로 묶어서,
서류를 넣기만 하면 HS Code 추천값과 위험 예측 확률이 동시에 나오고,
그중 '오늘 검토가 필요한 건'만 자동으로 상단에 걸러지는 리포트를 만든다.

핵심 설계 포인트 (Step 9 트러블슈팅 반영):
    두 모델의 출력을 이어붙이지 않고, 반드시 '문서번호' 기준으로
    병합(merge)한다. 이렇게 해야 두 예측 결과의 행 순서가 달라져도
    엉뚱한 서류끼리 섞이지 않는다.

이 스크립트의 IntegratedPredictionAgent 클래스는 5일차 통합 대응
대시보드에서도 그대로 재사용할 수 있도록 독립된 모듈로 설계했다.

■ 실행 방법
    python block_c_integrated_agent.py

■ 필요한 파일 (이 스크립트와 같은 폴더에 둘 것)
    master_dataset.csv (또는 .xlsx)   — 학습용 마스터 데이터셋
    블록C_검증서류_샘플.csv            — Step 9 검증용 신규 서류 3건
    선적서류_샘플.csv                  — Step 10 리포트용 이번 주 선적 서류 (블록 B와 공용)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))

# 교육용 예시 워치리스트 — 실제 제재·전략물자 리스트가 아니라
# 스크리닝이 이렇게 코드에 연결된다는 것을 보여주는 예시일 뿐이다.
EXAMPLE_WATCHLIST_HS_CODES = {"853400"}


# ============================================================
# 에이전트 본체 — Step 9의 "통합 파이프라인"을 재사용 가능한 클래스로 구현
# ============================================================
class IntegratedPredictionAgent:
    """HS Code 모델 + ESG 위험도 모델을 하나로 묶은 통합 예측 에이전트.

    build(df_master)로 두 모델을 한 번 학습해두면,
    predict(docs_df)를 호출할 때마다 새 서류 묶음에 바로 적용할 수 있다.
    """

    def __init__(self):
        self.tfidf = None
        self.hs_model = None
        self.esg_model = None
        self.hs_test_acc = None

    # ---- 모델 학습 (Step 3 + Step 5에 해당, 한 번만 실행) ----
    def build(self, df_master: pd.DataFrame):
        df = df_master.copy()

        # HS Code 모델
        df["상품설명"] = df["상품영문명"].astype(str) + " " + df["사양"].astype(str)
        y_hs = df["HS코드_정답"].astype(str)
        Xtr_t, Xte_t, ytr_hs, yte_hs = train_test_split(
            df["상품설명"], y_hs, test_size=0.2, random_state=42, stratify=y_hs
        )
        self.tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
        Xtr_vec = self.tfidf.fit_transform(Xtr_t)
        Xte_vec = self.tfidf.transform(Xte_t)
        self.hs_model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.hs_model.fit(Xtr_vec, ytr_hs)
        self.hs_test_acc = accuracy_score(yte_hs, self.hs_model.predict(Xte_vec))

        # ESG 위험도 모델
        X_esg = df[["협력사_환경위반건수", "협력사_노동평가점수"]]
        y_esg = (df["리스크_정답"].astype(str).str.strip() == "위험").astype(int)
        self.esg_model = LogisticRegression(random_state=42)
        self.esg_model.fit(X_esg, y_esg)

        return self

    # ---- Step 9: 병렬 예측 후 문서번호 기준 병합(Merge) ----
    def predict(self, docs_df: pd.DataFrame, id_col: str = "문서번호") -> pd.DataFrame:
        if id_col not in docs_df.columns:
            raise ValueError(f"입력 서류에 '{id_col}' 열이 없습니다. 문서번호 기준 병합이 불가능합니다.")

        docs = docs_df.copy()
        docs["상품설명"] = docs["상품영문명"].astype(str) + " " + docs["사양"].astype(str)

        # ① HS Code 모델 결과 (문서번호 + 예측값만 담은 별도 표)
        vec = self.tfidf.transform(docs["상품설명"])
        hs_result = docs[[id_col]].copy()
        hs_result["예측_HS코드"] = self.hs_model.predict(vec)

        # ② ESG 모델 결과 (문서번호 + 예측값만 담은 별도 표)
        X_esg = docs[["협력사_환경위반건수", "협력사_노동평가점수"]]
        esg_result = docs[[id_col]].copy()
        esg_result["위험_확률"] = self.esg_model.predict_proba(X_esg)[:, 1]

        # ③ Merge 위젯에 해당하는 부분: 순서가 아니라 문서번호로 병합
        merged = docs.merge(hs_result, on=id_col).merge(esg_result, on=id_col)
        merged["워치리스트_관찰대상"] = merged["예측_HS코드"].apply(
            lambda code: "예" if code in EXAMPLE_WATCHLIST_HS_CODES else "-"
        )
        return merged

    # ---- Step 10: 위험도순 정렬 + 임계값 필터 → "검토 필요" 리포트 ----
    def risk_report(self, merged_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        sorted_df = merged_df.sort_values("위험_확률", ascending=False).reset_index(drop=True)
        sorted_df["판정"] = sorted_df["위험_확률"].apply(lambda p: "위험" if p >= threshold else "정상")
        review_needed = sorted_df[sorted_df["판정"] == "위험"].reset_index(drop=True)
        return sorted_df, review_needed


# ============================================================
# 데이터 로드 유틸
# ============================================================
def load_csv_or_xlsx(basename):
    csv_path = os.path.join(HERE, f"{basename}.csv")
    xlsx_path = os.path.join(HERE, f"{basename}.xlsx")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, encoding="utf-8-sig")
    if os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path)
    raise FileNotFoundError(f"{basename}.csv 또는 .xlsx 파일이 필요합니다.")


# ============================================================
# 실행부: Step 9 검증 → Step 10 리포트
# ============================================================
def main():
    THRESHOLD = 0.5  # 3.7 Step 7에서 팀이 정한 임계값으로 바꿔서 재실행하세요.

    print("=" * 70)
    print("에이전트 빌드 — HS Code 모델 + ESG 위험도 모델 학습")
    print("=" * 70)
    df_master = load_csv_or_xlsx("master_dataset")
    agent = IntegratedPredictionAgent().build(df_master)
    print(f"학습 완료 — HS Code 모델 테스트 정확도: {agent.hs_test_acc:.1%}")
    print()

    # ---- Step 9: 오늘 사용하지 않은 신규 서류 2~3건으로 파이프라인 검증 ----
    print("=" * 70)
    print("[Step 9] 통합 예측 파이프라인 검증 — 신규 서류 3건 (블록C_검증서류_샘플.csv)")
    print("=" * 70)
    docs = pd.read_csv(os.path.join(HERE, "블록C_검증서류_샘플.csv"), encoding="utf-8-sig")
    merged = agent.predict(docs, id_col="문서번호")
    print(merged[["문서번호", "상품영문명", "예측_HS코드", "위험_확률", "워치리스트_관찰대상"]].to_string(index=False))
    print()
    print("확인: 문서번호 기준으로 HS Code 추천값과 위험 예측 확률이 같은 행에 정확히 합쳐졌는지 확인하세요.")
    print()

    # ---- Step 10: 이번 주 선적 서류 전체에 적용해 위험 신호 자동 정렬 리포트 생성 ----
    print("=" * 70)
    print(f"[Step 10] 위험 신호 자동 정렬 리포트 — 선적서류_샘플.csv (임계값 {THRESHOLD})")
    print("=" * 70)
    ships = pd.read_csv(os.path.join(HERE, "선적서류_샘플.csv"), encoding="utf-8-sig")
    ships = ships.rename(columns={"선적ID": "문서번호"})  # 병합 기준 열 이름 통일
    ship_merged = agent.predict(ships, id_col="문서번호")
    sorted_all, review_needed = agent.risk_report(ship_merged, threshold=THRESHOLD)

    print("전체 정렬 결과 (위험도 높은 순):")
    print(sorted_all[["문서번호", "상품영문명", "예측_HS코드", "위험_확률", "판정"]].to_string(index=False))
    print()
    print(f"오늘 검토가 필요한 건 ({len(review_needed)}건):")
    if len(review_needed) > 0:
        print(review_needed[["문서번호", "상품영문명", "예측_HS코드", "위험_확률", "워치리스트_관찰대상"]].to_string(index=False))
    else:
        print("(임계값을 넘는 위험 건 없음)")
    print()

    out_path = os.path.join(HERE, "오늘의_검토리포트.csv")
    review_needed.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"리포트 저장 완료: {out_path}")
    print()
    print("핵심 포인트: 이 agent.predict() + agent.risk_report() 두 함수만 있으면,")
    print("내일부터는 서류 파일만 바꿔 넣어도 사람이 두 모델을 따로 돌릴 필요 없이")
    print("'검토가 필요한 건'만 자동으로 걸러집니다. (5일차 대시보드에서 그대로 재사용)")


if __name__ == "__main__":
    main()
