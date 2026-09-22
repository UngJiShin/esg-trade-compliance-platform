# -*- coding: utf-8 -*-
"""
[블록 B] 공급망 ESG 위험도 예측 및 선적 서류 스크리닝 — 파이썬 버전
======================================================================
Orange3로 하던 Step 5~8을 파이썬 코드로 그대로 구현한 스크립트.
[블록 A]에서 만든 HS Code 모델과 이 스크립트의 ESG 위험도 모델을
함께 적용해, 신규 선적 서류 묶음에서 위험 건을 스크리닝한다.

■ 실행 방법
    python block_b_esg_screening.py

■ 필요한 파일 (이 스크립트와 같은 폴더에 둘 것)
    master_dataset.csv (또는 .xlsx)  — 학습용 마스터 데이터셋
    선적서류_샘플.csv                — Step 8에서 스크리닝할 신규 선적 서류 묶음
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))

# 교육용 예시 워치리스트: 실제 제재·전략물자 리스트가 아니라
# "HS Code 스크리닝이 이런 방식으로 붙는다"를 보여주기 위한 샘플일 뿐이다.
# 실무에서는 전략물자관리원 / OFAC SDN 리스트 등 공식 데이터베이스를 연동해야 한다.
EXAMPLE_WATCHLIST_HS_CODES = {"853400"}  # 예: 회로기판류 → 이중용도 품목 가능성 관찰 대상(예시)


# ============================================================
# Step 5. ESG 위험도 예측 모델 구성
#   [Orange 위젯: File→선택] 환경위반건수·노동점수를 특징으로 선택
#   [Orange 위젯: Logistic Regression] 분류 모델 학습
# ============================================================
def load_master_dataset():
    csv_path = os.path.join(HERE, "master_dataset.csv")
    xlsx_path = os.path.join(HERE, "master_dataset.xlsx")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError("master_dataset.csv 또는 master_dataset.xlsx가 필요합니다.")
    return df


def train_esg_model(df):
    X = df[["협력사_환경위반건수", "협력사_노동평가점수"]]
    y = (df["리스크_정답"].astype(str).str.strip() == "위험").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]

    print(f"[Step 5] ESG 위험도 예측 모델 학습 완료 — 학습 {len(X_train)}건 / 테스트 {len(X_test)}건")
    print()
    return model, X_test, y_test, proba_test


# ============================================================
# Step 6. 혼동행렬로 예측 결과 확인
#   [Orange 위젯: Confusion Matrix / Test & Score]
# ============================================================
def show_confusion_matrix(y_test, proba_test, threshold=0.5):
    y_pred = (proba_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["실제_정상", "실제_위험"], columns=["예측_정상", "예측_위험"])

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"[Step 6] Confusion Matrix (임계값 {threshold})")
    print(cm_df)
    print(f"Precision: {precision:.3f} / Recall: {recall:.3f}")
    if recall < 0.7:
        print("체크포인트: Recall이 낮습니다 — 실제 위험 건을 놓치는 비율이 높아 컴플라이언스 관점에서 주의가 필요합니다.")
    print()
    return cm_df, precision, recall


# ============================================================
# Step 7. 임계값(Threshold) 조정 실습
#   [Orange 위젯: Predictions + 임계값 슬라이더]
# ============================================================
def compare_thresholds(y_test, proba_test):
    rows = []
    for th in [0.7, 0.5, 0.3]:
        y_pred = (proba_test >= th).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        rows.append({"임계값": th, "Precision": round(p, 3), "Recall": round(r, 3)})
    table = pd.DataFrame(rows)
    print("[Step 7] 임계값별 Precision·Recall 비교")
    print(table.to_string(index=False))
    print("토론 포인트: 우리 회사라면 Recall을 우선할지(위험을 더 많이 잡아냄),")
    print("             Precision을 우선할지(오탐을 줄임) 근거를 1~2문장으로 정리해보세요.")
    print()
    return table


# ============================================================
# Step 8. 선적 서류 스크리닝 자동 적용 및 정리
#   [Orange 위젯: File + Merge] HS Code 모델·ESG 모델을 동시 통과
# ============================================================
def train_hs_model(df):
    df = df.copy()
    df["상품설명"] = df["상품영문명"].astype(str) + " " + df["사양"].astype(str)
    y = df["HS코드_정답"].astype(str)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["상품설명"], y, test_size=0.2, random_state=42, stratify=y
    )
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train_text)
    X_test_vec = tfidf.transform(X_test_text)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_vec, y_train)
    hs_acc = accuracy_score(y_test, model.predict(X_test_vec))
    return tfidf, model, hs_acc


def screen_shipment_documents(tfidf, hs_model, esg_model, threshold=0.5):
    ship_path = os.path.join(HERE, "선적서류_샘플.csv")
    if not os.path.exists(ship_path):
        raise FileNotFoundError("선적서류_샘플.csv가 필요합니다.")

    ships = pd.read_csv(ship_path, encoding="utf-8-sig")
    ships["상품설명"] = ships["상품영문명"].astype(str) + " " + ships["사양"].astype(str)

    # HS Code 추천
    vec = tfidf.transform(ships["상품설명"])
    ships["예측_HS코드"] = hs_model.predict(vec)

    # ESG 위험 확률
    X_esg = ships[["협력사_환경위반건수", "협력사_노동평가점수"]]
    ships["위험_확률"] = esg_model.predict_proba(X_esg)[:, 1]
    ships["판정"] = ships["위험_확률"].apply(lambda p: "위험" if p >= threshold else "정상")

    # 워치리스트 스크리닝 (교육용 예시 — 실제 제재 리스트 아님)
    ships["워치리스트_관찰대상"] = ships["예측_HS코드"].apply(
        lambda code: "예" if code in EXAMPLE_WATCHLIST_HS_CODES else "-"
    )

    # 위험도 높은 순으로 정렬
    ships_sorted = ships.sort_values("위험_확률", ascending=False).reset_index(drop=True)

    print(f"[Step 8] 선적 서류 스크리닝 결과 (임계값 {threshold}, 위험도 높은 순 정렬)")
    print(ships_sorted[
        ["선적ID", "상품영문명", "예측_HS코드", "위험_확률", "판정", "워치리스트_관찰대상"]
    ].to_string(index=False))
    print()

    risky = ships_sorted[ships_sorted["판정"] == "위험"]
    print(f"스크리닝 요약: 전체 {len(ships_sorted)}건 중 위험 판정 {len(risky)}건")
    if len(risky) > 0:
        print("위험 건 목록:", ", ".join(risky["선적ID"].tolist()))
    print()
    print("※ 워치리스트_관찰대상은 교육용 예시 코드 목록 기준입니다.")
    print("   실제 스크리닝은 전략물자관리원·OFAC SDN 등 공식 리스트 연동이 필요합니다.")
    print()

    out_path = os.path.join(HERE, "screening_result.csv")
    ships_sorted.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"결과 저장 완료: {out_path}")
    return ships_sorted


def print_submission_summary(hs_acc, cm_df, precision, recall, threshold, risky_count, total_count):
    print("=" * 70)
    print("3.9 제출 양식에 채울 값 요약 (참고용 — 최종 검토 후 직접 기입)")
    print("=" * 70)
    print(f"- HS Code 예측 모델 1차 정확도: {hs_acc:.1%}")
    print(f"- ESG 위험도 예측 Confusion Matrix: {cm_df.to_dict()}")
    print(f"- 선택한 임계값: {threshold}")
    print(f"- 스크리닝에서 발견된 위험 건: {risky_count}건 / 전체 {total_count}건")
    print("- 신규 상품 예측 테스트 결과, 임계값 선택 근거, 인사이트는 실습자가 직접 작성")


def main():
    THRESHOLD = 0.5  # Step 7 토론 후 팀에서 정한 값으로 바꿔서 재실행해보세요.

    df = load_master_dataset()

    esg_model, X_test, y_test, proba_test = train_esg_model(df)
    cm_df, precision, recall = show_confusion_matrix(y_test, proba_test, threshold=THRESHOLD)
    compare_thresholds(y_test, proba_test)

    tfidf, hs_model, hs_acc = train_hs_model(df)
    result = screen_shipment_documents(tfidf, hs_model, esg_model, threshold=THRESHOLD)

    risky_count = (result["판정"] == "위험").sum()
    print_submission_summary(hs_acc, cm_df, precision, recall, THRESHOLD, risky_count, len(result))


if __name__ == "__main__":
    main()
