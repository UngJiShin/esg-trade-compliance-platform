# -*- coding: utf-8 -*-
"""
[3일차 실습] HS Code 자동 추천 + ESG 위험도 예측 파이썬 스크립트
------------------------------------------------------------------
마스터 데이터셋(master_dataset.csv)을 이용해 두 가지를 예측한다.

  1) HS Code 추천 모델
     - 상품영문명 + 사양 텍스트를 TF-IDF로 벡터화
     - RandomForest 분류 모델로 HS Code를 학습
     - 학습에 쓰지 않은 신규 상품 5건의 HS Code를 예측

  2) ESG 위험도 예측 모델
     - 협력사 환경위반건수·노동평가점수를 특징으로
     - 리스크 정답(정상/위험)을 이진분류 모델로 학습
     - Confusion Matrix, Precision, Recall 출력
     - 분류 임계값(threshold) 0.7 / 0.5 / 0.3 비교표 출력

각 단계 주석에 '이 코드가 Orange3의 어떤 위젯 역할을 하는지' 표시해 두었다.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
)

pd.set_option("display.unicode.east_asian_width", True)

# ============================================================
# [Orange 위젯: File] 마스터 데이터셋 불러오기
# ============================================================
DATA_PATH = "master_dataset.csv"
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

print("=" * 70)
print("1. 마스터 데이터셋 로드")
print("=" * 70)
print(f"전체 행 수: {len(df)}")
print(f"컬럼: {list(df.columns)}")
print(df.head(3).to_string(index=False))
print()


# ============================================================
# PART 1. HS Code 추천 모델
# ============================================================
print("=" * 70)
print("2. HS Code 추천 모델 학습")
print("=" * 70)

# [Orange 위젯: Preprocess Text] 상품영문명 + 사양을 하나의 텍스트로 결합
df["상품설명"] = df["상품영문명"] + " " + df["사양"]

X_text = df["상품설명"]
y_hs = df["HS코드_정답"].astype(str)

X_train_text, X_test_text, y_train_hs, y_test_hs = train_test_split(
    X_text, y_hs, test_size=0.2, random_state=42, stratify=y_hs
)

# [Orange 위젯: Bag of Words] TF-IDF로 텍스트를 숫자 벡터로 변환
tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train_text)
X_test_vec = tfidf.transform(X_test_text)

# [Orange 위젯: Tree / kNN] 분류 모델 학습 (여기서는 RandomForest 사용)
#   - kNN으로 바꾸려면: from sklearn.neighbors import KNeighborsClassifier
#                       hs_model = KNeighborsClassifier(n_neighbors=5)
hs_model = RandomForestClassifier(n_estimators=200, random_state=42)
hs_model.fit(X_train_vec, y_train_hs)

# [Orange 위젯: Test & Score] 테스트셋으로 1차 정확도 확인
y_pred_hs = hs_model.predict(X_test_vec)
hs_accuracy = accuracy_score(y_test_hs, y_pred_hs)
print(f"[Test & Score] HS Code 예측 정확도: {hs_accuracy:.1%}  (테스트 {len(y_test_hs)}건)")
print()
print(classification_report(y_test_hs, y_pred_hs, zero_division=0))

# [Orange 위젯: Predictions] 학습에 쓰지 않은 신규 상품 5건 예측
new_products = pd.DataFrame({
    "상품영문명": [
        "Cotton Graphic T-Shirt",
        "Stainless Steel Pipe Fitting",
        "Recycled Aluminum Wire",
        "LED Ceiling Panel Light",
        "Vegan Leather Sneaker",
    ],
    "사양": [
        "cotton-polyester blend, short sleeve, size 10, export packaging standard",
        "stainless steel, threaded end, size 20, export packaging standard",
        "recycled aluminum, drawn wire, size 30, export packaging standard",
        "polycarbonate diffuser, energy-efficient, size 15, export packaging standard",
        "synthetic leather sole, cushioned insole, size 9, export packaging standard",
    ],
})
new_products["상품설명"] = new_products["상품영문명"] + " " + new_products["사양"]
new_vec = tfidf.transform(new_products["상품설명"])
new_products["예측_HS코드"] = hs_model.predict(new_vec)

print("[Predictions] 신규 상품 5건 HS Code 예측 결과")
print(new_products[["상품영문명", "예측_HS코드"]].to_string(index=False))
print()


# ============================================================
# PART 2. ESG 위험도 예측 모델
# ============================================================
print("=" * 70)
print("3. ESG 위험도 예측 모델 학습")
print("=" * 70)

# [Orange 위젯: File→선택] 협력사 환경/노동 지표를 특징(feature)으로 사용
X_esg = df[["협력사_환경위반건수", "협력사_노동평가점수"]]
y_esg = (df["리스크_정답"] == "위험").astype(int)  # 위험=1, 정상=0

X_train_esg, X_test_esg, y_train_esg, y_test_esg = train_test_split(
    X_esg, y_esg, test_size=0.2, random_state=42, stratify=y_esg
)

# [Orange 위젯: Logistic Regression] 이진분류 모델 학습
esg_model = LogisticRegression(random_state=42)
esg_model.fit(X_train_esg, y_train_esg)

# [Orange 위젯: Test & Score / Confusion Matrix] 기본 임계값(0.5) 기준 평가
y_pred_esg = esg_model.predict(X_test_esg)
cm = confusion_matrix(y_test_esg, y_pred_esg)
precision = precision_score(y_test_esg, y_pred_esg)
recall = recall_score(y_test_esg, y_pred_esg)

print("[Confusion Matrix] (행=실제, 열=예측 / 순서: 정상=0, 위험=1)")
print(pd.DataFrame(cm, index=["실제_정상", "실제_위험"], columns=["예측_정상", "예측_위험"]))
print()
print(f"[Test & Score] Precision: {precision:.3f} / Recall: {recall:.3f}")
print()

# [Orange 위젯: Predictions + 임계값 슬라이더] 임계값(threshold) 0.7/0.5/0.3 비교
proba_esg = esg_model.predict_proba(X_test_esg)[:, 1]  # '위험'일 확률

threshold_rows = []
for th in [0.7, 0.5, 0.3]:
    y_pred_th = (proba_esg >= th).astype(int)
    p = precision_score(y_test_esg, y_pred_th, zero_division=0)
    r = recall_score(y_test_esg, y_pred_th, zero_division=0)
    threshold_rows.append({"임계값(threshold)": th, "Precision": round(p, 3), "Recall": round(r, 3)})

threshold_df = pd.DataFrame(threshold_rows)
print("[임계값별 Precision·Recall 비교표]")
print(threshold_df.to_string(index=False))
print()
print("해석: 임계값을 낮출수록(0.3) '위험'으로 더 많이 잡아내 Recall은 오르지만,")
print("      정상 협력사를 위험으로 오판하는 경우도 늘어 Precision은 떨어지는 경향을 보인다.")
