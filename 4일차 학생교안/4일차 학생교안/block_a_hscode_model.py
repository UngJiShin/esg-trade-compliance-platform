# -*- coding: utf-8 -*-
"""
[블록 A] 자사 영문 상품명·사양 기반 HS Code 추천 모델 구현 — 파이썬 버전
======================================================================
Orange3(노코드)로 하던 Step 1~4를 파이썬 코드로 그대로 구현한 스크립트.
Orange3 위젯 하나하나가 아래 코드의 어느 부분에 대응하는지 주석으로 표시했다.

■ 실행 방법
    python block_a_hscode_model.py

■ 필요한 파일
    master_dataset.csv (또는 master_dataset.xlsx) — 이 스크립트와 같은 폴더에 둘 것
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Step 1. 실습 환경 준비 및 학습 데이터 확인
#   [Orange 위젯: File] 마스터 데이터셋 불러오기
# ============================================================
def load_master_dataset():
    csv_path = os.path.join(HERE, "master_dataset.csv")
    xlsx_path = os.path.join(HERE, "master_dataset.xlsx")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        source = "master_dataset.csv"
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
        source = "master_dataset.xlsx"
    else:
        raise FileNotFoundError(
            "master_dataset.csv 또는 master_dataset.xlsx 파일을 이 스크립트와 같은 폴더에 두세요."
        )

    required_cols = ["상품영문명", "사양", "HS코드_정답"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    print(f"[Step 1] 데이터 로드 완료 — {source} / 총 {len(df)}행")
    print(df[required_cols].head(3).to_string(index=False))
    print()
    return df


# ============================================================
# Step 1-1. 자사 상품 데이터 매칭 및 결합
#   ※ 이미 master_dataset.csv 안에서 상품영문명·사양·HS코드_정답이
#     한 행(=상품 1개)으로 결합돼 있으므로, 이 스크립트에서는
#     결합 결과(구조)가 맞는지 확인만 한다.
# ============================================================
def check_structure(df):
    ok = all(c in df.columns for c in ["상품영문명", "사양", "HS코드_정답"])
    print(f"[Step 1-1] 데이터 구조 확인 — 행=상품 1개, 열=영문명·사양·HS Code 형태: {'정상' if ok else '비정상'}")
    print()


# ============================================================
# Step 2. 텍스트 데이터 인코딩
#   [Orange 위젯: Preprocess Text] 상품영문명 + 사양 결합·토큰화
#   [Orange 위젯: Bag of Words]     TF-IDF로 숫자 벡터화
# ============================================================
def encode_text(df):
    df = df.copy()
    df["상품설명"] = df["상품영문명"].astype(str) + " " + df["사양"].astype(str)

    y = df["HS코드_정답"].astype(str)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["상품설명"], y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train_text)
    X_test_vec = tfidf.transform(X_test_text)

    print(f"[Step 2] 텍스트 인코딩 완료 — 학습 {X_train_vec.shape[0]}건 / 테스트 {X_test_vec.shape[0]}건, "
          f"벡터 차원 {X_train_vec.shape[1]}")
    print()
    return tfidf, X_train_vec, X_test_vec, y_train, y_test


# ============================================================
# Step 3. HS Code 분류 모델 학습
#   [Orange 위젯: Tree / kNN]  분류 모델 학습 (여기서는 RandomForest)
#   [Orange 위젯: Test & Score] 테스트셋 정확도 확인
# ============================================================
def train_and_evaluate(X_train_vec, y_train, X_test_vec, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"[Step 3] HS Code 분류 모델 학습 완료 — 테스트 정확도: {acc:.1%}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return model


# ============================================================
# Step 4. 신규 상품 HS Code 예측 테스트
#   [Orange 위젯: Predictions] 학습에 없던 신규 상품 3~5건 예측
# ============================================================
def predict_new_products(tfidf, model):
    new_products = pd.DataFrame({
        "상품영문명": [
            "Cotton Graphic T-Shirt",
            "Stainless Steel Pipe Fitting",
            "Recycled Aluminum Wire",
            "LED Ceiling Panel Light",
            "Vegan Leather Sneaker",
        ],
        "사양": [
            "cotton-polyester blend, short sleeve, export packaging standard",
            "stainless steel, threaded end, export packaging standard",
            "recycled aluminum, drawn wire, export packaging standard",
            "polycarbonate diffuser, energy-efficient, export packaging standard",
            "synthetic leather sole, cushioned insole, export packaging standard",
        ],
    })
    new_products["상품설명"] = new_products["상품영문명"] + " " + new_products["사양"]
    vec = tfidf.transform(new_products["상품설명"])
    new_products["예측_HS코드"] = model.predict(vec)

    print("[Step 4] 신규 상품 5건 HS Code 예측 결과")
    print(new_products[["상품영문명", "예측_HS코드"]].to_string(index=False))
    print()
    print("체크리스트: 위 예측 HS Code를 실제 관세사 자문 결과(또는 강사 제공 정답)와")
    print("비교해 일치 여부를 별도로 기록하세요 (Step 4 실습 가이드 참고).")
    return new_products


def main():
    df = load_master_dataset()
    check_structure(df)
    tfidf, X_train_vec, X_test_vec, y_train, y_test = encode_text(df)
    model = train_and_evaluate(X_train_vec, y_train, X_test_vec, y_test)
    predict_new_products(tfidf, model)


if __name__ == "__main__":
    main()
