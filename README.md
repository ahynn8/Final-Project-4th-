# 🛒 Instacart 고객 세분화 및 상품 추천 시스템

> "장바구니를 엿보다 : 고객 행동을 이해하는 상품 추천 시스템"

## 📌 프로젝트 개요

Instacart의 실제 장바구니 데이터를 활용하여 고객 행동 패턴을 분석하고, 고객 세분화 기반 맞춤형 상품 추천 전략을 설계한 프로젝트입니다.

- **기간**: 2025.04.18 ~ 2025.05.14
- **데이터**: Kaggle - Instacart Online Grocery Basket Analysis Dataset (3백만건+) + 직접 다운로드 후 실행
- **팀**: Build-up (5인)

---

## 🔍 분석 흐름

```
데이터 로드 및 전처리
    ↓
고객별 주요 지표 생성 (EDA)
    ↓
상관관계 분석
    ↓
KMeans 클러스터링 (고객 세분화)
    ↓
PCA 시각화
    ↓
FP-Growth 연관 규칙 분석 (장바구니 분석)
    ↓
클러스터별 마케팅 전략 도출
```

---

## 📂 데이터셋 구성

| 파일명 | 설명 |
|--------|------|
| `orders.csv` | 주문 정보 (user_id, 요일, 시간대 등) |
| `order_products__prior.csv` | 과거 주문 상품 내역 |
| `order_products__train.csv` | 학습용 주문 데이터 |
| `products.csv` | 상품 목록 |
| `aisles.csv` | 상품 소분류 |
| `departments.csv` | 상품 대분류 |

---

## 🛠 사용 기술

- **언어**: Python
- **분석**: Pandas, NumPy, Scikit-learn
- **시각화**: Matplotlib, Seaborn, Tableau
- **ML**: KMeans, PCA, FP-Growth, Apriori
- **환경**: Google Colab, Jupyter Notebook

---

## 👥 고객 세분화 결과 (K=5)

| 클러스터 | 고객 유형 | 특징 |
|---------|----------|------|
| Cluster 0 | 비활동적 소량 구매 고객 | 낮은 주문 빈도, 작은 장바구니 |
| Cluster 1 | 고가치 충성 고객 | 높은 주문 빈도, 높은 재구매율 |
| Cluster 2 | 비활동적 대량 구매 저충성 고객 | 낮은 빈도, 낮은 재구매율 |
| Cluster 3 | 활동적 고충성 고객 | 중간 빈도, 높은 재구매율 |
| Cluster 4 | 중빈도 초대량 구매 고객 | 가장 큰 장바구니, 높은 다양성 |

---

## 🔗 연관 규칙 분석 주요 결과

- 유기농 과일/채소 간 강한 연관성 (Lift > 2.5)
- `Organic Raspberries → Organic Strawberries` : Lift = 2.95 (최고)
- 채소-과일 세트 할인 전략 유효

---

## 🚀 실행 방법

```bash
# 필요 라이브러리 설치
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend

# 분석 실행 (데이터 파일이 같은 경로에 있어야 함)
python instacart_analysis.py
```

---

## 📊 산출물

- `eda_customer_metrics.png` : 고객 지표 분포 시각화
- `correlation_matrix.png` : 지표 간 상관관계 히트맵
- `kmeans_k_selection.png` : 최적 K 선정 (Elbow + Silhouette)
- `pca_clustering.png` : PCA 기반 클러스터 시각화
- `association_rules.png` : 연관 규칙 분포 시각화
