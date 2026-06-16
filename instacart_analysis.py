# ============================================================
# Instacart 고객 세분화 및 상품 추천 시스템
# 프로젝트명: 장바구니를 엿보다 - 고객 행동을 이해하는 상품 추천 시스템
# ============================================================

# ── 0. 라이브러리 임포트 ──────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from itertools import combinations
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import BoundaryNorm

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ── 1. 데이터 로드 ────────────────────────────────────────────
df_aisles      = pd.read_csv('aisles.csv')
df_departments = pd.read_csv('departments.csv')
df_products    = pd.read_csv('products.csv')
df_prior       = pd.read_csv('order_products__prior.csv')
df_train       = pd.read_csv('order_products__train.csv')
df_orders      = pd.read_csv('orders.csv')

# ── 2. 데이터 전처리 및 병합 ──────────────────────────────────
# products + aisles + departments 병합
df_products_aisles = pd.merge(df_products, df_aisles, on='aisle_id', how='left')
merged_df = pd.merge(df_products_aisles, df_departments, on='department_id', how='left')

# prior + train 병합
df_prior_train = pd.concat([df_prior, df_train], ignore_index=True)
df_prior_train = df_prior_train.sort_values(by='order_id').reset_index(drop=True)

# 전체 병합
df_merged = pd.merge(df_prior_train, merged_df, on='product_id', how='left')
df = pd.merge(df_merged, df_orders, on='order_id', how='inner')
df.drop(columns=['eval_set'], inplace=True)

print("데이터 로드 및 병합 완료:", df.shape)

# ── 3. 고객별 주요 지표 생성 (EDA - 장아현 파트) ─────────────
customer_orders = df.groupby('user_id')['order_number'].max().reset_index(name='total_orders')
customer_spent  = df.groupby('user_id')['unit_price'].sum().reset_index(name='total_spent') if 'unit_price' in df.columns else None
reorder_ratio   = df.groupby('user_id')['reordered'].mean().reset_index(name='reorder_rate')
avg_order_dow   = df.groupby('user_id')['order_dow'].mean().reset_index(name='avg_order_dow')
avg_order_hour  = df.groupby('user_id')['order_hour_of_day'].mean().reset_index(name='avg_order_hour')

# 지표 병합
dfs = [customer_orders, reorder_ratio, avg_order_dow, avg_order_hour]
if customer_spent is not None:
    dfs.insert(1, customer_spent)

customer_metrics = reduce(lambda left, right: pd.merge(left, right, on='user_id', how='inner'), dfs)
print("고객 지표 생성 완료:", customer_metrics.shape)

# ── 4. EDA 시각화 ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 총 주문 수 분포
axes[0, 0].hist(customer_metrics['total_orders'], bins=50, color='teal', edgecolor='white')
axes[0, 0].set_title('Distribution of Total Orders per Customer')
axes[0, 0].set_xlabel('Total Orders')
axes[0, 0].set_ylabel('Number of Customers')

# 재구매 비율 분포
axes[0, 1].hist(customer_metrics['reorder_rate'], bins=20, color='cadetblue', edgecolor='white')
axes[0, 1].set_title('Customer Reorder Rate Distribution')
axes[0, 1].set_xlabel('Reorder Rate')
axes[0, 1].set_ylabel('Number of Customers')

# 평균 주문 요일
dow_counts = customer_metrics.groupby(customer_metrics['avg_order_dow'].round())['avg_order_dow'].count()
axes[1, 0].bar(dow_counts.index, dow_counts.values, color='mediumseagreen')
axes[1, 0].set_title('Average Order Day of Week per Customer')
axes[1, 0].set_xlabel('Day of Week (0=Sun)')
axes[1, 0].set_ylabel('Number of Customers')

# 평균 주문 시간대
hour_counts = customer_metrics.groupby(customer_metrics['avg_order_hour'].round())['avg_order_hour'].count()
axes[1, 1].bar(hour_counts.index, hour_counts.values, color='limegreen')
axes[1, 1].set_title('Average Order Hour of Day per Customer')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Number of Customers')

plt.tight_layout()
plt.savefig('eda_customer_metrics.png', dpi=150)
plt.show()
print("EDA 시각화 저장 완료")

# ── 5. 상관관계 분석 ─────────────────────────────────────────
feature_cols = [c for c in ['total_orders', 'total_spent', 'reorder_rate', 'avg_order_dow', 'avg_order_hour']
                if c in customer_metrics.columns]
corr_matrix = customer_metrics[feature_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Customer Metrics')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()

# ── 6. 고객 세분화 - KMeans 클러스터링 ───────────────────────
X = customer_metrics[feature_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 최적 K 탐색 (Elbow + Silhouette)
K_range = range(2, 11)
inertia = []
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    score = silhouette_score(X_scaled, km.labels_, sample_size=10000, random_state=42)
    silhouette_scores.append(score)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(K_range, inertia, marker='o')
ax[0].set_title('Elbow Method')
ax[0].set_xlabel('K')
ax[0].set_ylabel('Inertia')

ax[1].plot(K_range, silhouette_scores, marker='o', color='green')
ax[1].set_title('Silhouette Score vs K')
ax[1].set_xlabel('K')
ax[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.savefig('kmeans_k_selection.png', dpi=150)
plt.show()

# 최적 K=5 로 최종 클러스터링
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

customer_metrics_clustered = customer_metrics.dropna(subset=feature_cols).copy()
customer_metrics_clustered['cluster'] = cluster_labels

# 클러스터별 특성 요약
cluster_summary = customer_metrics_clustered.groupby('cluster')[feature_cols].mean()
print("\n클러스터별 특성 평균:")
print(cluster_summary)

cluster_counts = customer_metrics_clustered['cluster'].value_counts().sort_index()
print("\n클러스터별 고객 수:")
print(cluster_counts)

# ── 7. PCA 시각화 ────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

cmap = plt.cm.get_cmap('tab10', optimal_k)
bounds = np.arange(optimal_k + 1) - 0.5
norm = BoundaryNorm(bounds, cmap.N)

centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(12, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=cluster_labels, cmap=cmap, norm=norm, s=10, alpha=0.6)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='red', s=200, marker='X', edgecolor='black', label='Centroids')
plt.title('PCA - KMeans Clustering')
plt.xlabel(f'PCA 1 ({explained_variance[0]:.1%})')
plt.ylabel(f'PCA 2 ({explained_variance[1]:.1%})')
cbar = plt.colorbar(scatter, ticks=np.arange(optimal_k))
cbar.set_label('Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_clustering.png', dpi=150)
plt.show()

# ── 8. 장바구니 연관 분석 (FP-Growth) ───────────────────────
print("\n연관 규칙 분석 시작...")

# aisle 단위로 분석 (메모리 효율)
order_aisle_list = df.groupby('order_id')['aisle'].apply(list)

te = TransactionEncoder()
te_ary = te.fit(order_aisle_list).transform(order_aisle_list)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df_encoded, min_support=0.02, use_colnames=True, max_len=2)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.2)

# 필터링
rules_filtered = rules[
    ((rules['confidence'] >= 0.4) & (rules['lift'] >= 1.2)) |
    ((rules['confidence'] >= 0.35) & (rules['lift'] >= 1.5))
].sort_values('lift', ascending=False)

print(f"도출된 연관 규칙 수: {len(rules_filtered)}")
print("\nTop 10 연관 규칙 (Lift 기준):")
print(rules_filtered[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string())

# 연관 규칙 시각화
plt.figure(figsize=(10, 6))
plt.scatter(rules_filtered['support'], rules_filtered['confidence'],
            c=rules_filtered['lift'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules: Support vs Confidence (color=Lift)')
plt.tight_layout()
plt.savefig('association_rules.png', dpi=150)
plt.show()

# ── 9. 클러스터별 해석 출력 ──────────────────────────────────
cluster_names = {
    0: '비활동적 소량 구매 고객',
    1: '매우 활동적인 고가치 충성 고객',
    2: '비활동적 대량 구매 저충성 고객',
    3: '활동적인 중량 구매 고충성 고객',
    4: '중빈도 초대량 구매 고객'
}

print("\n=== 클러스터별 고객 해석 ===")
for k, name in cluster_names.items():
    count = cluster_counts.get(k, 0)
    print(f"Cluster {k} ({name}): {count:,}명")

print("\n분석 완료. 생성된 파일: eda_customer_metrics.png, correlation_matrix.png, kmeans_k_selection.png, pca_clustering.png, association_rules.png")
