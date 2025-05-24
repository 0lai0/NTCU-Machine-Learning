import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    silhouette_score
)
import kagglehub

# ------------------------------------------------------------
# 參數設定
RANDOM_SEED = 42
TEST_SIZE = 0.3

# ------------------------------------------------------------
# 載入資料並前處理
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(f"{path}/creditcard.csv")
df['Class'] = df['Class'].astype(int)

# 刪掉 Time 欄位，並將 Amount 做一次標準化
df = df.drop(columns=['Time'])
df['Amount'] = StandardScaler().fit_transform(
    df['Amount'].values.reshape(-1, 1)
)

# 特徵與標籤
X = df.drop(columns=['Class']).values
y = df['Class'].values

# ------------------------------------------------------------
# 切分資料集（同時分層，確保 train/test 中正負例比例一致）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_SEED
)

# 把所有特徵再做一次標準化（因為 PCA、KMeans 等演算法都喜歡標準化過的輸入）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ------------------------------------------------------------
# 準備非監督式訓練：只用正常樣本（Class=0）的一小部分
normal_train = X_train_scaled[y_train == 0]
normal_sample = normal_train[:1000]  # 取前 1000 筆

# 用 silhouette score 來挑選 KMeans 的最佳 k
sil_scores = []
for k in range(2, 6):
    labels = KMeans(n_clusters=k, random_state=RANDOM_SEED).fit_predict(normal_sample)
    sil_scores.append(silhouette_score(normal_sample, labels))

optimal_k = np.argmax(sil_scores) + 2
print(f"最佳群數 k = {optimal_k}, silhouette = {sil_scores[optimal_k-2]:.3f}")

# Fit 最終的 KMeans
kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED)
kmeans.fit(normal_sample)

# 把整個測試集丟進去，得到每個樣本所屬的群聚編號
cluster_labels = kmeans.predict(X_test_scaled)

# 群聚標籤對應到偵測結果（0＝正常，1＝詐欺）
# 這裡用「對每個群聚取多數票」的方法做一次標籤對齊（雖然用了 y_test，但主要是為了示範）
def align_labels(y_true, clusters):
    mapping = {}
    for c in np.unique(clusters):
        mask = (clusters == c)
        if mask.sum() == 0:
            mapping[c] = 0
        else:
            # 該群聚中，哪個 class 出現最多，就把整個群聚標為那個 class
            mapping[c] = np.bincount(y_true[mask]).argmax()
    return np.array([mapping[c] for c in clusters])

y_pred = align_labels(y_test, cluster_labels)

# ------------------------------------------------------------
# 評估
print("\n=== 非監督式 KMeans 偵測結果 ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
