import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# === 固定設定 ===
RANDOM_SEED = 42
TEST_SIZE = 0.3

# === 載入資料 ===
data = pd.read_csv("creditcard.csv")
data = data.drop(["Time"], axis=1)
data["Amount"] = StandardScaler().fit_transform(data["Amount"].values.reshape(-1, 1))

X = np.asarray(data.drop(columns=["Class"]))
y = data["Class"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# === PCA 降維 ===
pca = PCA(n_components=15, random_state=RANDOM_SEED)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# === Local Outlier Factor ===
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.0014,  # 可微調
    novelty=True  # 才能用在 test set 預測
)
lof.fit(X_train_pca)

preds = lof.predict(X_test_pca)
preds = (preds == -1).astype(int)  # 將 outliers 設為 1（fraud）

# === 評估 ===
def evaluation(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{model_name} Evaluation:")
    print("=" * 45)
    print("         Accuracy:", accuracy)
    print("  Precision Score:", precision)
    print("     Recall Score:", recall)
    print("         F1 Score:", f1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

evaluation(y_test, preds, model_name="Local Outlier Factor (LOF)")
