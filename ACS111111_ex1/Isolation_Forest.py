import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import kagglehub

# general setting. do not change TEST_SIZE
RANDOM_SEED = 42
TEST_SIZE = 0.3

################################ 資料處理 ####################################

# load dataset（from kagglehub）
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
# 0 for nonfraud 1 for fraud
data["Class"] = data["Class"].astype(int)

# prepare data
data = data.drop(["Time"], axis=1)  # 去除 time
data["Amount"] = StandardScaler().fit_transform(
    data["Amount"].values.reshape(-1, 1)
)  # 標準化

# 計算詐騙和正常交易的數量 (資料集極度不平衡)
fraud = data[data["Class"] == 1]
nonfraud = data[data["Class"] == 0]
print(f"Fraudulent:{len(fraud)}, non-fraudulent:{len(nonfraud)}")
print(
    f"the positive class (frauds) percentage: {len(fraud)}/{len(fraud) + len(nonfraud)} ({len(fraud)/(len(fraud) + len(nonfraud))*100:.3f}%)"
)

# 選擇非class的值轉成numpy array 且如果原本就是np array時不複製
X = np.asarray(data.iloc[:, ~data.columns.isin(["Class"])])
# 最佳化為 Pandas → NumPy 的不複製轉換
Y = data["Class"].to_numpy()

# split training set and data set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

pca = PCA(n_components=17, random_state=RANDOM_SEED)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

########################## train model ################################


def evaluation(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{model_name} Evaluation:")
    print("===" * 15)
    print("         Accuracy:", accuracy)
    print("  Precision Score:", precision)
    print("     Recall Score:", recall)
    print("         F1 Score:", f1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


for c in [0.0013,0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019]:
    iso_model = IsolationForest(
        n_estimators=62,
        contamination=c,
        max_samples=1.0,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    iso_model.fit(X_train_pca)
    preds = (iso_model.predict(X_test_pca) == -1).astype(int)
    evaluation(y_test, preds, model_name=f"IsolationForest (contamination={c:.4f})")
