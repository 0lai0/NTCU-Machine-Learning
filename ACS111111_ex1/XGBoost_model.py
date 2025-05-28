import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
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

########################## train model ################################

xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=8,
    scale_pos_weight=len(nonfraud) * 2 / len(fraud),  # 不平衡補償
    eval_metric="logloss",
    random_state=RANDOM_SEED,
)

xgb_model.fit(
    X_train,
    y_train,
)

scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="f1")
print("CV F1 scores:", scores)
print("Average F1:", scores.mean())

y_proba = xgb_model.predict_proba(X_test)[:, 1]

# 嘗試不同 threshold
for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
    y_pred_thresh = (y_proba > threshold).astype(int)
    print(classification_report(y_test, y_pred_thresh))

y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

f1_train = f1_score(y_train, y_train_pred)
f1_test = f1_score(y_test, y_test_pred)

print("Train F1:", f1_train)
print("Test  F1:", f1_test)
