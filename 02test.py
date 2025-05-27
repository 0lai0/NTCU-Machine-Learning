import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 參數設定
RANDOM_SEED = 42
TEST_SIZE = 0.3

# 載入資料
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)

# 前處理
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

X = data.drop('Class', axis=1)
y = data['Class']

# 先拆分資料，避免資訊洩漏
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# 用 Isolation Forest 找異常
iso_forest = IsolationForest(contamination=0.017, random_state=RANDOM_SEED)  # contamination要根據資料中fraud比例調整
iso_forest.fit(X_train)

# 將異常分數轉成新特徵加入原資料
train_scores = iso_forest.decision_function(X_train)
test_scores = iso_forest.decision_function(X_test)

X_train['iso_score'] = train_scores
X_test['iso_score'] = test_scores

# 用 XGBoost 訓練分類器，帶入新特徵
# scale_pos_weight 調整對於不平衡資料的權重
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_SEED,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

# 預測與評估
y_pred = xgb.predict(X_test)

print("Evaluation Metrics:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
