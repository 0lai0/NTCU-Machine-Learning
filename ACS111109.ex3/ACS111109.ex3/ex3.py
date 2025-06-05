import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import kagglehub

# 下載資料
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(f"{path}/creditcard.csv")

# 資料預處理
df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))
df = df.drop(["Time"], axis=1)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.001, random_state=42)
df["is_anomaly"] = iso_forest.fit_predict(df.drop(columns=["Class"]))
df["is_anomaly"] = df["is_anomaly"].map({1: 0, -1: 1})

# 特徵分開
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# XGBoost 強化版模型
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    learning_rate=0.01,
    max_depth=12,
    n_estimators=1000,
    subsample=0.95,
    colsample_bytree=0.95,
    scale_pos_weight=300,
    gamma=0.1,
    min_child_weight=0,
    random_state=42
)

# 模型訓練
xgb.fit(X_train, y_train)

# 預測與評估
y_pred = xgb.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
