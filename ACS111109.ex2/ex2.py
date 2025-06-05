# ex2.py - 混合模型：Isolation Forest + XGBoost
# 安裝所需套件（第一次執行時取消註解）
# pip install kagglehub xgboost pandas numpy scikit-learn

import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# 載入資料
print("載入資料中...")
path = kagglehub.dataset_download('mlg-ulb/creditcardfraud')
data = pd.read_csv(f"{path}/creditcard.csv")
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time'], axis=1)

# Isolation Forest 異常偵測
print("執行 Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
iso_pred = iso_forest.fit_predict(data.drop(columns=['Class']))
data['Anomaly'] = (iso_pred == -1).astype(int)

# 分割資料集
X = data.drop(columns=['Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# 防呆轉換
X_train = X_train.fillna(0).values
X_test = X_test.fillna(0).values
y_train = y_train.astype(int).values
y_test = y_test.astype(int).values

# XGBoost 模型訓練與預測
print("訓練 XGBoost 模型...")
model = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 顯示評估報告
print("\n=== 預測結果 ===")
print(classification_report(y_test, y_pred))
