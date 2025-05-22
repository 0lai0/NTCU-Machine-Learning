import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score

# 載入資料
data = pd.read_csv("creditcard.csv")
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time'], axis=1)

# 建立 IsolationForest 模型（無監督）
iso_forest = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
iso_scores = iso_forest.fit_predict(data.drop(columns=['Class']))  # -1 代表異常，1 代表正常
anomaly_score = iso_forest.decision_function(data.drop(columns=['Class']))  # 分數越小越可能異常

# 加入為新特徵
data['isolation_label'] = (iso_scores == -1).astype(int)
data['anomaly_score'] = anomaly_score

# 分離特徵與標籤
X = data.drop(columns=['Class'])
y = data['Class']

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立 XGBoost 模型
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 預測
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# 評估
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"AUPRC (Average Precision): {average_precision_score(y_test, y_prob):.4f}")
