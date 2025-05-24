import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import kagglehub

# 參數
RANDOM_SEED = 42
TEST_SIZE = 0.3

# 讀資料
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(f"{path}/creditcard.csv")
df['Class'] = df['Class'].astype(int)

# 前處理
df = df.drop(columns=['Time'])
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

# 特徵與標籤
X = df.drop(columns=['Class']).values
y = df['Class'].values

# 切分並標準化
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 訓練隨機森林
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_SEED,
    class_weight='balanced_subsample'  # 建議同時開啟平衡權重
)
rf_model.fit(X_train, y_train)

# 定義評估函式
def evaluation(y_true, y_pred, model_name="Model"):
    print(f'\n{model_name} Evaluation:')
    print('='*40)
    print('Accuracy:       ', accuracy_score(y_true, y_pred))
    print('Precision Score:', precision_score(y_true, y_pred))
    print('Recall Score:   ', recall_score(y_true, y_pred))
    print('F1 Score:       ', f1_score(y_true, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred))

# **這裡先定義 y_pred，再呼叫 evaluation()**
y_pred = rf_model.predict(X_test)
evaluation(y_test, y_pred, model_name="Random Forest")
