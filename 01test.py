import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import kagglehub

# 固定設定
RANDOM_SEED = 42
TEST_SIZE = 0.3

# 載入資料集
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)

# 資料預處理
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# 欠抽樣讓資料平衡
fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]
nonfraud_downsampled = resample(nonfraud, replace=False, n_samples=len(fraud), random_state=RANDOM_SEED)
balanced_data = pd.concat([fraud, nonfraud_downsampled])

# 切分特徵與標籤
X = balanced_data.drop("Class", axis=1).values
Y = balanced_data["Class"].values

# 資料標準化（全部欄位）
X = StandardScaler().fit_transform(X)

# 切分訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# 建立 Random Forest 模型
rf_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
rf_model.fit(X_train, y_train)

# 預測（機率值）
y_scores = rf_model.predict_proba(X_test)[:, 1]

# 評估函式
def evaluation(y_true, y_prob, threshold=0.5, model_name="Model"):
    y_pred = (y_prob >= threshold).astype(int)
    
    print(f'\n{model_name} Evaluation:')
    print('=' * 40)
    print(f'Accuracy : {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'Recall   : {recall_score(y_true, y_pred):.4f}')
    print(f'F1 Score : {f1_score(y_true, y_pred):.4f}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred))

# 呼叫評估函式（可調整 threshold）
evaluation(y_test, y_scores, threshold=0.3, model_name="Random Forest (Balanced)")
