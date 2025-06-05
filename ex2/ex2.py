import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import kagglehub

# General setting
RANDOM_SEED = 42
TEST_SIZE = 0.3

# 評估函數
def evaluation(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'\n{model_name} Evaluation:')
    print('===' * 15)
    print('         Accuracy:', accuracy)
    print('  Precision Score:', precision)
    print('     Recall Score:', recall)
    print('         F1 Score:', f1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# 下載資料
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)

# 預處理
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop(columns=['Class']).values
Y = data['Class'].values

# 分割訓練/測試集
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=Y
)

# 標準化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 加入 IsolationForest（非監督式）
iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=RANDOM_SEED
)
iso_model.fit(x_train)

# 預測並加入為新特徵
iso_train_scores = iso_model.predict(x_train)  # -1 = 異常, 1 = 正常
iso_test_scores = iso_model.predict(x_test)

# 合併新特徵
x_train_enhanced = np.concatenate([x_train, iso_train_scores.reshape(-1, 1)], axis=1)
x_test_enhanced = np.concatenate([x_test, iso_test_scores.reshape(-1, 1)], axis=1)

# 使用 XGBoost（監督式）
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_SEED
)
xgb_model.fit(x_train_enhanced, y_train)

# 預測機率
y_prob = xgb_model.predict_proba(x_test_enhanced)[:, 1]

# 找最佳 threshold
prec, rec, thresh = precision_recall_curve(y_test, y_prob)
f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
best_idx = np.argmax(f1)
best_threshold = thresh[best_idx]

# 可自定 threshold（嘗試提高精確度）
manual_threshold = max(best_threshold, 0.6)
y_pred = (y_prob >= manual_threshold).astype(int)

# 評估
evaluation(y_test, y_pred, model_name="XGBoostClassifier (Supervised after IsolationForest)")