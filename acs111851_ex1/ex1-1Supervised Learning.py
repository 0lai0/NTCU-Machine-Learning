#################################監督式學習(random forest)###################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score,
    precision_recall_curve
)
from xgboost import XGBClassifier
import kagglehub

# 一般設定
RANDOM_SEED = 42
TEST_SIZE = 0.3

# 載入資料
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)

# 預處理
data.drop(['Time'], axis=1, inplace=True)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop(columns=['Class']).values
y = data['Class'].values

# 訓練/測試集切分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# 建立 XGBoost 模型（無 SMOTE）
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=200,  # 根據你資料比例微調。非越大越好
    random_state=RANDOM_SEED,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# 預測機率
y_probs = xgb_model.predict_proba(X_test)[:, 1]

# 自動尋找最佳 threshold（以 F1 score 為優先）
#precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
#f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
#best_idx = np.argmax(f1_scores)
#best_threshold = thresholds[best_idx]

# 使用最佳門檻進行分類
y_pred = (y_probs > best_threshold).astype(int)

# 評估結果
print(f"\n✅ Best Threshold Found: {best_threshold:.3f}")
print(f"Precision: {precision[best_idx]:.4f}")
print(f"Recall:    {recall[best_idx]:.4f}")
print(f"F1 Score:  {f1_scores[best_idx]:.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_probs):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
