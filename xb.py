import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import kagglehub
import xgboost as xgb

# general setting. do not change TEST_SIZE
RANDOM_SEED = 42
TEST_SIZE = 0.3

# load dataset（from kagglehub）
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)  # 0 for nonfraud, 1 for fraud

# prepare data
data = data.drop(['Time'], axis=1)  # 去除 Time 欄位
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))  # 標準化 Amount

fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]
print(f'Fraudulent: {len(fraud)}, non-fraudulent: {len(nonfraud)}')
print(f'The positive class (frauds) percentage: {len(fraud)}/{len(fraud) + len(nonfraud)} ({len(fraud)/(len(fraud) + len(nonfraud))*100:.3f}%)')

# split data
X = np.asarray(data.iloc[:, ~data.columns.isin(['Class'])])
Y = data['Class'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# 計算 class imbalance 權重
scale_weight = len(nonfraud) / len(fraud)


# build XGBoost model (with GPU)
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_SEED
)

# fit model
xgb_model.fit(X_train, y_train)

# 評估函式
def evaluation(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
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

# 預測與評估
y_pred = xgb_model.predict(X_test)
evaluation(y_test, y_pred, model_name="XGBoost (GPU)")
