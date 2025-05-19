import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import kagglehub

# 一般參數
RANDOM_SEED = 42
TEST_SIZE = 0.3

# 載入資料集
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)
data.drop('Time', axis=1, inplace=True)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# 顯示類別比例
fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]
print(f'Fraudulent:{len(fraud)}, non-fraudulent:{len(nonfraud)}')
print(f'the positive class (frauds) percentage: {len(fraud)/(len(fraud)+len(nonfraud))*100:.3f}%')

# 特徵與標籤
X = data.drop(columns=['Class']).to_numpy()
Y = data['Class'].to_numpy()

# 切分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# 標準化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 建立 XGBoost 模型
xgb_model = XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=200,
    subsample=0.8,
    scale_pos_weight=2.5,           
    eval_metric='logloss',
    tree_method='gpu_hist',          
    predictor='gpu_predictor',
    random_state=RANDOM_SEED
)

# 訓練模型
xgb_model.fit(X_train_std, y_train)

# 模型機率預測
y_prob = xgb_model.predict_proba(X_test_std)[:, 1]

threshold = 0.7671
y_pred_custom = (y_prob > threshold).astype(int)

# 分類報告
print(classification_report(y_test, y_pred_custom))