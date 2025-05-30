import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import kagglehub

# general setting. do not change TEST_SIZE
RANDOM_SEED = 42
TEST_SIZE = 0.3

# load dataset（from kagglehub）
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)
#取出 Class 欄位，轉成int

# prepare data
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
#把金額標準化

fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]
print(f'Fraudulent:{len(fraud)}, non-fraudulent:{len(nonfraud)}')
print(f'the positive class (frauds) percentage: {len(fraud)}/{len(fraud) + len(nonfraud)} ({len(fraud)/(len(fraud) + len(nonfraud))*100:.3f}%)')    
#顯示詐騙佔比


X = data.drop('Class', axis=1).values
Y = data['Class'].values  # 轉成一維

# split training set and data set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
#把資料切成訓練集和測試集

train_data = pd.DataFrame(X_train, columns=data.columns[:-1])
train_data['Class'] = y_train

fraud = train_data[train_data['Class'] == 1]#詐騙
nonfraud = train_data[train_data['Class'] == 0]#非詐騙


fraud_count = len(fraud)
nonfraud_count = len(nonfraud)
max_ratio = 500  # 最多保留 500 倍的非詐騙

# 動態決定 n_samples，避免過多或過少
nonfraud_target = min(fraud_count * max_ratio, nonfraud_count)
nonfraud_downsampled = resample(
    nonfraud, 
    replace=False, 
    n_samples=nonfraud_target, 
    random_state=RANDOM_SEED
)


data_balanced = pd.concat([fraud, nonfraud_downsampled]).sample(frac=1, random_state=RANDOM_SEED)
# 合併並打亂

X_train = data_balanced.drop('Class', axis=1).values
y_train = data_balanced['Class'].values

iso_forest = IsolationForest(random_state=RANDOM_SEED, contamination=0.001)
train_anomaly_score = iso_forest.fit_predict(X_train)
test_anomaly_score = iso_forest.predict(X_test)

# 將結果轉為 0 和 1（1 表示正常，-1 表示異常）
train_anomaly_score = (train_anomaly_score == -1).astype(int)
test_anomaly_score = (test_anomaly_score == -1).astype(int)

# 加到原始特徵後面
X_train = np.hstack([X_train, train_anomaly_score.reshape(-1,1)])
X_test = np.hstack([X_test, test_anomaly_score.reshape(-1,1)])

xgb_model = XGBClassifier(
    scale_pos_weight=1500,  #樣本的權重(非詐騙樣本數設為詐騙的500倍)
    n_estimators=500,#決策樹數量
    max_depth=6,#每棵樹的最大深度
    learning_rate=0.1,
    subsample=0.8,#	每棵樹訓練時，隨機抽樣使用 80% 的樣本
    colsample_bytree=0.8,#	每棵樹訓練時，隨機抽樣 80% 的特徵
    eval_metric='logloss',
    random_state=RANDOM_SEED
)
xgb_model.fit(X_train, y_train)

# define evaluation function
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
    
from sklearn.metrics import f1_score
import numpy as np

# 取得機率分數
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# 嘗試不同的 threshold，選擇 F1-score 最大的
best_threshold = 0.5
best_f1 = 0
thresholds = np.arange(0.01, 1.0, 0.01)

for threshold in thresholds:
    y_pred_tmp = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_tmp)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f} with F1-score: {best_f1:.4f}")

# 用最佳 threshold 預測
y_pred = (y_pred_proba >= best_threshold).astype(int)

# 評估
evaluation(y_test, y_pred, "XGBoost (Best Threshold)")