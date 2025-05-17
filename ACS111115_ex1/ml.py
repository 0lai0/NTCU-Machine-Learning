import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import kagglehub

# 一般參數設定
RANDOM_SEED = 42     # 固定隨機種子，確保每次結果一致
TEST_SIZE = 0.3      # 測試集佔比 30%

# 載入 Kaggle 資料集（信用卡詐騙資料）
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")

# 將 'Class' 欄位轉為整數（0：正常，1：詐騙）
data['Class'] = data['Class'].astype(int)

# 移除 'Time' 欄位（通常無預測意義）
data = data.drop(['Time'], axis=1)

# 對 'Amount' 欄位做標準化處理（均值為 0，標準差為 1）
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# 顯示正負樣本數量與比例
fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]
print(f'Fraudulent:{len(fraud)}, non-fraudulent:{len(nonfraud)}')
print(f'the positive class (frauds) percentage: {len(fraud)}/{len(fraud) + len(nonfraud)} ({len(fraud)/(len(fraud) + len(nonfraud))*100:.3f}%)')    

# 特徵與標籤準備
# X：所有除了 'Class' 的欄位（轉成 NumPy 陣列）
X = np.asarray(data.iloc[:, ~data.columns.isin(['Class'])])

# Y：'Class' 欄位（轉成 NumPy 陣列）
Y = data['Class'].to_numpy()

# 訓練集 / 測試集 切分
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# 建立 Random Forest 分類模型
rf_model = RandomForestClassifier(
    n_estimators=85,                 # 使用 85 棵樹進行投票
    class_weight='balanced_subsample',  # 每棵樹針對類別權重調整，幫助學習詐騙樣本
    min_samples_leaf=2,             # 每個葉節點最少 2 筆樣本（可防過擬合）
    min_samples_split=3,            # 每個節點至少 3 筆樣本才能再分裂
    oob_score=True,                 # 啟用 Out-of-Bag 估計
    max_features='sqrt',            # 每棵樹每次分裂只考慮 sqrt(n) 個特徵（通常可提升泛化）
    max_depth=25,                   # 限制樹的最大深度，避免太深過擬合
    random_state=RANDOM_SEED        # 固定隨機種子
)

# 訓練模型
rf_model.fit(X_train, y_train)


# 定義評估函式
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

# 模型預測與結果輸出
y_pred = rf_model.predict(X_test)

# 輸出完整報告
print(classification_report(y_test, y_pred))
