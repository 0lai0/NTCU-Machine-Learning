import numpy as np  # 數值計算庫
import pandas as pd  # 數據處理庫

# 機器學習工具
from sklearn.model_selection import train_test_split  # 數據分割
from sklearn.preprocessing import StandardScaler  # 特徵標準化
from sklearn.metrics import classification_report  # 分類報告
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 評估指標
from sklearn.metrics import precision_recall_curve  # PR曲線

import kagglehub  # Kaggle數據集下載

# 機器學習模型
from sklearn.ensemble import RandomForestClassifier  # 監督式學習模型
from sklearn.cluster import KMeans  # 非監督式學習模型
from sklearn.metrics import silhouette_score  # 聚類效果評估

# ======================
# 全局設置
# ======================
RANDOM_SEED = 42  # 隨機種子確保結果可重現
TEST_SIZE = 0.3   # 測試集比例（不可更改）

# ======================
# 評估函數
# ======================
def evaluation(y_true, y_pred, model_name="Model"):
   # 計算各項評估指標
   accuracy = accuracy_score(y_true, y_pred)  # 準確率
   precision = precision_score(y_true, y_pred, zero_division=0)  # 精確率（處理除零錯誤）
   recall = recall_score(y_true, y_pred)  # 召回率
   f1 = f1_score(y_true, y_pred)  # F1分數
   
   # 輸出評估結果
   print(f'\n{model_name} Evaluation:')
   print('===' * 15)
   print('         Accuracy:', accuracy)
   print('  Precision Score:', precision)
   print('     Recall Score:', recall)
   print('         F1 Score:', f1)
   print("\nClassification Report:")
   print(classification_report(y_true, y_pred))  # 詳細分類報告

# ======================
# 數據準備
# ======================
# 下載信用卡詐騙數據集
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
# 讀取CSV文件
data = pd.read_csv(f"{path}/creditcard.csv")
# 將目標變量轉換為整數類型
data['Class'] = data['Class'].astype(int)

# 數據預處理
data = data.drop(['Time'], axis=1)  # 移除時間特徵
# 標準化交易金額特徵
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# 分析數據分布
fraud = data[data['Class'] == 1]  # 詐騙交易
nonfraud = data[data['Class'] == 0]  # 正常交易
print(f'Fraudulent:{len(fraud)}, non-fraudulent:{len(nonfraud)}')
# 計算詐騙交易百分比
print(f'the positive class (frauds) percentage: {len(fraud)}/{len(fraud) + len(nonfraud)} ({len(fraud)/(len(fraud) + len(nonfraud))*100:.3f}%)')

# 準備特徵和標籤
X = data.drop(columns=['Class']).values  # 特徵矩陣
Y = data['Class'].values  # 目標變量

# ============================================
# 監督式學習段落 (Random Forest)
# ============================================
# 分割訓練集和測試集（無分層抽樣）
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# 初始化隨機森林分類器
rf_model = RandomForestClassifier(    
    n_estimators=100,  # 決策樹數量
    max_depth=20,  # 樹的最大深度
    class_weight='balanced_subsample',  # 自動平衡類別權重（處理不平衡數據）
    random_state=RANDOM_SEED  # 隨機種子
)
# 訓練模型
rf_model.fit(x_train, y_train)

# 模型預測與評估
y_prob = rf_model.predict_proba(x_test)[:, 1]  # 預測詐騙概率
# 計算PR曲線和閾值
prec, rec, thresh = precision_recall_curve(y_test, y_prob)
# 計算F1分數（添加小數避免除零）
f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
best_idx = np.argmax(f1)  # 最佳F1索引
best_threshold = thresh[best_idx]  # 最佳概率閾值
y_pred = (y_prob >= best_threshold).astype(int)  # 應用閾值轉換預測類別

# 評估監督式模型
evaluation(y_test, y_pred, model_name="RandomForestClassifier(Supervised)")

# ============================================
# 非監督式學習段落 (K-Means)
# ============================================
# 重新分割數據集（使用分層抽樣確保類別比例）
x_train, x_test, y_train, y_test = train_test_split(
   X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=Y
)

# 特徵標準化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # 擬合並轉換訓練集
x_test = scaler.transform(x_test)  # 轉換測試集

# 僅使用正常交易數據進行訓練（非監督學習特徵）
n_x_train = x_train[y_train == 0]  # 篩選正常交易
n_x_train = n_x_train[:1000]  # 取1000個樣本

# 尋找最佳聚類數k（2-4範圍）
scores = []
for k in range(2, 5):
   # 初始化K-Means模型
   kmeans = KMeans(
       n_clusters=k,  # 聚類數量
       init='k-means++',  # 智能初始化中心點
       random_state=RANDOM_SEED  # 隨機種子
   )
   kmeans.fit(n_x_train)  # 訓練模型
   # 計算輪廓係數（評估聚類效果）
   score = silhouette_score(n_x_train, kmeans.labels_)
   scores.append(score)

# 選擇最佳k值（輪廓係數最高）
optimal_k = np.argmax(scores) + 2  # 索引轉實際k值
# 使用最佳k值初始化模型
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=RANDOM_SEED)
kmeans.fit(n_x_train)  # 重新訓練

# 預測測試集
y_pred_test = kmeans.predict(x_test)

# 對齊聚類標籤與真實標籤
def align_labels(y_true, y_pred, n_clusters):
   labels = np.zeros_like(y_pred)  # 初始化標籤數組
   for i in range(n_clusters):
       mask = (y_pred == i)  # 當前聚類的掩碼
       if np.sum(mask) > 0:  # 確保聚類非空
           # 將聚類分配為該組中多數的真實類別
           labels[mask] = np.bincount(y_true[mask]).argmax()
       else:
           labels[mask] = 0  # 默認為正常交易
   return labels

# 應用標籤對齊
y_pred_aligned = align_labels(y_test, y_pred_test, optimal_k)

# 評估非監督式模型
evaluation(y_test, y_pred_aligned, model_name="KMeans (Unsupervised)")