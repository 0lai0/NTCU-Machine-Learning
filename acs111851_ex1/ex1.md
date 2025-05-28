# 監督式學習 
一開始先採用隨機森林來做參數調整，但測試了多組數據與超參數設置後發現結果不盡理想  
難以全面超越原始數據結果  
所以後來改採用 XGBoost做參數設置與調整  
### 資料集分析
信用卡詐欺預測是一個 二元分類問題（fraud vs. non-fraud），且資料極度不平衡（fraud 約佔 0.17%）  
這類問題的挑戰可分為以下兩點:  
1.資料不平衡（少數類別重要）  
2.權重差異大，容易導致過擬合或忽略少數類別  
### 而 XGBoost 優點  
針對不平衡分類問題的強化支援：scale_pos_weight 可自訂正負樣本重要性  
梯度提升決策樹（GBDT）核心：具備優異的泛化能力  
可調超參數較多元：可控制模型複雜度、防止過擬合等等  
內建 early stopping 與 logloss 最佳化：適合處理詐欺偵測這類 precision/recall 驅動的任務  
### 模型與參數設定說明
共使用了以下參數去做調整  
xgb_model = XGBClassifier(  
    n_estimators=200,  
    max_depth=7,  
    learning_rate=0.1,  
    min_child_weight=1,  
    gamma=0.2,  
    subsample=0.9,  
    colsample_bytree=0.9,  
    scale_pos_weight=200,  
    random_state=RANDOM_SEED,  
    eval_metric='logloss'  
### 各核心參數說明與調整邏輯
參數	作用	調整邏輯  
n_estimators=200	樹的數量	初期設成中等值，避免過擬合  
max_depth=7	每棵樹最大深度	控制模型複雜度，7 通常是中等偏高值  
learning_rate=0.1	學習率	預設學習率，與樹數互補（越小要更多樹）  
min_child_weight=1	最小葉節點權重和	設為 1，允許模型擷取細微差異  
gamma=0.2	節點分裂的最小損失減益	抑制過度分裂（過擬合風險）  
subsample=0.9	每棵樹訓練時使用的樣本比例	降低過擬合  
colsample_bytree=0.9	每棵樹訓練時使用的特徵比例	降低特徵依賴性  
scale_pos_weight=200	權重平衡（樣本不平衡）	根據「負樣本數 / 正樣本數」設定，大約 = 85307 / 136 ≈ 627，但你採用 200 是經實驗調整的折衷值，為了 提高 recall 同時保持 high precision  
eval_metric='logloss'	評估指標	適合機率預測任務，與 precision/recall 不衝突  
### 門檻調整（閾值設定）
y_probs = xgb_model.predict_proba(X_test)[:, 1]  
取得預測機率後，因為threshold預設值為 = 0.5  
這邊選擇不直接使用預設門檻（0.5）分類  
而是透過以下邏輯：  
去找一組最佳的 threshold 來用    
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)    
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  
best_idx = np.argmax(f1_scores)  
best_threshold = 0.941  
最後得出的最佳結果為 {best_threshold = 0.941}  

根據預測機率與真實標籤計算多組閾值下的 Precision、Recall 值  
並計算每個 threshold 的 F1 score  
找到能達到最大 F1 score 的最佳門檻  

y_pred = (y_probs > best_threshold).astype(int)  
避免預設門檻導致高 precision 但 recall 太低，因為在詐欺偵測中 recall 通常更重要）  

### 結果評估指標說明  
各項指標代表的意義與分析  
Precision: 0.9576 表示每 100 筆判定為詐欺的交易，有 95 筆真的有問題（誤判率低）  
Recall: 0.8309 表示 100 筆詐欺中模型抓到了 83 筆（漏判率小）  
F1 Score: 0.8898 綜合考量 precision 與 recall  
Accuracy: 0.9997 雖然高，但在不平衡問題中參考價值低  
ROC AUC: 0.9871 模型區分正負樣本能力極佳  
### 我的結論與採用此方法的理由  
使用 XGBoost	為了在不平衡樣本下取得良好效果，且可調整 scale_pos_weight  
使用 precision-recall curve 找門檻	避免 0.5 預設值造成的 recall 偏低問題  
調參以提升泛化	調整 gamma、subsample、colsample_bytree 等避免過擬合  
選擇 logloss 評估指標	能平衡正負類分類，適合詐欺預測場景  
這邊沒有採用 SMOTE	保持資料真實性，僅透過類別權重調整處理不平衡問題  
原始調整Random Forest時，也有採用過 SMOTE 來做資料平衡但未達到要求的數據結果  
這邊改採 XGBoost 做監督式學習後就未使用 SMOTE 來做資料平衡了，此方式也可以保持資料真實性  

# 非監督式學習
實驗說明：K-Means + Mahalanobis 距離法
初始方法與實驗動機
在監督式學習難以取得足夠標註資料的場景下，本次實驗採用 非監督式學習（Unsupervised Learning） 進行詐欺交易偵測。

### 邏輯和實驗方法

透過「正常樣本」進行聚類，建立資料分布輪廓
以 Mahalanobis 距離 衡量測試樣本與聚類中心的偏離程度
利用距離大小判定樣本是否為潛在異常（詐欺）

### 此方法之優勢包括：

無需大量標記資料，適合樣本極度不平衡的任務
專注於異常檢測，更貼合詐欺行為的本質
Mahalanobis 距離考量共變異，對高維資料更敏感

### 資料集與前處理
使用資料集：creditcard.csv（Kaggle 信用卡詐欺資料）
處理步驟：
移除 Time 欄位
標準化 Amount 欄位與其餘 V1 ~ V28 特徵
切分資料為訓練集與測試集（Stratified 方式，保持類別比例）
模型設計與邏輯架構
Step 1：降維（PCA）
採用 主成分分析（PCA） 降維
測試保留變異量：0.96、0.97、0.98
Step 2：樣本過濾
僅取正常樣本（label = 0）建立聚類模型
使用 z-score 過濾 移除極端值（|z| < 1.8）
Step 3：聚類模型（K-Means）
聚類數 K 搜尋範圍：10 ~ 15
以 Silhouette Score 評估聚類效果
擇最佳 K 值進行 KMeans 聚類
Step 4：Mahalanobis 距離計算
對測試樣本，計算與所有聚類中心的 Mahalanobis 距離
取最小值作為樣本「正常性分數」
此距離值用於分類詐欺 vs 正常

### 閾值設定邏輯（Threshold Selection）
模型輸出為異常距離，非機率值，無法直接用 0.5 作為門檻
改以 分位數閾值（percentile threshold） 決定分類界線
測試區間：99.85% ~ 99.99%
評估指標：Precision / Recall / F1 Score
最終門檻：以最大 F1 score 對應的 threshold 為主

### 參數設定與搜尋組合
參數名稱	功能描述	嘗試值 / 調整依據
pca_var	PCA 保留變異量	[0.96, 0.97, 0.98]
sample_size	正常樣本訓練筆數	[1200, 1500, 1800, 2000]
k	聚類中心數	自動搜尋 [10~15]，取 Silhouette 最佳
threshold	Mahalanobis 距離閾值	[99.85% ~ 99.99%]（取 F1 最佳）

### 最佳組合：
 'pca_var': 0.97,
 'sample_size': 1500,
 'k': 14,
 'threshold': 1.3085
### 模型結果與評估指標
Precision: 0.9063 → 表示每 100 筆被判定為詐欺的交易中，有 90 筆是真的（誤報低）
Recall: 0.7838 → 表示所有詐欺樣本中有 78% 被偵測出來（漏判低）
F1 Score: 0.84 → Precision / Recall 平衡佳
Accuracy: 0.9994 → 雖高，但在極度不平衡問題中參考性較低

### 我的結論與實驗方法結果
 本方法不依賴監督學習，適合無標註資料或樣本極度不平衡場景
 Mahalanobis 距離考慮樣本共變異，能更有效偵測異常行為
 門檻選擇以 precision-recall 驅動，避免固定 0.5 帶來的偏誤
 保留原始資料分佈，不進行 SMOTE 或其他過採樣，維持資料真實性
 在未標記場景下仍能達到相當於監督學習模型的性能表現（F1 ≈ 0.84）


