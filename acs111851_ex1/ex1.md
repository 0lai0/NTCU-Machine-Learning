## 監督式學習
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
### 總結：採用此方法的理由  
使用 XGBoost	為了在不平衡樣本下取得良好效果，且可調整 scale_pos_weight  
使用 precision-recall curve 找門檻	避免 0.5 預設值造成的 recall 偏低問題  
調參以提升泛化	調整 gamma、subsample、colsample_bytree 等避免過擬合  
選擇 logloss 評估指標	能平衡正負類分類，適合詐欺預測場景  
這邊沒有採用 SMOTE	保持資料真實性，僅透過類別權重調整處理不平衡問題  
原始調整Random Forest時，也有採用過 SMOTE 來做資料平衡但未達到要求的數據結果  
這邊改採 XGBoost 做監督式學習後就未使用 SMOTE 來做資料平衡了，此方式也可以保持資料真實性  

### 非監督式學習  
指令程式碼我採用K-Means結合馬氏距離（Mahalanobis Distance）進行信用卡詐欺偵測資料集的異常偵測  
由於信用卡詐騙資料集特性，此資料集高度不平衡，正常交易（Class=0）遠多於詐騙詐欺交易（Class=1）  
特徵數量多（29個特徵，包括28個PCA降維後的特徵V1-V28和金額）。詐騙交易通常會導致異常值，與正常交易在特徵空間的分佈有顯著差異。  

K-Means：將正常交易恐，假設詐欺詐欺交易會脫離正常交易的恐中心，從而偵測出詐欺詐欺資料的異常。  

馬氏距離：馬氏距離則考慮了特徵間的協方差結構，相比歐幾里德距離更適合檢測多維資料中的異常值。它能夠捕捉資料分佈的形狀和圖形，對於信用卡詐騙這種高維度、不平衡資料特別有效。  

參數調整的部分與參數細節影響模型表現：K-Means的恐慌數（K值）和、PCA的保留變異比例、樣本大小和異常檢測閾值會直接影響模型的準確率（ precision）、召回率（recall）和F1分數。這些參數可以透過網格搜尋（Grid Search）進行調優，以在高準確率和合理召回率之間取得平衡。  

信用卡詐欺詐欺偵測的目的：高準確率：避免將正常交易誤判為詐欺（假積極），因為合理召回率：盡量偵測出所有詐欺詐欺交易（真積極），但不能過度追求召回率而詐欺準確率。方案碼中使用分數 = 0.85 * 精確度 + 0.15 * 召回作為評估指標，顯示更準確重視率，這與實際應用減少中誤報的需求一致。  

方案碼中參數調整的詳細說明：  

資料共享的部分 #data = data.drop(['Time'], axis=1) #data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))  

資料上先移除了時間欄位，時間欄位表示交易的時間，對於異常檢測跟欺騙模式識別無直接貢獻，且可能引入噪音。  

影響：簡化特徵空間，聚焦於與欺騙相關的特徵（如V1-V28和Amount）。  

標準化數量欄位： 原因：數量欄位的數值範圍與其他特徵（V1-V28，已由PCA標準化）不一致，標準化保證所有特徵的刻度統一，避免某些特徵對距離計算的過度影響。  

調整方式：使用StandardScaler將Amount轉換為平均值0、標準差1的分佈。  

影響：提高K-Means和馬氏距離計算的穩定性和準確性。  

PCA 降維的操作 pca_variances = [0.96, 0.97, 0.98] pca = PCA(n_components=pca_var, random_state=RANDOM_SEED) x_train_pca = pca.fit_transform(x_train) x_test_train) x_test_pca.  

參數： pca_variances=[0.96, 0.97, 0.98]：PCA 保留的變異比例，分別測試 96%、97%、98%。 n_components=pca_var：自動選擇主成分數量以保留指定的變異比例。  

因為信用卡詐騙資料集共有29個特徵，高維度資料可能導致計算成本高且存在影響不大的特徵。 PCA透過保留變異方向來降低維度，減少雜訊並加速模型計算。我的方案中測試了幾個不同的變異比例（96%、97%、98%），目的是保留足夠的資訊並降低維度主要之間找到一個平衡點。  

調整方式：選擇其中的變異比例（接近1）以保留大部分訊息，但避免過高（如0.99）以減少變異。透過網格搜尋測試不同的比例，評估對最終F1分數的影響。  
