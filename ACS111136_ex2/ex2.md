Hybrid Model
1.目的:
混合模型結合監督式學習（如 XGBoost）與非監督式學習（如 Isolation Forest），利用兩者優勢提升在信用卡詐欺偵測中的準確度與召回率，特別適用於處理不平衡資料集與未知異常樣本。
2.流程：

非監督異常檢測：Isolation Forest
使用 IsolationForest 模型對標準化後的交易資料進行訓練。

模型會為每筆樣本產生 anomaly score（異常分數），分數愈低代表該樣本愈異常。

此步驟不使用資料標籤，僅依據資料分佈學習異常行為。

目的：從資料中額外萃取「異常程度」作為新特徵。

監督式分類器：XGBoost
使用 XGBClassifier 對新特徵集進行訓練。

主要參數設定如下：

n_estimators=100, max_depth=5, learning_rate=0.1：常見的控制樹模型深度與學習速率。

scale_pos_weight=負:正類比例：動態調整權重以應對類別不平衡問題。

模型會預測每筆交易為詐欺的機率（probability）

門檻調整
使用 precision_recall_curve 評估不同門檻下的 Precision 與 Recall。

採用策略：

精確率≥ 0.90 前提下，選擇（Recall）最大 的門檻值。

目的：降低誤報（false positives）並盡可能捕捉更多詐欺樣本。

3.適用範圍
適用於異常樣本稀少、異常行為模式多變的金融詐欺偵測任務

評估指標
Accuracy：整體分類正確率。

Precision：詐欺預測中實際為詐欺的比例（低誤報率）。

Recall：成功預測出詐欺的比例（高召回能力）。

F1-score：Precision 與 Recall 的加權平均數。

Threshold Adjusted Hybrid Model Evaluation:
=============================================
         Accuracy: 0.9994
  Precision Score: 0.9024
     Recall Score: 0.75
         F1 Score: 0.8192

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.90      0.75      0.82       148

    accuracy                           1.00     85443
   macro avg       0.95      0.87      0.91     85443
weighted avg       1.00      1.00      1.00     85443

4.數據為甚麼recall score 會低於預期?
Recall 是針對正類（詐欺交易），XGBoost 被過度限制以追求高 Precision。
詐欺樣本數量太少，模型學不到足夠樣式
Isolation Forest 的分數幫助有限
可能是這些受到影響?