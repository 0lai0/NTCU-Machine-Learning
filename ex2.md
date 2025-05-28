### 選用方式
這次使用的是三種不同的東西混和在一起進行辨識
1. Isolation Forest
由這個東西去做異常檢測，把異常分數作為特徵
2. PCA+Kmeans
然後 PCA 去做降為，接著讓去進行分類來抓局部特徵
3. XGBoost
最後 XGBoost 來把所有特徵整合再一起，用之前 ex1 拿到的最佳參數去進行訓練，最終調整一下閾值來平衡 Precision 跟 Recall

### 參數內容
- Isolation Forest
```
iso_forest = IsolationForest(
    n_estimators=100, 
    contamination=0.002,  # 與詐欺比例匹配
    random_state=RANDOM_SEED
)
```

- KMeans 聚類
```
kmeans = KMeans(
    n_clusters=2,
    init=seed_centroids,  # 用標籤資料初始化
    n_init=1,             # 固定初始化提升穩定性
    random_state=RANDOM_SEED
)
```
- XGBoost 最佳化參數
```
xgb_hybrid = XGBClassifier(
    n_estimators=500,      # 足夠的樹數
    max_depth=8,           # 平衡擬合與泛化
    learning_rate=0.02,    # 細粒度學習
    scale_pos_weight=2,    # 調整類別權重
    gamma=0.05,            # 正則化控制
    eval_metric='logloss'  # 二元分類最佳指標
)
```
### 為甚麼選擇三個混和在一起
其實就是很簡單的其中兩個互相組合都不太有好結果，數據普遍都沒有到很好看
所以才嘗試將三個組在一起
結果就得到意外的狀況
而監督式學習本身 Ex1 就挑了主流的模型跑過一次
基本可以確定 XGBoost 是目前最適合這個資料集的
然後就跑出以下結果ㄌ

### 結果

```
=== Multi-Hybrid Model (PCA + Isolation Forest + KMeans + XGBoost) ===
Best Threshold: 0.9963

Multi-Hybrid Model Evaluation:
=============================================
         Accuracy: 0.9995084442259752
  Precision Score: 0.9732142857142857
     Recall Score: 0.7364864864864865
         F1 Score: 0.8384615384615385

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.97      0.74      0.84       148

    accuracy                           1.00     85443
   macro avg       0.99      0.87      0.92     85443
weighted avg       1.00      1.00      1.00     85443

ROC-AUC: 0.9717
```
