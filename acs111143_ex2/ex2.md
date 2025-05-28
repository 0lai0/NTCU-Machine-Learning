### 1. **模型選擇：KMeans 輔助特徵結合 XGBoost**


- **XGBoost**：
  - 是一個高效的梯度提升樹（Gradient Boosting Decision Tree, GBDT）實作，訓練快速且效果優異。
  - 對於不平衡資料（詐欺偵測中詐欺樣本極少）可透過參數 `scale_pos_weight` 進行調整。
  - 具有良好的泛化能力與調參空間，適合本任務。

- **KMeans 聚類輔助特徵**：
  - 對訓練資料做 PCA 降維後，使用 KMeans 進行聚類，並取得每筆資料到兩個中心點的距離作為新特徵。
  - 這些距離特徵能提供模型額外的資料分布資訊，幫助提升判斷詐欺樣本的能力。
  - 透過先使用部分有標籤的資料建立聚類中心，提升聚類的代表性。

---

### 2. **參數設定與流程說明**

- **資料前處理**：
  - 移除 `Time` 欄位，因其無明顯意義。
  - 對 `Amount` 欄位做標準化。
  - 使用 PCA 降維，保留 7 個主成分，保持與原始特徵維度相近。
  
- **KMeans 聚類**：
  - 使用部分訓練資料中正常與詐欺樣本分別挑選一定數量資料，建立初始聚類中心（`centroids_init`）。
  - 用這些中心初始化 KMeans，避免隨機初始化造成效果不穩。
  - 將資料點到兩個中心的距離作為額外特徵加入訓練資料。

- **XGBoost 參數**：

```python
xgb_model = XGBClassifier(
    n_estimators=250,           # 樹數較多以強化模型
    max_depth=6,                # 中等深度避免過擬合
    learning_rate=0.08,         # 穩健學習率
    subsample=0.8,              # 隨機抽樣訓練資料增加多樣性
    colsample_bytree=1.0,       # 使用所有特徵
    scale_pos_weight=10,        # 調整類別不平衡權重
    gamma=0.05,                 # 控制樹複雜度避免過擬合
    use_label_encoder=False,
    eval_metric='aucpr',        # 以 AUC-PR 衡量表現
    tree_method='hist',
    random_state=RANDOM_SEED
)
```

- **自動threshold**


利用模型預測的機率分數，設定較高閾值（0.80），提高 Precision 同時仍維持合理 Recall。

透過掃描不同閾值，找出最佳 F1-score 對應的閾值，取得模型最佳平衡點。

- **小結**

調整過後他的precison非常高但是recall卻沒有很理想，推測是模型較保守判定以及資料極度不平衡的緣故
