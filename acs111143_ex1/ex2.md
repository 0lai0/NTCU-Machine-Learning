### 1. **模型選擇：XGBoost 與 Isolation Forest**

我選擇使用 `XGBoost` 作為分類模型，原因如下：

- **處理速度快**：XGBoost 是經過優化的梯度提升模型，能快速處理大量資料(因為我的電腦配備不好，Random Forest真的太慢了)。
- **穩定性佳，表現強**：實務中在 Kaggle、金融、醫療等領域表現蠻好的。

我選擇使用 `Isolation Forest` 的理由:

- **不需要依賴標籤即可學習資料分布特性、執行速度快**: 但是好像非監督式都挺快速的。
---

### 2. **參數設定說明**

參數設計理由：

```python
xgb = XGBClassifier(
    n_estimators=250,           # 樹的數量較多，增強模型表現
    max_depth=6,                # 深度 適中，避免過度擬合
    learning_rate=0.08,         # 低學習率，更穩健收斂
    subsample=0.8,              # 80%樣本比例訓練，增加多樣性
    colsample_bytree=1.0,       
    scale_pos_weight=15,       # 約為非詐欺與詐欺比率，解決類別不平衡
    gamma=0.05,                  # 控制樹的複雜度，防止過擬合
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_SEED
)
```
主要調整為scale_pos_weight，選擇15是因為執行出來的正負樣本比率為17倍，那測試過後15的表現是最好的
max_depth=6深度太小會導致欠擬合，太大容易過擬合。爬文後深度 6 為 XGBoost 處理 tabular data 時的常見推薦值。
學習率設為 0.08。在 0.05~0.2 範圍中測試後，0.08 能穩定學習且不會太快陷入局部最佳。
```python
iso_forest = IsolationForest(
    n_estimators=500,                     
    max_samples="auto",                  # 每棵樹使用的樣本數（自動選擇）
    contamination=sum(y_train)/len(y_train), # 異常比例，設定為訓練集中詐欺樣本比例
    random_state=RANDOM_SEED,            # 固定隨機種子以利重現
    bootstrap=True                       
)

```
每棵樹所使用的樣本數量設定為 "auto"，表示預設使用 min(256, n_samples)。這能平衡模型訓練速度與效能，且是官方建議的預設值。
### 3. **最終結果**


Combined (XGB + IsoForest) Evaluation:
=============================================
         Accuracy: 0.9995669627705019
  Precision Score: 0.937007874015748
     Recall Score: 0.8040540540540541
         F1 Score: 0.8654545454545455

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.94      0.80      0.87       148

    accuracy                           1.00     85443
   macro avg       0.97      0.90      0.93     85443
weighted avg       1.00      1.00      1.00     85443
