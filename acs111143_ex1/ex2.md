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
xgb_model = XGBClassifier(
    n_estimators=500,        # 越多模型越穩定但計算成本提高
    max_depth=7,             # 避免過擬合
    learning_rate=0.05,      # 學習率，控制每棵樹對最終預測的貢獻程度
    subsample=0.8,           #（防止過擬合）
    colsample_bytree=1.0,    # 每棵樹使用的特徵比例
    scale_pos_weight=5,      # 正負樣本不平衡時使用，放大正類（詐欺）權重
    gamma=0.05,              # 節點分裂的最小損失減少，避免過度分裂
    use_label_encoder=False, 
    bootstrap=True,          # 啟用自助採樣（類似隨機森林）
    eval_metric='logloss',  
    random_state=RANDOM_SEED 
)
```
調整 scale_pos_weight 為 10 時，Precision（精確率）大幅降低，可能是模型過度傾向預測為正類（詐欺），導致誤判正常交易。

```python
iso_forest = IsolationForest(
    n_estimators=500,                     
    max_samples="auto",                  # 每棵樹使用的樣本數（自動選擇）
    contamination=sum(y_train)/len(y_train), # 異常比例，設定為訓練集中詐欺樣本比例
    random_state=RANDOM_SEED,            # 固定隨機種子以利重現
    bootstrap=True                       
)

```

### 3. **設定預測機率閾值Threshold**

用下來結果當 combined_score 的 threshold 調高至 0.3 以上時，Recall（召回率）明顯下降至 0.75 以下。
推測是因為非監督式模型在資料不平衡或異常比例極低的情況下，其偵測結果準確性有限
```python
combined_score = 0.9 * xgb_prob + 0.1 * (-iso_scores)  # iso_scores 越小越異常
combined_pred = (combined_score > 0.3).astype(int)


### 4. **最終結果**


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
