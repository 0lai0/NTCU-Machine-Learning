### 1. **模型選擇：XGBoostClassifier**

我選擇使用 `XGBoost` 作為分類模型，原因如下：

- **處理速度快**：XGBoost 是經過優化的梯度提升模型，能快速處理大量資料(因為我的電腦配備不好，Random Forest真的太慢了)。
- **穩定性佳，表現強**：實務中在 Kaggle、金融、醫療等領域表現蠻好的。

---

### 2. **參數設定說明**

參數設計理由：

```python
xgb_model = XGBClassifier(
    n_estimators=220,           # 建立 220 棵樹，提升模型表現
    max_depth=6,                # 控制每棵樹的最大深度，避免過擬合，並維持學習能力
    learning_rate=0.12,         # 較低學習率可讓模型穩定收斂（避免梯度爆炸）
    subsample=0.8,              # 每棵樹只使用 80% 的樣本，增加模型的泛化能力
    colsample_bytree=1.0,       # 每棵樹使用全部特徵（對此資料量無太大負擔）
    scale_pos_weight=2.5,       # 非詐欺與詐欺類別比例為 ~2.5：1，設定此參數解決類別不平衡問題
    gamma=0.05,                 # 控制模型複雜度，避免生成不必要的分支，抑制過擬合
    use_label_encoder=False,    # 避免 label encoding 警告
    eval_metric='logloss',      # 二分類常用的評估指標
    random_state=42             # 固定隨機種子以利重現結果
)
```
### 3. **設定預測機率閾值Threshold**

用下來結果預設的0.5是最好的
```python
y_prob = xgb_model.predict_proba(X_test)[:, 1]  # 取得預測為正類（詐欺）的機率
threshold = 0.5                                 # 根據 precision-recall 或驗證結果選定
y_pred_custom = (y_prob > threshold).astype(int)
```

XGBoost Evaluation:
=============================================
         Accuracy: 0.9997074072773662
  Precision Score: 0.9512195121951219
     Recall Score: 0.8602941176470589
         F1 Score: 0.9034749034749036

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.95      0.86      0.90       136

    accuracy                           1.00     85443
   macro avg       0.98      0.93      0.95     85443
weighted avg       1.00      1.00      1.00     85443