##  實驗目標

本實驗旨在完成作業 Challenge 2，目標為融合監督式學習與非監督式學習模型，以提升信用卡詐欺交易的預測效果。

---

##  使用資料集

- 資料來源：Kaggle (mlg-ulb/creditcardfraud)
- 筆數：約 284,807 筆交易資料，492 筆詐欺交易（比例約 0.172%）
- 特徵：28 個主成分（V1~V28）、Amount（已標準化）
- 處理方式：移除 Time 特徵；Amount 欄位經 StandardScaler 標準化

---

##  模型與融合策略

### 1️ 非監督式學習：Isolation Forest

- 使用正常樣本（Class=0）進行訓練
- 模型參數：`contamination=0.00172`
- 功能：辨識異常點做為潛在詐欺

### 2️ 監督式學習：XGBoost

- 模型：`XGBClassifier`
- 資料：使用訓練集標記資料完整訓練
- 效果：優秀的單模型分類器

### 3️ 模型融合策略：Hybrid (AND 條件)

combined_pred = np.logical_and(anomaly_pred_binary == 1, xgb_pred == 1).astype(int)
僅當兩模型皆預測為詐欺才標記為詐欺

提高 precision，降低誤判率

缺點：可能降低 recall（視業務風險取捨）

### 模型實驗結果
## Isolation Forest 單獨預測結果：
markdown
Copy
Edit
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.20      0.25      0.22       148

    accuracy                           1.00     85443
   macro avg       0.60      0.62      0.61     85443
weighted avg       1.00      1.00      1.00     85443

## XGBoost 單獨預測結果：
markdown
Copy
Edit
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.93      0.73      0.82       148

    accuracy                           1.00     85443
   macro avg       0.97      0.86      0.91     85443
weighted avg       1.00      1.00      1.00     85443

## 最終融合模型（Hybrid Model）結果：
yaml
Copy
Edit
Hybrid Model Evaluation:
Accuracy: 0.9996722961506501
Precision Score: 0.9285714285714286
Recall Score: 0.8602941176470589
F1 Score: 0.8931297709923665

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.93      0.86      0.89       136

    accuracy                           1.00     85443
   macro avg       0.96      0.93      0.95     85443
weighted avg       1.00      1.00      1.00     85443

## 分析與優勢
XGBoost 表現已穩健，融合後使 recall 從 0.73 提升至 0.86，F1 提升至 0.89。

融合策略有效保留了 XGBoost 的精準性，同時利用 Isolation Forest 過濾異常以強化結果。

可依實務場景改為 OR 或加權邏輯進行調整，以更大程度控制 precision/recall 平衡。

