
# 💼 ex1.md - 挑戰一 練習作業報告
> 📌 課程練習目標：使用監督與非監督式方法進行詐欺交易偵測

---

## 👤 基本資訊
- 作者：[資三甲周庭嫻ACS111125]
- 作業名稱：挑戰一 - 信用卡詐欺資料分析

---

## 🎯 目標說明

本次練習目的為透過機器學習方法，對信用卡詐欺偵測資料集進行處理與模型建立，並嘗試比較 **監督式學習（Random Forest）** 以及 **非監督式學習（KMeans）** 的偵測效果。

---

## 📦 使用資料集

- 資料來源：[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 資料筆數：284,807 筆
- 欄位數量：30 個欄位（經 PCA 處理）
- 資料不平衡比例：
  - 正常交易（Class=0）：99.8%
  - 詐欺交易（Class=1）：0.2%

---

## 🧹 資料前處理

1. 移除 `Time` 欄位
2. 使用 `StandardScaler` 對 `Amount` 欄位進行標準化
3. 分離特徵 X 與標籤 Y
4. 使用 `train_test_split` 切分資料為 70% 訓練 / 30% 測試（保留類別比例）

---

## ✅ 模型一：Random Forest（有監督式學習）

- 模型參數：
  - `n_estimators=100`
  - `class_weight='balanced'`
- 訓練資料：70%
- 測試資料：30%

### 📊 評估結果

```
Accuracy : 0.9994
Precision: 0.9719
Recall   : 0.7027
F1 Score : 0.8159
```

| 類別 | precision | recall | f1-score | support |
|------|-----------|--------|----------|---------|
| 0    | 1.00      | 1.00   | 1.00     | 85295   |
| 1    | 0.97      | 0.70   | 0.82     | 148     |

> ✅ 模型在極度不平衡資料下仍能達到高 precision 與合理 recall，透過 `class_weight='balanced'` 可提升少數類別偵測效果。

---

## ✅ 模型二：KMeans（非監督式學習）

- 使用非詐欺樣本（Class = 0）前 1000 筆進行群聚學習
- 探索最佳群數 k ∈ {2, 3, 4}
- 使用 silhouette score 選出最佳群數
- 最佳 k = **2**

### 📊 評估結果

```
Accuracy : 0.9983
Precision: 0.00
Recall   : 0.00
F1 Score : 0.00
```

| 指標類型     | 值   |
|--------------|------|
| macro avg    | 0.50 |
| weighted avg | 1.00 |

> ⚠️ 雖然整體 accuracy 看似很高，但實際未能有效辨識詐欺樣本，precision 與 recall 為 0，表示所有詐欺樣本皆被錯判。

---

## 📊 模型比較與總結

| 模型        | Accuracy | Precision | Recall | F1 Score |
|-------------|----------|-----------|--------|----------|
| RandomForest| 0.9994   | 0.9719    | 0.7027 | 0.8159   |
| KMeans      | 0.9983   | 0.0000    | 0.0000 | 0.0000   |

> ✅ **Random Forest** 在處理不平衡資料中表現穩定，能有效抓出詐欺樣本。  
> ⚠️ **KMeans** 雖然是無監督方法，accuracy 偏高主要是因為資料極度不平衡，實際上未成功辨識任何詐欺交易。

---

## 🔁 建議改進方向（TODO）

- 嘗試 **XGBoost**，進一步提升 Recall 與 F1 分數
- 使用 **SMOTE** 進行少數類別過採樣
- 嘗試調整門檻值（如 `predict_proba >= 0.3~0.5`）
- 加入更多非監督模型：如 Isolation Forest、DBSCAN
- 將模型評估結果可視化（ROC 曲線、PR 曲線、混淆矩陣）

---

## ✅ 結論

本次實驗展示監督與非監督學習方法在信用卡詐欺偵測中的應用，Random Forest 能有效從極度不平衡的資料中辨識詐欺行為，而 KMeans 雖為無監督方法，在未經調整的情況下效果有限。透過後續進階技術（如 SMOTE 或 Boosting），可望進一步提升偵測表現。

---
