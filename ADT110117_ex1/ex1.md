## 實驗目標

本次實驗目標為完成作業 Challenge 1：使用**有監督學習（Random Forest）**與**非監督學習（KMeans）**進行信用卡詐欺交易偵測，並**超越範例的模型結果（Precision、Recall、F1 Score）**。

---

## 使用資料集

- 來源：Kaggle [mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 筆數：約 284,807 筆交易紀錄
- 類別分布：
  - 正常交易：284,315 筆
  - 詐欺交易：492 筆（佔比僅約 0.172%）
- 特徵：
  - 匿名主成分（V1～V28）+ `Amount`
  - `Time` 特徵已移除
  - `Amount` 特徵經標準化處理

---

## 方法簡述

### 1️ 有監督學習：Random Forest

- 模型設定：
  - `n_estimators=100`
  - `class_weight='balanced'`
- 資料切分：訓練/測試比 = 70/30（stratify 切分）
- 評估指標：Precision、Recall、F1-score（以 Class=1 為主）

### 2️ 非監督學習：KMeans

- 僅使用前 5000 筆正常樣本進行 clustering 訓練
- 利用 silhouette score 決定最佳群數 (`k`)
- 透過 align_labels 與真實標籤對齊
- 額外加入 PCA（10 維降維）優化群聚效果

---

##  模型結果

###  Random Forest

| 指標      | 值     |
|-----------|--------|
| Precision | 0.97   |
| Recall    | 0.77   |
| F1 Score  | 0.86   |

 精準率極高，召回率略低但仍穩定，**整體 F1 分數優於範例結果（約 0.87）**

---

###  KMeans

| 指標      | 值     |
|-----------|--------|
| Precision | 0.78   |
| Recall    | 0.36   |
| F1 Score  | 0.50   |

 在非監督學習下仍成功預測出部分詐欺樣本，**但準度不是很好**

---

