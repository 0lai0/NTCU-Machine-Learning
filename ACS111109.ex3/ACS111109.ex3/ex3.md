# ex3 - 混合模型：Isolation Forest + XGBoost

## 實驗目標
這次實驗要把非監督式的 Isolation Forest 和監督式的 XGBoost 結合，提升信用卡詐欺交易的預測準確度。

## 使用方法

- **Isolation Forest**：先幫忙找出可能的異常交易。
- **XGBoost**：用來做最後的分類預測。

## 資料處理

- 使用 Kaggle 上的「信用卡詐欺資料集」
- 移除 `Time` 欄位
- 把 `Amount` 欄位做標準化（standardization）
- 資料非常不平衡，詐欺只佔 0.17%

## 模型流程

1. 用 Isolation Forest 找出異常交易，新增欄位 `is_anomaly`
2. 把 `is_anomaly` 當成額外的特徵
3. 把全部特徵（包含 `is_anomaly`）拿去訓練 XGBoost 分類器

## 評估指標

- 主要看 Precision、Recall、F1-score
- 如果結果比單純用 Random Forest 或 KMeans 更好，就算成功

## 小結

加入非監督模型找出異常交易後，能幫助分類模型做得更準，特別是抓到更多詐欺樣本。
