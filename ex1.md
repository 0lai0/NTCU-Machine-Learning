# 信用卡詐欺偵測 - 監督式學習 (XGBoost) 與非監督式學習 (KMeans)

## 專案說明
本專案利用信用卡交易資料集，採用兩種不同的機器學習方法來進行詐欺偵測：

- **監督式學習**：使用 XGBoost (Extreme Gradient Boosting) 分類器，透過標註的資料學習如何辨識詐欺交易。
- **非監督式學習**：使用 KMeans 聚類，根據交易特徵進行分群，並分析群組內的詐欺比例來做異常偵測。

此資料集類別極度不平衡，詐欺交易佔比約 0.173%。因此，我們在模型設計上特別考慮不平衡問題。

---

## 資料來源
- [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 監督式學習：XGBoost

### 模型說明
XGBoost 是一種基於梯度提升樹（Gradient Boosting Tree）的強大分類器，具備高效能與彈性，適合處理大量且複雜的資料。

### 資料前處理
- 移除 `Time` 欄位，因為其意義對模型幫助有限。
- 對 `Amount` 欄位做標準化（StandardScaler），讓特徵值集中於均值為 0，標準差為 1，有助模型學習。
- 分割訓練與測試資料，比例為 7:3，確保模型能在未見資料上評估。

### 主要參數及意義

| 參數               | 設定值  | 說明                                                         |
|--------------------|---------|--------------------------------------------------------------|
| `colsample_bytree` | 1.0     | 每棵樹隨機抽取的特徵比例，1.0 代表使用全部特徵。             |
| `learning_rate`    | 0.1     | 學習率，控制每棵樹對模型的貢獻，較小的值使訓練更穩定但較慢。  |
| `max_depth`        | 6       | 單棵樹最大深度，控制模型複雜度與防止過擬合。                 |
| `n_estimators`     | 200     | 樹的數量，更多樹通常帶來更好的擬合，但也增加計算成本。        |
| `subsample`        | 0.8     | 每棵樹用於訓練的樣本比例，幫助減少過擬合。                   |
| `scale_pos_weight` | 2.5     | 正類（詐欺）的權重調整，因為資料不平衡，用以提升對詐欺的識別。|
| `eval_metric`      | 'logloss'| 評估指標，對於二分類問題來說，logloss 可以衡量模型預測的準確性。|
| `tree_method`      | 'hist'  | 使用 histogram 演算法加速訓練，適用於大資料。                 |
| `random_state`     | 42      | 設定隨機種子，確保結果可重現。                               |

### 模型結果

| 評估指標      | 結果          | 說明                           |
|---------------|---------------|--------------------------------|
| Accuracy      | 0.9997        | 整體正確率                     |
| Precision     | 0.9435        | 正確預測詐欺的比例             |
| Recall        | 0.8603        | 成功找到詐欺樣本的比例         |
| F1 Score      | 0.9           | Precision 與 Recall 的調和平均 |

### 調整說明
- 使用 `scale_pos_weight` 來平衡詐欺類別權重。
- 自訂概率閾值（threshold = 0.43）來調整 Precision 與 Recall 的平衡，提升偵測能力。

---

## 非監督式學習：KMeans 聚類

### 模型說明
KMeans 是一種無監督學習方法，透過將資料分群，找出群內異質性的異常點。在詐欺偵測中，期望詐欺交易能分布於特定群組，透過群內詐欺比例評估異常程度。

### 資料前處理
- 同樣標準化 `Amount` 特徵。
- 移除 `Time` 欄位。
- 使用 Stratified Split 確保訓練與測試資料中詐欺比例一致。

### 主要參數及流程
- `n_clusters`：群數，透過 Silhouette Score 在 2~10 群中尋找最佳群數。
- 使用 `k-means++` 初始化，避免隨機初始化帶來的不穩定。
- 迭代上限設定較大（max_iter=700）確保收斂。

### 結果
- 最佳群數為 2。
- 以群內詐欺比例作為異常分數。
- ROC AUC 約 0.753，顯示模型對詐欺的區分能力一般。
- F1 Score 最佳閾值 0.8333，Precision=0.83，Recall=0.51。

### 模型限制
- KMeans 本質為距離導向，對於極度不平衡與詐欺樣本分佈複雜的資料，偵測效果有限。
- Precision 與 Recall 的權衡偏向保守，召回率不高，可能漏掉部分詐欺樣本。

---

## 總結
- XGBoost 作為監督式模型，利用標註資料及類別不平衡調整，有較好的詐欺識別表現。
- KMeans 作為非監督式模型，雖無法完全取代監督式方法，但能在無標籤資料情況下提供初步異常偵測。
- 兩者結合可針對不同場景與資料可用性做彈性調整。

---

## 執行環境與套件

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- kagglehub

---

## 使用說明

1. 下載信用卡詐欺資料集
2. 執行監督式學習程式碼訓練 XGBoost 模型
3. 執行非監督式學習程式碼進行 KMeans 分群分析
4. 依據輸出評估結果調整參數

---

## 參考資料
- [XGBoost 官方文件](https://xgboost.readthedocs.io/en/stable/)
- [Scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- Kaggle - Credit Card Fraud Detection Dataset

