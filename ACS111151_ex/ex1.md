1. 前處理
- 資料來源：`data/creditcard.csv`
- 刪除 `Time` 欄位，對 `Amount` 做 StandardScaler。

2. 監督式實驗：SMOTE + RandomForest
- SMOTE 過採樣後的訓練集：正/負樣本比例接近平衡。
- RandomForest 參數：`n_estimators=100, class_weight='balanced'`。
- 結果：
  - Precision、Recall、F1-score、ROC AUC 如下表。

| 類別 | Precision | Recall | F1    |
|----|---------|-------|-------|
| 0  | …       | …     | …     |
| 1  | …       | …     | …     |

3. 非監督式實驗：KMeans(k=3)
- 對全資料做標準化後聚成三群，每群以多數真實標籤做預測
- 結果：
  - Precision、Recall、F1-score 如下。

4. 結論
- 監督式方法效果遠優於非監督式。
