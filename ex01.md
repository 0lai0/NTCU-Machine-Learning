# 信用卡詐欺檢測報告

## 1. 監督式學習：XGBoost

**參數設定**  
- `n_estimators=600`  
- `max_depth=8`  
- `learning_rate=0.1`  
- `scale_pos_weight=8`  
- `random_state=42`  
- `eval_metric='logloss'`  

> 調高 `n_estimators` 與 `max_depth`，並加大 `scale_pos_weight` 以處理嚴重不平衡問題。

### 範例結果

```
Random Forest Evaluation:
--------------------------------------------
Accuracy       : 0.99963719
Precision Score: 0.94117647
Recall Score   : 0.82352941
F1 Score       : 0.87843137

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.94      0.82      0.88       136

    accuracy                           1.00     85443
   macro avg       0.97      0.91      0.94     85443
weighted avg       1.00      1.00      1.00     85443
--------------------------------------------
```

### 實作結果

```
XGBoost Evaluation:
--------------------------------------------
Accuracy       : 0.99969570
Precision Score: 0.94354839
Recall Score   : 0.86029412
F1 Score       : 0.90000000

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.94      0.86      0.90       136

    accuracy                           1.00     85443
   macro avg       0.97      0.93      0.95     85443
weighted avg       1.00      1.00      1.00     85443
--------------------------------------------
```

---

## 2. 非監督式學習：KMeans

> **改動**：原本只用正常樣本訓練，改成取 800 筆正常樣本與 200 筆詐騙樣本共同訓練。

### 範例結果

```
KMeans (Unsupervised) Evaluation:
--------------------------------------------
Accuracy       : 0.99872430
Precision Score: 0.78260870
Recall Score   : 0.36486486
F1 Score       : 0.49769585

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.78      0.36      0.50       148

    accuracy                           1.00     85443
   macro avg       0.89      0.68      0.75     85443
weighted avg       1.00      1.00      1.00     85443
--------------------------------------------
```

### 實作結果

```
MY KMeans Evaluation:
--------------------------------------------
Accuracy       : 0.99897007
Precision Score: 0.83333333
Recall Score   : 0.50675676
F1 Score       : 0.63025210

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.83      0.51      0.63       148

    accuracy                           1.00     85443
   macro avg       0.92      0.75      0.81     85443
weighted avg       1.00      1.00      1.00     85443
--------------------------------------------
```
