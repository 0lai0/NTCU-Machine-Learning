## 1. 用KMeans訓練模型

### 1 訓練資料篩選
- 使用正常(non-fraud)資料進行 KMeans 訓練  
- 取前 1000 筆正常資料以降低計算成本  
- 取前 100 筆不正常資料
### 2 參數調整
```python
    RANDOM_SEED = 42
    TEST_SIZE = 0.3
    N_NORMAL_SEED = 1000
    N_FRAUD_SEED = 100
    N_PCA_COMPONENTS = 7
    ENSEMBLE_RUNS = 1
```
經過測試 N_NORMAL_SEED 只要不是1000 不管其他參數如何挑整都會是0
PCA降維測試過5~20， 最終測試區間在5~10叫好， 細部測試後發現7是最高的(有個有趣的現象如果設定是4以下，那他會直接壞掉，Kmeans會無法聚合)
使用ENSEMBLE_RUNS投票，發現這功能好像沒啥關係，推測因為數據極度不平衡所以每次投票結果都是一樣的，所以後來就把他設定為1了減少計算成本
---

## 3. 評估結果
```python
Kmeans (Unsupervised) Evaluation:
=============================================
         Accuracy: 0.9989817773252344
  Precision Score: 0.8144329896907216
     Recall Score: 0.5337837837837838
         F1 Score: 0.6448979591836734

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.81      0.53      0.64       148

    accuracy                           1.00     85443
   macro avg       0.91      0.77      0.82     85443
weighted avg       1.00      1.00      1.00     85443
```
---

## 4. 其他探討

發現簡報上面的範例有先取一小部分的資料，且只挑選正常樣本，這樣能算是純非監督嗎?


