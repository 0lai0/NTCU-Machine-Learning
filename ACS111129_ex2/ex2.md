## 實驗結果
* Accuracy: 比範例多約 **0.00001**
* Precision Score: 比範例多約 **0.01**
* F1 Score: 比範例多約 **0.01**

```
Hybrid Model Evaluation:
=============================================
         Accuracy: 0.9996839998595555
  Precision Score: 0.936
     Recall Score: 0.8602941176470589
         F1 Score: 0.896551724137931

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.94      0.86      0.90       136

    accuracy                           1.00     85443
   macro avg       0.97      0.93      0.95     85443
weighted avg       1.00      1.00      1.00     85443
```

## 使用模型說明
在這個實驗中先是採用 **PCA** 將特徵維度下降，再用 **IsolationForest** 偵測異常交易數值，並加入訓練資料中作為特徵之一，再用 **XGBoost Classifier** 做分類
主要調整了以下參數：
* scale_pos_weight：因為資料極度不平衡，因此調高詐欺交易的 weight
