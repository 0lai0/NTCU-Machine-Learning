# ex2
## Random Forest + LabelSpreading
labelSpreading應用於label data不足但有大量unlabel data的場景
最後再用生出來的虛擬標籤跑random forest
```
Model Evaluation:
=============================================
         Accuracy: 0.9987594068560327
  Precision Score: 0.7058823529411765
     Recall Score: 0.4864864864864865
         F1 Score: 0.576

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.71      0.49      0.58       148

    accuracy                           1.00     85443
   macro avg       0.85      0.74      0.79     85443
weighted avg       1.00      1.00      1.00     85443
```