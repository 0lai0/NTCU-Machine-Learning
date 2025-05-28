# ex1
## 監督式學習
使用Random Forest
範例程式中新增threshold，使判定為詐騙的機率提高
```python=
y_score = rf_model.predict_proba(X_test)[:,1]
threshold = 0.25
y_pred = (y_score > threshold).astype(int)
evaluation(y_test, y_pred, model_name="RandomForestClassifier(Supervised)")
```
提升recall，降低precision

```
RandomForestClassifier(Supervised) Evaluation:
=============================================
         Accuracy: 0.9996254813150287
  Precision Score: 0.9
     Recall Score: 0.8602941176470589
         F1 Score: 0.8796992481203008

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.90      0.86      0.88       136

    accuracy                           1.00     85443
   macro avg       0.95      0.93      0.94     85443
weighted avg       1.00      1.00      1.00     85443
```

## h2 非監督式學習
使用Kmeans
範例程式中新增與中心要夠進才會判定為詐騙
```python=
y_pred_aligned = align_labels(y_test, y_pred_test, optimal_k)
distances = kmeans.transform(x_test)
max_dist = np.max(distances, axis=1)
threshold = np.percentile(max_dist, 99)
inliers = (max_dist > threshold).astype(int)
y_pred_final = ((y_pred_aligned == 1) & (inliers == 1)).astype(int)
```

```
KMeans (Unsupervised) Evaluation:
=============================================
         Accuracy: 0.998537036386831
  Precision Score: 0.9259259259259259
     Recall Score: 0.16891891891891891
         F1 Score: 0.2857142857142857

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.93      0.17      0.29       148

    accuracy                           1.00     85443
   macro avg       0.96      0.58      0.64     85443
weighted avg       1.00      1.00      1.00     85443
```

雖大幅提升了precision，但recall大幅降低



