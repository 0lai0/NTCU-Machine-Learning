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

![image](https://hackmd.io/_uploads/HJ4oz5EMxl.png)

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

![image](https://hackmd.io/_uploads/SJVbNq4Ggg.png)

雖大幅提升了precision，但recall大幅降低



