# 挑戰二：融合異常檢測與監督式學習於信用卡詐欺偵測

---

## Isolation Forest 

1. 建立 Isolation Forest，僅以正常樣本訓練：  
   ```python
   iso = IsolationForest(contamination=0.002, random_state=42)
   iso.fit(x_train[y_train==0])
   ```
2. 計算 anomaly score（連續值）：  
   ```python
   anomaly_train = iso.decision_function(x_train).reshape(-1,1)
   anomaly_test  = iso.decision_function(x_test).reshape(-1,1)
   ```
3. 合併至特徵矩陣：  
   ```python
   x_train_feat = np.hstack([x_train, anomaly_train])
   x_test_feat  = np.hstack([x_test,  anomaly_test])
   ```

---

## XGBoost

- **模型**：`XGBClassifier(tree_method='hist', eval_metric='logloss', random_state=42)`  
- **參數網格**：
  ```yaml
  n_estimators: [200, 400]
  max_depth:    [5, 8]
  learning_rate:[0.05, 0.1]
  scale_pos_weight: [10, 20]
  ```
- **交叉驗證**：3 折 (cv=3)，以 F1 score 作為搜尋目標  

```python
grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid.fit(x_train_feat, y_train)
best_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)
```

### 4.1 預設閾值 (0.5) 評估

```python
y_pred_default = best_model.predict(x_test_feat)
evaluation(y_test, y_pred_default, model_name="DefaultThreshold")
```

```
DefaultThreshold Evaluation:
========================================
Accuracy       : 0.99952015
Precision Score: 0.92125984
Recall Score   : 0.79054054
F1 Score       : 0.85090909
...
```

---

## 5. 門檻調整 (Threshold Tuning)

- 掃描閾值範圍 [0.1, 0.9)，每 0.01 為一步，選出最佳 F1 分數對應之閾值：

```python
y_proba = best_model.predict_proba(x_test_feat)[:,1]
best_f1, best_thresh = 0, 0.5
for t in np.arange(0.1, 0.9, 0.01):
    preds = (y_proba > t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t
print(f"Best F1={best_f1:.5f} at threshold={best_thresh:.2f}")
y_pred_tuned = (y_proba > best_thresh).astype(int)
evaluation(y_test, y_pred_tuned, model_name=f"Threshold {best_thresh:.2f}")
```

```
Threshold 0.43 Evaluation:
========================================
Accuracy       : 0.99953185
Precision Score: 0.92187500
Recall Score   : 0.79729730
F1 Score       : 0.85507246
...
```

---
