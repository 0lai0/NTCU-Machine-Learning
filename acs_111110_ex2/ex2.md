混和(Isolation+XGBoost)

參數調整

iso_forest__n_estimators = 200
iso_forest__contamination = 0.0017
xgb__n_estimators = 200
xgb__max_depth = 6
xgb__learning_rate = 0.1
xgb__eval_metric = 'aucpr'
xgb__min_child_weight = 5
xgb__gamma = 0.5
xgb__colsample_bytree = 1.0
xgb__subsample = 0.6
xgb__alpha = 0.1
scale_pos_weight_val_fixed = 3
pca__n_components = 12

方式

標準化 -> Isolation -> PCA降維 -> 合併特徵 -> XGBoost -> 預測機率 -> 找最佳閥值 -> 評斷output

output


Best Threshold for F1-score: 0.1940 (F1: 0.9091)

Hybrid Evaluation:
=============================================
         Accuracy: 0.9997074072773662
  Precision Score: 0.937007874015748
     Recall Score: 0.875
         F1 Score: 0.9049429657794676

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.94      0.88      0.90       136

    accuracy                           1.00     85443
   macro avg       0.97      0.94      0.95     85443
weighted avg       1.00      1.00      1.00     85443