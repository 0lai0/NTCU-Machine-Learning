學長：
監督式學習：
Random Forest Evaluation:
===============================
Accuracy: 0.9996371850239341
Precision Score: 0.9411764705882353
Recall Score: 0.8235294117647058
F1 Score: 0.8784313725490196

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.94      0.82      0.88       136

    accuracy                           1.00     85443
   macro avg       0.97      0.91      0.94     85443
weighted avg       1.00      1.00      1.00     85443

非監督式學習：
KMeans (Unsupervised) Evaluation:
=============================================
         Accuracy: 0.9987242957293166
  Precision Score: 0.782608695652174
     Recall Score: 0.36486486486486486
         F1 Score: 0.4976958525345622

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.78      0.36      0.50       148

    accuracy                           1.00     85443
   macro avg       0.89      0.68      0.75     85443
weighted avg       1.00      1.00      1.00     85443

ACS111144：
ex1_1：(第一版：本監督式學習precision勝；非監督式學習recall勝、f1-score勝）
--- Supervised: LightGBM with Optuna 評估報告 ---
======================================================
        準確率 (Accuracy): 0.9996
        精確率 (Precision): 0.9435
        召回率 (Recall): 0.7905
        F1 分數 (F1 Score): 0.8603

分類報告 (Classification Report):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.94      0.79      0.86       148

    accuracy                           1.00     85443
   macro avg       0.97      0.90      0.93     85443
weighted avg       1.00      1.00      1.00     85443

======================================================

--- 自動編碼器 (非監督式) 評估報告 ---
======================================================
        準確率 (Accuracy): 0.9986
        精確率 (Precision): 0.5535
        召回率 (Recall): 0.6471
        F1 分數 (F1 Score): 0.5966

分類報告 (Classification Report):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.55      0.65      0.60       136

    accuracy                           1.00     85443
   macro avg       0.78      0.82      0.80     85443
weighted avg       1.00      1.00      1.00     85443

======================================================

ex1_2：(第二版：本監督式學習recall勝；非監督式學習recall勝、f1-score勝）
--- XGBoost 分類器 評估結果 ---
======================================================
準確率 (Accuracy)           : 0.9993
精準度 (Precision, 詐騙類)     : 0.7785
召回率 (Recall, 詐騙類)        : 0.8311
F1 分數 (F1 Score, 詐騙類)    : 0.8039
ROC AUC                  : 0.9730
PR AUC (Average Precision): 0.8338

分類報告 (Classification Report):
                precision    recall  f1-score   support

正常 (Class 0)       1.00      1.00      1.00     85295
詐騙 (Class 1)       0.78      0.83      0.80       148

      accuracy                           1.00     85443
     macro avg       0.89      0.92      0.90     85443
  weighted avg       1.00      1.00      1.00     85443

======================================================

--- Autoencoder (非監督式) 評估結果 ---
=============================================
準確率 (Accuracy)           : 0.9985
精準度 (Precision, 詐騙類)     : 0.5208
召回率 (Recall, 詐騙類)        : 0.5515
F1 分數 (F1 Score, 詐騙類)    : 0.5357

分類報告 (Classification Report):
                precision    recall  f1-score   support

正常 (Class 0)       1.00      1.00      1.00     85307
詐騙 (Class 1)       0.52      0.55      0.54       136

      accuracy                           1.00     85443
     macro avg       0.76      0.78      0.77     85443
  weighted avg       1.00      1.00      1.00     85443
