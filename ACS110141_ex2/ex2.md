# 半監督式學習
**使用XGBoost+IsolationForest**<br>
![image](https://github.com/piHD/NTCU-Machine-Learning/blob/main/ACS110141_ex2/result_Pic/Hybrid.png)<br>
- 從先前最佳結果進行調整。
- 能使IsolationForest結果較高的處理資料(有stratify=Y)，會使XGBoost結果下滑。所以刪除該參數，雖然在非監督的部分結果下滑(如下表)，但結合的成果有明顯上升。

```
IsolationForest去除資料的stratify=Y :
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.25      0.68      0.37       136

    accuracy                           1.00     85443
   macro avg       0.62      0.84      0.68     85443
weighted avg       1.00      1.00      1.00     85443
```