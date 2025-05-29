# 挑戰二
結果
```
hybrid Evaluation:
=============================================
         Accuracy: 0.9996605924417448
  Precision Score: 0.9572649572649573
     Recall Score: 0.8235294117647058
         F1 Score: 0.8853754940711462
```
percision提升0.03，recall降低0.03，1些微下降
### 資料處理
- 先對資料進行標準化
- 再使用PCA降維保留95%變異量，以提升訓練效率
### Isolation Forest
使用`Isolation Forest`檢測異常樣本(非監督學習)
- `contamination=0.01` 表示預期約 1% 的資料為異常
- `predict()` 的輸出為：1（正常）或 -1（異常）
** 把異常結果加入PCA，當作一個新的特徵，讓後面的XGboost更好的捕捉詐欺特徵 **
### XGboost
與挑戰一的監督式學習一樣，設定參數並配合配合Bayesian Optimization調整。

### 參數調整
