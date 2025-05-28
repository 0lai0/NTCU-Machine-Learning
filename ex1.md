# ex1

## 監督式
在這邊其實嘗試了很多不同的監督式學習的模型
H2O GrandientBoostingEstimator、LGB、CatBoost
甚至弄了簡單的 MLP
本來想說要用 MLP 去做，但可能資料量太小或本身就是二元分類不需要這麼複雜
結果就是沒有一組數據是比較好的
最後還是回來用 XGB 來做
既然要用 XGB 來做了，那就不要一直手動慢慢測很累
所以建立了個 array 來自動跑

```
xgb_best = XGBClassifier(
    tree_method='hist',
    device='cuda',
    n_estimators=500,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=1.0,
    scale_pos_weight=2,
    gamma=0.05,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_SEED
)
```
可以找到上面這組參數的數據是最佳的
```
   n_estimators  max_depth  learning_rate  subsample  colsample_bytree  \
0           500          8           0.02      0.700               1.0   
1           500          8           0.02      0.750               1.0   
2           500          8           0.02      0.650               1.0   
3           500          8           0.02      0.725               1.0   

   Accuracy  Precision    Recall        F1  
0  0.999707   0.966387  0.845588  0.901961  
1  0.999661   0.934959  0.845588  0.888031  
2  0.999696   0.958333  0.845588  0.898438  
3  0.999684   0.950413  0.845588  0.894942  
```
```
XGB Evaluation:
=============================================
         Accuracy: 0.9997074072773662
  Precision Score: 0.9663865546218487
     Recall Score: 0.8455882352941176
         F1 Score: 0.9019607843137255

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.97      0.85      0.90       136

    accuracy                           1.00     85443
   macro avg       0.98      0.92      0.95     85443
weighted avg       1.00      1.00      1.00     85443
```
```
Random Forest Evaluation:
=============================================
         Accuracy: 0.9996137776061234
  Precision Score: 0.9478260869565217
     Recall Score: 0.8014705882352942
         F1 Score: 0.8685258964143426

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.95      0.80      0.87       136

    accuracy                           1.00     85443
   macro avg       0.97      0.90      0.93     85443
weighted avg       1.00      1.00      1.00     85443
```
各項數據都超過了原先的 Random Forest 的結果，並且速度上因為我加入了 GPU 加速，所以速度快很多

```
XGB Evaluation:
=============================================
         Accuracy: 0.9997074072773662
  Precision Score: 0.9663865546218487
     Recall Score: 0.8455882352941176
         F1 Score: 0.9019607843137255
```

## 非監督式
```
KMeans (Unsupervised) Evaluation:
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
Kmeans 這邊也是嘗試了許久
可能因為分群不會太多，只有對或錯而已
但嘗試將分群設為兩個的時候，卻又比較差
```
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
```
於是就用迴圈自動跑多種不同的分群數
最終能找到 8 為最好的結果


