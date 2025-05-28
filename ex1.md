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
![image](https://hackmd.io/_uploads/HyRzb5bGgl.png)
可以找到上面這組參數的數據是最佳的
![image](https://hackmd.io/_uploads/Bk-NWcWGee.png)
![image](https://hackmd.io/_uploads/HJRVbcbfel.png)
![image](https://hackmd.io/_uploads/ByvBZqWMgg.png)
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
![image](https://hackmd.io/_uploads/ryVg0DEfgl.png)
Kmeans 這邊也是嘗試了許久
可能因為分群不會太多，只有對或錯而已
但嘗試將分群設為兩個的時候，卻又比較差
![image](https://hackmd.io/_uploads/B1QMJ_Vzxl.png)
於是就用迴圈自動跑多種不同的分群數
最終能找到 8 為最好的結果
