ex1說明文件

監督式學習(random forest)
我觀察資料集後發現資料嚴重不平衡(詐騙資料占比不足1%),為了改善模型在少數類別上的辨識能力，於是修改了參數
class_weight={0:1, 1:10} 解決資料嚴重不平衡的問題，把詐騙資料的權重提高，強迫模型更加關注少數類別
max_depth=25 限制樹的最大深度，避免overfitting的狀況

雖然 F1-score 與提供的範例相同（0.88），但 Precision 略高（0.96 vs 0.94），表示模型的預測更準確，誤判率較低

![image](https://github.com/user-attachments/assets/8499dd90-f87c-4b39-b83c-ce2d346197da)


非監督式學習(KMeans)
KMeans由於維度太高、特徵間的區別性不明確；詐騙樣本稀少，導致分群容易全部落在正常群為改善，我加入了PCA降維
所以我嘗試透過加入PCA提升模型的準確程度，先用PCA處理過資料可以讓其分類更有代表性，透過PCA(n_components=0.95)進行降維，保留95%資料變異的重要特徵去除冗餘的特徵與雜訊，有助於提升KMeans的效果與穩定度

最終實驗中來的 KMeans 模型在所有評分指標上均略有提升，顯示 PCA 的降維與樣本選擇策略有效改善非監督式學習的效果

![image](https://github.com/user-attachments/assets/0c1e5aa9-80a1-4750-9087-de4dee0a7aac)

