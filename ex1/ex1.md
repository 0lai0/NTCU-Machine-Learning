1. 監督式學習（隨機森林）
使用100 n_estimators

參數：
max_depth=20：限制樹深度防止過擬合
class_weight='balanced_subsample'：自動調整權重

並且透過PR曲線尋找最佳 threshold，來最大化F1分數

2. 非監督式學習（K-Means）
僅使用正常交易數據訓練模型（模擬真實場景）

自動尋找最佳的 cluster count（2-4範圍）
使用 silhouette score 評估 clustering 效果

標籤對齊：將 clusters 結果映射到真實的標籤，並且基於多數決原則分配類別