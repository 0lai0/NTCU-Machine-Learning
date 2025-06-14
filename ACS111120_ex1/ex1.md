
# ACS111120 - 挑戰一說明

## 模型架構
- 監督式學習: Random Forest (100 顆樹)
- 非監督式學習: KMeans (群數根據 silhouette score 動態選擇)

## 資料處理
- Amount 標準化
- 移除 Time 欄位

## 評估指標
- Precision, Recall, F1, Accuracy

## 改進建議
- 嘗試調整 Random Forest 樹數、深度
- KMeans 增加群數搜尋範圍
