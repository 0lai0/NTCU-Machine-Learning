
# ACS111109.ex2 - 混合模型說明文件

## 資料處理流程

1. 使用 kagglehub 載入 creditcard.csv。
2. 移除 Time 欄位，對 Amount 欄位標準化。
3. 將資料分為特徵 X 和標籤 Y。
4. 使用 train_test_split 切成訓練集與測試集。
5. 只用正常樣本訓練 AutoEncoder。

## 模型架構

- 輸入層 → Dense(17, tanh) → Dense(8, relu)
- → Dense(17, tanh) → 輸出層
- 損失函數：MSE
- 優化器：Adam
- 訓練次數：100 epochs
- 批次大小：64

## 異常偵測

- 模型對測試資料進行重建，計算每筆資料的 MSE。
- 使用 precision_recall_curve 找出最佳 F1 分數的門檻值。
- 將 MSE 大於門檻者判為詐欺。

## 評估指標

- Accuracy
- Precision
- Recall
- F1 Score
- classification_report

## 輸出

- 每個 epoch 的訓練與驗證損失
- 最佳門檻值與其 precision / recall / f1
- 最終分類結果評估
- 訓練損失圖
