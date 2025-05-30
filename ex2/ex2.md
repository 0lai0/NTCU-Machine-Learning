本程式碼使用 XGBoost融合Isolation Forest，Isolation Forest 模型來預測每筆交易是0 或 1
contamination=0.001 設定代表預期異常比例為 0.1%

相比於ex1加入了自動選擇閾值的功能，會遍歷閾值 0.01–0.99，找到 F1-score 最高的作為分類依據
還加入自動欠取樣，能自行選擇非詐騙樣本數設為詐騙的n倍
