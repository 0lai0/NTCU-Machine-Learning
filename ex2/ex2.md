# 非監督式學習階段：Isolation Forest
核心概念：Anomalous data points 比 normal points 更容易被「隔離」
運作方式：通過隨機選擇特徵和分割值構建 decision trees

優勢：更有效率的處理高維的數據，並且使計算複雜度低

## 程式內容

contamination 參數：設定預期異常值比例，影響判定 threshold
不使用標籤：完全基於數據分布特性檢測異常
輸出轉換：將預測結果作為新特徵加入原始數據

初步篩選可疑的交易內容，透過「異常程度」的量化指標，增強後續監督模型的 feature space

# 監督式學習階段：XGBoost
核心概念：串聯多個 weak learners 形成 strong predictor

優勢：處理不平衡數據能力強，防止過擬合
XGBoost 的特點：內建正則化，自動處理缺失的數值