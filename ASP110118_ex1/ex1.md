# EX1: 信用卡詐騙檢測進階版本

## 1. 專案概述
本專案使用進階機器學習技術建立信用卡詐騙檢測模型，透過特徵工程、採樣技術比較、模型ensemble方法，提升詐騙檢測效能。

**固定參數**: RANDOM_SEED=42, TEST_SIZE=0.3  
**資料集**: KaggleHub信用卡詐騙資料集 (mlg-ulb/creditcardfraud)

## 2. 技術方法

### 2.1 選擇的技術
- **特徵工程**: 時間特徵(Hour/Day/Time_Period)、金額轉換(log/sqrt/binning)、V特徵統計量
- **異常值檢測**: IsolationForest偵測並移除異常值(contamination=0.1)
- **採樣技術**: 比較Original、SMOTE、ADASYN、SMOTE+ENN四種方法
- **模型算法**: RandomForest、GradientBoosting、LogisticRegression、SVM
- **評估分析**: ROC-AUC、PR曲線、混淆矩陣、特徵重要性

### 2.2 實作策略
1. **特徵工程**: 新增多種工程特徵包含時間、金額和V特徵統計量
2. **資料清理**: 使用IsolationForest移除異常值樣本
3. **採樣對比**: 系統比較四種採樣技術效果
4. **模型比較**: 基於效能指標選擇最佳模型

### 2.3 模型選擇理由
選擇RandomForest、GradientBoosting等樹型模型因其對不平衡資料的穩健性，LogisticRegression提供線性基準比較，SVM在高維特徵空間表現優異。各模型使用class_weight='balanced'處理類別不平衡。

## 3. 實驗設計

### 3.1 實驗設置
- **資料分割**: 70%訓練集，30%測試集
- **特徵縮放**: RobustScaler處理特徵標準化
- **採樣比較**: 四種方法在相同模型架構下對比
- **評估重點**: F1-Score和ROC-AUC指標

### 3.2 評估標準
針對不平衡資料特性，重點關注F1-Score平衡精確率與召回率，ROC-AUC評估整體分類能力，PR曲線分析少數類別表現。

## 4. 結果分析

### 4.1 定量結果
- **最佳組合**: 通過採樣方法和ensemble模型組合獲得最佳效能
- **特徵效果**: 工程特徵對模型預測能力有正面貢獻
- **採樣效果**: 不同採樣方法在效能上展現差異
- **異常值處理**: 移除異常值後改善模型表現

### 4.2 視覺化分析
通過效能指標熱力圖展示模型比較結果，ROC/PR曲線分析分類效能，混淆矩陣詳細分析預測結果，特徵重要性識別關鍵預測因子。

### 4.3 性能分析
個別模型在不同採樣技術下展現不同優勢，通過系統性比較能有效識別最適合的模型與採樣組合，達到處理不平衡資料的分類效果。

## 5. 結論與討論

### 5.1 主要發現
1. **特徵工程**: 時間和金額相關特徵對詐騙檢測有幫助
2. **採樣技術**: 不同採樣方法適用性有差異
3. **模型比較**: 系統性評估多種算法的效能差異
4. **異常值處理**: IsolationForest改善資料品質

### 5.2 技術貢獻
- 實現完整的特徵工程pipeline
- 提供採樣技術系統性比較
- 建立多模型比較評估框架
- 達成進階技術整合應用

### 5.3 限制與改進方向
1. **計算成本**: 多模型訓練增加運算時間
2. **特徵選擇**: 可探索更多特徵組合
3. **參數優化**: 可進行更細緻的超參數調整
4. **部署考量**: 需評估實際應用的效能需求

## 6. 參考資料
- Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Oversampling Technique
- Breiman, L. (2001). Random Forests
- Liu, F. T. et al. (2008). Isolation Forest
- Kaggle Credit Card Fraud Detection Dataset 