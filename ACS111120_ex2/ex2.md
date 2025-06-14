
# ACS111120 - 挑戰二說明

## 模型架構
- 非監督式學習: Isolation Forest
- 監督式學習: XGBoost

## 融合方式
- Isolation Forest 先標記異常 (作為新特徵)
- XGBoost 結合原始特徵與異常特徵進行分類

## 評估指標
- Precision, Recall, F1, Accuracy

## 改進建議
- Isolation Forest 可調整 contamination 比例
- 嘗試其他異常檢測模型融合
