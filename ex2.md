# 挑戰二：融合監督與非監督學習

## 🧪 實驗目標
結合非監督式學習（IsolationForest）與監督式學習（XGBoost），提升信用卡詐騙偵測的分類能力，並比較與單一模型的差異。

---

## 📊 使用資料集
- 來源：Kaggle - Credit Card Fraud Detection
- 筆數：284,807 筆
- 詐騙比例：僅 0.172%
- 欄位：經 PCA 處理的 V1~V28、Amount（已標準化）、Class（標記）

---

## 🧠 融合模型設計

### 1️⃣ IsolationForest（Unsupervised）
- 作用：找出資料中潛在的異常交易
- 產出：每筆資料的 `anomaly_score`
- 新增欄位：將 anomaly_score 加入 X 特徵資料中

### 2️⃣ XGBoost（Supervised）
- 以原始特徵 + anomaly_score 作為輸入特徵
- 訓練二元分類模型以偵測詐騙

---

## 📈 模型效能評估（融合模型）

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.93      0.76      0.84       148

    accuracy                           1.00     85443
   macro avg       0.96      0.88      0.92     85443
weighted avg       1.00      1.00      1.00     85443
```

---

## ✅ 成效分析

- 融合模型的 recall（召回率）從單一模型的 82% 提升到 84%
- XGBoost 能有效利用非監督模型提供的 anomaly_score，增強分類能力
- IsolationForest 對正常資料建立背景，有助於詐欺樣本被更準確識別

---

## 📁 檔案結構（繳交）
```
<學號>_ex2/
├── ex2.ipynb
└── ex2.md
```