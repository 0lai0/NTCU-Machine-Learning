# 信用卡詐欺偵測 - 有監督與非監督學習比較報告

## 專題架構：信用卡詐欺偵測之分類與分群實作  

---

## 挑戰一：有監督式學習（Supervised Learning）

### 模型：Random Forest 分類器

#### 執行步驟簡述：
- 使用 `creditcard.csv` 資料集
- 資料標準化與特徵選擇（去除 `Time`、標準化 `Amount`）
- 使用 `train_test_split()` 切分訓練與測試集
- 應對資料不平衡問題，使用 `class_weight='balanced'`
- 建立 `RandomForestClassifier` 模型訓練與測試

#### 評估結果（實際執行）：
Random Forest Evaluation:
Accuracy: 0.9995318516437859
Precision Score: 0.9576271186440678
Recall Score: 0.7635135135135135
F1 Score: 0.849624060150376

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 85295   |
| 1     | 0.96      | 0.76   | 0.85     | 148     |

#### 解讀：
- Accuracy 高達 99.95%，但更值得注意的是：
  - **召回率（Recall）為 76.3%**，能成功抓出大多數詐欺交易
  - F1 Score 達 0.85，整體模型表現穩定，對少數類別仍具備辨識能力

---

## 挑戰二：非監督式學習（Unsupervised Learning）

### 模型：KMeans 分群

#### 執行步驟簡述：
- 無使用標籤資料，直接以特徵進行分群
- 使用 `Silhouette Score` 決定最佳分群數（k = 2）
- 透過 `majority voting` 對齊預測標籤與實際標籤進行比較
- 使用 `classification_report` 評估

#### 評估結果（實際執行）：
KMeans (Unsupervised) Evaluation:
Accuracy: 0.9982678510820079
Precision Score: 0.0000000000000000
Recall Score: 0.0000000000000000
F1 Score: 0.0000000000000000

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 85295   |
| 1     | 0.00      | 0.00   | 0.00     | 148     |

#### 解讀：
- 雖然 Accuracy 為 99.83%，但實際上：
  - 模型完全無法預測任何詐欺交易（class 1）
  - 所有資料都被分為非詐欺類（class 0），導致 precision/recall 為 0

---

## 小結與比較：

| 模型類型             | 使用標籤 | Accuracy | Precision | Recall | F1 Score | 能力評估                    |
|----------------------|-----------|----------|-----------|--------|----------|-----------------------------|
| **有監督式（Random Forest）** | ✅ 有     | 0.9995   | 0.96      | 0.76   | 0.85     | ✅ 成功預測大部分詐欺行為     |
| **非監督式（KMeans）**         | ❌ 無     | 0.9983   | 0.00      | 0.00   | 0.00     | ❌ 完全無法預測詐欺行為       |

### 總結：
- 有監督學習（如 Random Forest）在處理**極度不平衡資料**時，仍能有效偵測少數的詐欺樣本。
- 非監督學習（如 KMeans）雖然在沒有標籤的情況下可做初步探索，但面對詐欺樣本數極低的資料時，容易將所有資料歸為正常類別，導致結果失真。
- **建議：**若資料具標籤且目標為分類預測，應以監督式方法為主。

---
