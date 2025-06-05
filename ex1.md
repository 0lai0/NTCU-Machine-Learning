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
