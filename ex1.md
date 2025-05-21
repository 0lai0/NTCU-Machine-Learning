
# **信用卡詐欺偵測：XGBoost 與 AutoEncoder 方法比較**

## **資料集說明**
本研究使用的資料集為 Kaggle 上的「信用卡詐欺資料集（Credit Card Fraud Detection）」。該資料包含 284,807 筆交易記錄，其中僅 492 筆為詐欺交易，比例約為 0.173%，具有嚴重的類別不平衡問題。

---

## **監督式學習：XGBoost**

### **使用理由**
- **適合處理非平衡資料集**：XGBoost 可透過 `scale_pos_weight` 權重調整機制處理類別不平衡。
- **高準確率與穩定性**：在分類問題中具備強大的表現力與泛化能力。
- **內建剪枝與缺失值處理**：提升模型效率與穩健性。

### **主要參數設定說明**

| 參數 | 值 | 說明 |
|------|----|------|
| `colsample_bytree` | 1.0 | 使用全部特徵建樹 |
| `learning_rate` | 0.1 | 控制模型收斂速度 |
| `max_depth` | 6 | 控制每棵樹的深度，防止過擬合 |
| `n_estimators` | 200 | 弱分類器的總數量 |
| `subsample` | 0.8 | 每棵樹訓練時使用 80% 的資料 |
| `scale_pos_weight` | 2.5 | 為了處理極端類別不平衡的問題（正負樣本比例約為 1:578） |
| `tree_method` | hist | 使用直方圖加速建模，適合大型資料集 |

### **XGBoost 成效**
- **Accuracy**: 0.9997
- **Precision**: 0.9435
- **Recall**: 0.8603
- **F1 Score**: 0.9

---

## **非監督式學習：AutoEncoder**

### **使用理由**
- **無需標記資料**：AutoEncoder 僅使用非詐欺資料進行訓練，適合偵測未知攻擊行為。
- **能自動學習正常樣本的特徵壓縮與還原**：透過重建誤差判斷異常行為。
- **可用於資料稀疏或標記昂貴的場景**

### **模型架構與訓練細節**

| 層級 | 結構 | 激活函數 |
|------|------|----------|
| Encoder Layer 1 | Linear(dim → 16) | ReLU |
| Encoder Layer 2 | Linear(16 → 8) | ReLU |
| Decoder Layer 1 | Linear(8 → 16) | ReLU |
| Decoder Layer 2 | Linear(16 → dim) | None |

- 損失函數：MSE (均方誤差)
- 最佳閾值：由 precision-recall curve 所導出
- 訓練資料：僅使用 `Class=0` 的正常樣本

### **AutoEncoder 成效**
- **Accuracy**: 0.9980
- **Precision**: 0.4515
- **Recall**: 0.7230
- **F1 Score**: 0.5558

---

## **比較分析**

| 指標 | XGBoost | AutoEncoder |
|------|---------|-------------|
| Accuracy | **0.9997** | 0.9980 |
| Precision | **0.9435** | 0.4515 |
| Recall | 0.8603 | **0.7230** |
| F1 Score | **0.9** | 0.5558 |

- **XGBoost 適用於有標記資料集，可達到較佳分類精度**
- **AutoEncoder 可用於無標記資料探索異常點，在 Recall 表現相對較佳**
- 實務中可搭配使用：先用 AutoEncoder 篩選可疑樣本，再以 XGBoost 精準分類

---

## **結論**
XGBoost 與 AutoEncoder 各有優勢。針對類別不平衡問題，XGBoost 透過樣本加權與樹模型可有效處理；而 AutoEncoder 則提供無監督異常偵測的能力。實務上可依據資料標記情況與應用需求選擇模型或進行混合建模以達最佳成效。
