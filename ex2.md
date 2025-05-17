---
# 挑戰二：Hybrid Model 
---

## ✅ 使用技術與思路

### 1. 🎯 特徵選擇

挑選出高規律性特徵進行訓練：

```python
selected_features = ['V1','V2','V3','V4','V5','V6','V7',
                     'V9','V10','V11','V12','V14','V16','V17','V18','V19','Amount']
```

---

### 2. 🔄 資料平衡處理

* 採用下採樣技術，從正常樣本中抽出 5000 筆與詐欺樣本結合，避免類別不平衡造成模型偏誤。

---

### 3. 🧊 Isolation Forest（非監督式異常分數）

* 只用正常樣本進行訓練（`y == 0`）
* 使用 `decision_function()` 產生 anomaly score（越低越異常）。
* 將異常分數「反向」後視為風險特徵加入原始資料中：

```python
iso_train = (-iso.decision_function(X_train)).reshape(-1, 1)
X_train_if = np.hstack([X_train, iso_train])
```

---

![image](https://github.com/user-attachments/assets/ed6e3c6e-740a-4390-930a-680bdc8c01aa)




