---

# 挑戰一
---

## 📌 使用特徵

我根據網路上別人分析的資料特性，挑選了具有明顯分佈規律的特徵進行建模：

```python
selected_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
                     'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'Amount']
```

---

## 🧠 監督式模型：XGBoost

### 使用技術

* **分類器**：`XGBoostClassifier`
* **類別不平衡處理**：使用 `scale_pos_weight` 自動調整類別權重
* **特徵選擇**：只使用高相關性的特徵欄位
* **資料處理**：

  * 使用下採樣技術從正常樣本中取出 5000 筆與詐欺資料結合
  * `train_test_split` 分割訓練與測試資料

---

![image](https://github.com/user-attachments/assets/32a61012-eee5-4997-9fc3-1ff5c6ec5ef2)



## 🔍 非監督式模型：Isolation Forest

### 使用技術

* **模型**：`IsolationForest`
* **訓練策略**：只使用正常樣本（`y == 0`）來避免資訊外洩
* **異常偵測**：

  * 利用 `decision_function` 計算異常分數（分數越低表示越異常）
  * 分數反向處理後使用百分位數（例如 96th percentile）決定分類閾值

---

![image](https://github.com/user-attachments/assets/4a277acb-fb25-4cc8-b9c5-da83caea7ced)



