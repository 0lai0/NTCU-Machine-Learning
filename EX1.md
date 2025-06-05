# Credit Card Fraud Detection - 實驗說明

作者：ACS111107  簡祐暄
作業類型：Machine Learning 練習作業 (挑戰一)  
日期：2025/5/29  

---

## 使用資料集

- 資料來源：[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 總筆數：284,807
- 詐騙交易數：492（佔 0.172%）
- Class：0 = 正常，1 = 詐騙

---

## 實驗設定

- 使用固定設定：
  - `RANDOM_SEED = 42`
  - `TEST_SIZE = 0.3`
- 評估指標：
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

## 使用的模型

### 1. 監督式學習
- 套件：`sklearn.ensemble.RandomForestClassifier`
- 調整參數：
  ```python
  rf_model = RandomForestClassifier(
      n_estimators=100, 
      random_state=42
  )
  ```
改用 XGBClassifier
- 套件 from xgboost import XGBClassifier
   ```python
    rf_model = XGBClassifier(
      n_estimators=235,
      max_depth=6,
      learning_rate=0.16,
      scale_pos_weight=100,
      eval_metric='logloss',
      use_label_encoder=False,
      random_state=42
    )
   ```
     就監督式學習，XGBClassifier 較適合使用在此實例上，再慢慢測試個參數的結果，即可得出此結果。

  ### 2. 非監督式學習
- 調整參數：
  ```python
  n_x_train = n_x_train[:1000]
  ```
- 增加範圍設置
   ```python
    normal = x_train[y_train == 0][:800]
    fraud = x_train[y_train == 1][:200]
    n_x_train = np.vstack([normal, fraud])
   ```
     從百位數開始慢慢測試到個位數，不知道是不是巧合發現800、200是最佳的數值。
  
   
