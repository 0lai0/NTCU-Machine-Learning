
# 🛡️ Credit Card Fraud Detection using Hybrid Unsupervised and Supervised Learning

本專案展示如何結合 **非監督式學習（Isolation Forest、PCA）** 與 **監督式學習（XGBoost）**，提升信用卡詐欺偵測的準確率與效能。

資料來源：[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📌 專案架構概述

我們將流程分為以下幾個步驟：

1. **資料前處理**
2. **標準化特徵**
3. **使用 Isolation Forest 擷取異常分數**
4. **使用 PCA 降維提取特徵**
5. **合併特徵**
6. **XGBoost 建模與評估**

---

## 📦 使用套件

```bash
pip install pandas numpy scikit-learn xgboost kagglehub
```

---

## 📁 資料載入與前處理

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")

data['Class'] = data['Class'].astype(int)
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
```

---

## 🔍 非監督式學習階段（特徵擴充）

### 1. **Isolation Forest**

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.0017, random_state=42)
iso_forest.fit(X_train_std)
```

| 參數名稱 | 說明 |
|----------|------|
| `contamination` | 設定異常樣本的比例（根據詐欺率） |
| `random_state` | 確保可重現性 |

### 2. **PCA 降維**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10, random_state=42)
pca.fit(X_train_std)
```

| 參數名稱 | 說明 |
|----------|------|
| `n_components` | 提取前 10 個主成分 |
| `random_state` | 確保可重現性 |

---

## 🧠 監督式學習階段（模型訓練）

### 合併特徵

```python
X_train_enhanced = np.hstack((X_train_std, train_anomaly_scores, X_train_pca))
X_test_enhanced = np.hstack((X_test_std, test_anomaly_scores, X_test_pca))
```

### 訓練 **XGBoost** 模型

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    scale_pos_weight=2.5,
    eval_metric='logloss',
    tree_method='hist',
    random_state=42
)
xgb_model.fit(X_train_enhanced, y_train)
```

| 參數名稱 | 說明 |
|------------|------|
| `learning_rate` | 控制每棵樹對最終結果的貢獻 |
| `max_depth` | 限制樹的深度防止過擬合 |
| `n_estimators` | 建立樹的數量 |
| `subsample` | 每棵樹的訓練樣本比例 |
| `scale_pos_weight` | 解決類別不平衡問題 |
| `tree_method` | 使用直方圖加速法訓練 |
| `eval_metric` | 評估方式：logloss |

---

## 🎯 模型預測與評估

```python
from sklearn.metrics import classification_report

y_prob = xgb_model.predict_proba(X_test_enhanced)[:, 1]
y_pred_custom = (y_prob > 0.43).astype(int)
```

### 評估函數

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluation(y_true, y_pred, model_name="Model"):
    print(f'\n{model_name} Evaluation:')
    print('===' * 15)
    print('         Accuracy:', accuracy_score(y_true, y_pred))
    print('  Precision Score:', precision_score(y_true, y_pred))
    print('     Recall Score:', recall_score(y_true, y_pred))
    print('         F1 Score:', f1_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
```

### 📊 範例結果

| 指標 | 分數 |
|------|------|
| **Accuracy** | 0.9997 |
| **Precision** | 0.9365 |
| **Recall** | 0.8676 |
| **F1 Score** | 0.9008 |

---

## 結論與優勢

本專案成功結合非監督與監督學習技術，達到高效能詐欺偵測。

**優勢包括：**

- 整合多種模型以強化辨識力
- 特徵組合具備代表性與靈活性
- 適用於其他異常檢測場景

---


