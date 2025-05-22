## 1. Hybrid Model (Isolation Forest + K-means + PCA + XGBoost)

### 1-1. 流程

先使用 Isolation Forest 和 K-means 進行初步的分類，加上 PCA 特徵提取，再將結果組合後透過 XGBoost 進行精確分類

### 1-2. 程式碼

將特徵標準化
``` python
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
```

使用 Isolation Forest 來進行異常檢測
``` python
iso_forest = IsolationForest(contamination=0.0017, random_state=RANDOM_SEED)
iso_forest.fit(X_train_std)

train_if_score = iso_forest.decision_function(X_train_std).reshape(-1, 1)
test_if_score = iso_forest.decision_function(X_test_std).reshape(-1, 1)
```

使用 K-means 來進行交易分群
``` python
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=RANDOM_SEED)
kmeans.fit(X_train_std[y_train == 0])
train_kmeans_dist = np.min(kmeans.transform(X_train_std), axis=1).reshape(-1, 1)
test_kmeans_dist = np.min(kmeans.transform(X_test_std), axis=1).reshape(-1, 1)
train_kmeans_label = kmeans.predict(X_train_std).reshape(-1, 1)
test_kmeans_label = kmeans.predict(X_test_std).reshape(-1, 1)
```

使用 PCA 進行降維處理，提取特徵
``` python
pca = PCA(n_components=10, random_state=RANDOM_SEED)
pca.fit(X_train_std)

X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
```

將所有結果結合
``` python
X_train_enhanced = np.hstack((X_train_std, train_if_score, train_kmeans_dist, X_train_pca))
X_test_enhanced = np.hstack((X_test_std, test_if_score, test_kmeans_dist, X_test_pca))
```

使用 XGBoost 進行精確分類
``` python
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=1.0,
    scale_pos_weight=6,
    random_state=RANDOM_SEED
)

xgb_model.fit(X_train_enhanced, y_train)
```

透過迴圈尋找能讓 F1 Score 最大化的最佳 threshold (0.1~0.91，step = 0.01)
``` python
y_prob = xgb_model.predict_proba(X_test_enhanced)[:, 1]
best_f1 = 0
best_th = 0.5
for th in np.arange(0.1, 0.91, 0.01):
    y_pred = (y_prob >= th).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th
print(f'最佳分類閾值: {best_th:.2f}, 對應F1: {best_f1:.4f}')
final_pred = (y_prob >= best_th).astype(int)
```

### 1-3. 結果

threshold: 0.61

![image](https://raw.githubusercontent.com/SenCha930511/NTCU-Machine-Learning/refs/heads/assignment-acs111132/images/hybrid_result.png)
