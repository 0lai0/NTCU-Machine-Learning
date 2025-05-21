## 1. 監督式：XGBoost

### 1-1. 為什麼選擇 XGBoost

我選擇使用 XGBoost 搭配 Randomized Search CV 來尋找最佳參數。

XGBoost 在訓練模型速度時較快，另外對於資料內少數類別的辨識能力，可以透過調整 scale_pos_weight，處理不平衡的問題，進而提高 F1 score。

### 1-2. 程式碼

進行參數選擇和訓練
``` python
# 候選訓練參數（提供給 RandomizedSearchCV 做隨機超參數搜尋）
param_dist = {
    'n_estimators': [100, 200, 300, 400],          # 決策樹的數量（越多越穩定，但訓練時間較久）
    'max_depth': [3, 5, 7, 9],                     # 每棵樹的最大深度（避免過擬合）
    'learning_rate': [0.01, 0.05, 0.1, 0.2],       # 每次新樹對模型貢獻的比例（越小越穩定，越大收斂越快）
    'subsample': [0.6, 0.8, 1.0],                  # 每棵樹訓練時隨機抽樣的樣本比例（控制過擬合）
    'colsample_bytree': [0.6, 0.8, 1.0],           # 建每棵樹時，抽樣使用的特徵比例（降低特徵間共線性）
    'gamma': [0, 0.5, 1, 2],                       # 分裂節點所需的最小損失減少量（越大越保守，抑制過擬合）
    'scale_pos_weight': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]         # 權重平衡，用來處理類別不平衡
}

# 建立 XGBoost 模型，設定初始參數
xgb = XGBClassifier(
    random_state=RANDOM_SEED,     # 固定隨機種子，確保結果可重現
    n_jobs=-1,                    # 使用所有 CPU 核心進行訓練（加快速度）
    eval_metric='logloss'        # 訓練過程中的評估指標設為 logloss
)

# 使用隨機搜尋來自動找出最好的超參數組合
random_search = RandomizedSearchCV(
    estimator=xgb,                    # 要進行搜尋的模型
    param_distributions=param_dist,  # 提供參數組合的範圍
    n_iter=250,                       # 總共會隨機搜尋 250 組參數
    scoring='f1',                     # 以 F1 score 作為搜尋時的評估指標（適合類別不平衡）
    cv=5,                             # 使用 5 折交叉驗證來穩定評估效果
    verbose=2,                        # 顯示詳細的訓練進度
    n_jobs=-1                         # 平行運算加快搜尋速度
)
```

調整評估 threshold

``` python
y_prob = random_search.predict_proba(X_test)[:, 1]

threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

evaluation(y_test, y_pred, "ex1")
```

### 1-3. 結果

![image](https://raw.githubusercontent.com/SenCha930511/NTCU-Machine-Learning/refs/heads/main/images/supervise_result.png)


| 參數             | 數值 |
| ---------------- | ---- |
| subsample        | 0.8  |
| scale_pos_weight | 2.5  |
| n_estimators     | 200  |
| max_depth        | 6    |
| learning_rate    | 0.1  |
| gamma            | 0.5  |
| colsample_bytree | 1.0  |


## 2. 非監督式: Kmeans

### 2-1. 程式

將正常和異常的資料一起訓練合併成訓練資料，並自動搜尋最佳 k。

```python
# 從訓練資料中抽樣資料（平衡一下比例）
n_x_train = x_train[y_train == 0][:1000]   # 抽取 1000 筆正常資料 (class=0)
f_x_train = x_train[y_train == 1][:100]    # 抽取 100 筆異常資料 (class=1)
mix_x_train = np.vstack([n_x_train, f_x_train])  # 合併成新的訓練資料（進行分群）

# 設定欲測試的 K 值範圍（群數從 2 到 10）
K_RANGE = range(2, 11)
scores = []  # 儲存每個 K 對應的 Silhouette score

# 針對每個群數 k 訓練 KMeans，並計算 Silhouette 分數
for k in K_RANGE:
    kmeans = KMeans(
        n_clusters=k,             # 指定群數 k
        init='k-means++',         # 初始化方法（選擇較穩定的初始點）
        n_init=20,                # 初始化嘗試次數（選擇最佳結果）
        max_iter=700,             # 最多迭代次數
        tol=1e-4,                 # 收斂容忍度
        random_state=RANDOM_SEED,
    )
    kmeans.fit(mix_x_train)                          # 執行 KMeans 分群
    score = silhouette_score(mix_x_train, kmeans.labels_)  # 計算 Silhouette 分數（分群品質指標）
    scores.append(score)                             # 加入分數列表

# 找出 Silhouette 分數最高時對應的 k 值
optimal_k = np.argmax(scores) + K_RANGE.start  # 加回 range 起始值，找出最佳群數

# 以最佳 k 值重新訓練一次 KMeans 模型
kmeans = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    n_init=20,             # 提高初始化次數（提升穩定性）
    max_iter=700,          # 提高最大迭代次數
    tol=1e-4,
    random_state=RANDOM_SEED,
)
kmeans.fit(mix_x_train)     # 用混合樣本訓練模型
```

### 2-2. 結果

k 設定值：2

![image](https://raw.githubusercontent.com/SenCha930511/NTCU-Machine-Learning/refs/heads/main/images/unsupervise_result.png)

