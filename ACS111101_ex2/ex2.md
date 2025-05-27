# 非監督式+監督式 - isolation forest + XGBoost
>* **最佳實驗結果**
```python
from imblearn.over_sampling import SMOTE

# Extract features and labels
X = np.asarray(data.drop(columns=['Class']))
y = np.asarray(data['Class'])

# Split the dataset into training and testing sets (with stratification)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# PCA 降維
pca = PCA(n_components=25)  # 嘗試 2-20 的不同值
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Step 1: 用全部交易訓練 Isolation Forest
iso = IsolationForest(
    contamination=0.0017, 
    random_state=RANDOM_SEED,
    n_estimators=600,
    max_features=15,
    bootstrap=True
    )
iso.fit(x_train)

# Step 2: 用全部資料做預測（正常/異常）
iso_labels = iso.predict(x_train)
iso_labels = (iso_labels == -1).astype(int)  # -1 是異常，轉成 1

x_train = np.hstack([x_train, iso_labels.reshape(-1, 1)])
iso_pred_test = iso.predict(x_test)
iso_feature_test = (iso_pred_test == -1).astype(int)
x_test = np.hstack((x_test, iso_feature_test.reshape(-1, 1)))

xg_model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=550,
    scale_pos_weight=6,
    max_depth=7,
    random_state=RANDOM_SEED,
)

xg_model.fit(x_train, y_train)

# Step 5: 預測測試集
threshold = 0.4
y_proba = xg_model.predict_proba(x_test)[:, 1]  # 取得預測為正類（詐欺）的機率
y_pred = (y_proba > threshold).astype(int)
```
![alt text](image-15.png)
> 相較於範例
Accuracy：增加約0.00003
Precision：增加約0.01
Recall：增加約0.007
F1：增加約0.01

> * 實驗方法&心得
在實驗過程中我嘗試使用AutoEncoder、lightGBM實驗，但結果都沒有isolation forest+XGBoost的結果來得好，isolation forest前我先使用PCA降維，刪除掉一些不必要的特徵值，再用isolation forest並將訓練結果作為一個特徵值加到training data中，接著利用監督式學習，透過n_estimators、scale_pos_weight的調整，讓模型更複雜但不至於overfitting

* **實驗過程**：
    *  實驗1：xgboost `n_estimators=200`
        ![alt text](image-1.png)
    *  實驗2：xgboost `max_depth=5`
        ![alt text](image-2.png)
    *  實驗3：isolation forest `contamination=0.0017`
        ![alt text](image-5.png)
    *  實驗4：不使用`stratify=y`
        ![alt text](image-8.png)
    *   實驗5：xgboost`scale_pos_weight=6`
        ![alt text](image-10.png)
    *   實驗6：xgboost`max_depth=7`
        ![alt text](image-12.png)
    *   實驗7：isolation`max_features=15`
        ![alt text](image-14.png)
    *   實驗8：threshold=0.4
        ![alt text](image-17.png)

