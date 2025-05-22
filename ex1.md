
# Machine Learning 作業報告 (ex1)


## 1-1監督式作業

  

### 1. 方法

使用 XGBoost 分類器（XGBClassifier）進行二元分類，直接利用正、負樣本標籤學習「正常／詐欺」的區分邊界。



### 2. 資料前處理

 **讀取資料**

-  `creditcard.csv`，共 284,807 筆交易，其中正常 ≈ 284,315 筆，詐欺 492 筆。



  **訓練／測試切分**

```python

x_tr, x_te, y_tr, y_te = train_test_split(

X, y, test_size=0.30, stratify=y, random_state=42

)

```

保持詐欺比例一致。

  

### 模型訓練

```python

from xgboost import XGBClassifier

  

xgb_model = XGBClassifier(

n_estimators=325,

max_depth=7,

learning_rate=0.066,

subsample=0.95,

colsample_bytree=0.85,

scale_pos_weight=7.5,

random_state=RANDOM_SEED,

use_label_encoder=False,

eval_metric='logloss',

tree_method='hist'

  

)
```
# 主要調整測試

主要是estimator數調整:在300~400之間，我自己實測是325到350差不多，接近400開始掉 
  
以及depth:我7的時候數值來到最好，8以後沒變動或走下坡

learning rate:測試到0.01到0.08區間，我在0.06跟0.07之間有達到很好的表現



  

## 1-2非監督作業

  

### 1. 方法

採用 **Denoising Auto-Encoder (DAE)**：

1. 只使用正常樣本 (Class=0) 進行訓練。

2. 輸入先加入高斯雜訊與 Dropout，讓模型學習在雜訊與隨機失活下重建正常行為。

3. 訓練完成後，計算所有樣本的重建誤差 MSE，並以訓練集重建誤差的**99.9%**百分位作為門檻。

4. 測試集 MSE 大於門檻即標為異常 (fraud)，否則為正常。

  

### 2. 資料前處理

與監督式相同：

- 刪除 `Time`

-  `Amount` 標準化

-  `V1`–`V28` 直接使用

- 切分及標準化保持一致

  

### 3. DAE 架構與訓練

```python

# Hyper-parameters

ENC_UNITS = [64,32,16]

DROPOUT_RATE = 0.2

NOISE_STD = 0.05

EPOCHS, BATCH = 70, 256

  

# 建立模型

inp = tf.keras.Input(shape=(x_tr.shape[1],))

x = tf.keras.layers.GaussianNoise(NOISE_STD)(inp)

for u in ENC_UNITS[:-1]:

x = tf.keras.layers.Dense(u, activation='relu')(x)

x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

encoded = tf.keras.layers.Dense(ENC_UNITS[-1], activation='relu')(x)

x = tf.keras.layers.Dense(ENC_UNITS[-2], activation='relu')(encoded)

x = tf.keras.layers.Dense(ENC_UNITS[-3], activation='relu')(x)

out = tf.keras.layers.Dense(x_tr.shape[1], activation='linear')(x)

dae = tf.keras.Model(inp, out)

dae.compile(optimizer='adam', loss='mse')

  

# 訓練

dae.fit(x_tr_norm, x_tr_norm,

epochs=EPOCHS, batch_size=BATCH,

shuffle=True, verbose=0)

```


### 4. 門檻選擇與評估

```python

# 計算訓練與測試 MSE

mse_tr = np.mean((dae.predict(x_tr) - x_tr)**2, axis=1)

mse_te = np.mean((dae.predict(x_te) - x_te)**2, axis=1)

  

# 門檻設定

threshold = np.percentile(mse_tr, 99.9)


```
# 主要調整測試

主要是調整threshold: 我從98.0一路測試到99.9
有時候99.8會比99.9表現來的好我也不知道為甚麼
  
編碼器（Encoder）與解碼器（Decoder）的隱藏層大小 [64,32,16] 我也有測試一下ae 用[32,8]但結果沒有比較好

在colab測試上dae表現比ae好

以及epoch 30 ,70 ,100 中，70算是時間與效果最好的epoch

雖然precision比kmeans有下降一些，但其他三項在colab測試後是有優於簡報內範例kmeans程式碼
  

