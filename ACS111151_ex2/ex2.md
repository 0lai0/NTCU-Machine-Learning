為什麼要用 AutoEncoder + XGBoost？
AutoEncoder 是一種神經網路架構，用來壓縮並還原輸入資料。如果某筆資料「無法被還原得很好」，那可能表示這是異常樣本。

XGBoost 是目前最受歡迎的梯度提升樹模型，對不平衡資料具有良好表現。

結合這兩者：利用 AutoEncoder 偵測異常的能力，為每筆資料生成一個「異常分數」，再加入到 XGBoost 當作額外特徵，使模型能更好地識別詐欺交易。

實作步驟與程式碼解說
載入並預處理資料
df = pd.read_csv("creditcard.csv")
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']
Class 是標籤，0 代表正常交易，1 代表詐欺。
Time 被移除，因為對模型學習幫助不大。

使用 MinMaxScaler 對資料進行正規化：
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

訓練 AutoEncoder
X_normal = X_scaled[y == 0]  # 只用正常樣本

設計一個簡單的 AutoEncoder 結構（中間隱藏層是 16 維）：
input_dim = X_normal.shape[1]
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(16, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_normal, X_normal, epochs=10, batch_size=256, shuffle=True)

計算重建誤差（異常分數）
X_reconstructed = autoencoder.predict(X_scaled)
recon_error = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
這個 recon_error 就是每筆資料與其重建結果的誤差，數值愈大，表示愈可能是異常。

將異常分數加入原始特徵
X_with_score = pd.DataFrame(X_scaled, columns=X.columns)
X_with_score['recon_error'] = recon_error

使用 XGBoost 做分類
X_train, X_test, y_train, y_test = train_test_split(X_with_score, y, test_size=0.2, stratify=y)

model = xgb.XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
scale_pos_weight=10 是為了解決資料不平衡問題，可以根據實際詐欺比例微調。

模型評估
print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))
輸出 Precision、Recall、F1-score 與 AUC 分數，讓你評估模型在詐欺樣本上的準確程度。
