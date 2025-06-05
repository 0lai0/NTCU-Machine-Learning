import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_curve
import kagglehub

# === 資料處理 ===
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(f"{path}/creditcard.csv")
df.drop("Time", axis=1, inplace=True)
df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))

X = df.drop("Class", axis=1).values
Y = df["Class"].values
print(f'Fraudulent:{sum(Y)}, non-fraudulent:{len(Y)-sum(Y)}')
print(f'the positive class (frauds) percentage: {sum(Y)/len(Y)*100:.3f}%')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
X_train_auto = X_train[y_train == 0]

# === 分出驗證集 ===
val_size = int(0.1 * len(X_train_auto))
train_data = X_train_auto[:-val_size]
val_data = X_train_auto[-val_size:]

# === 模型架構（Keras AutoEncoder）===
input_dim = X.shape[1]
encoding_dim = 17

input_layer = tf.keras.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='tanh')(input_layer)
encoded = tf.keras.layers.Dense(encoding_dim // 2, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(encoding_dim, activation='tanh')(encoded)
decoded = tf.keras.layers.Dense(input_dim)(decoded)

autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# === 訓練 ===
history = autoencoder.fit(
    train_data, train_data,
    epochs=100,
    batch_size=64,
    shuffle=True,
    validation_data=(val_data, val_data),
    verbose=1
)

# === 重建測試資料 ===
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# === 找出最佳 F1 的門檻 ===
prec, rec, thresholds = precision_recall_curve(y_test, mse)
f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"\nBest F1 Threshold: {best_threshold:.6f}")
print(f"Precision: {prec[best_idx]:.4f}, Recall: {rec[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")

# === 評估 ===
y_pred = (mse > best_threshold).astype(int)
print(best_threshold)

def evaluation(y_true, y_pred, model_name="Model"):
    print(f"\n{model_name} Evaluation:")
    print("===" * 15)
    print("         Accuracy:", accuracy_score(y_true, y_pred))
    print("  Precision Score:", precision_score(y_true, y_pred, zero_division=0))
    print("     Recall Score:", recall_score(y_true, y_pred))
    print("         F1 Score:", f1_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

evaluation(y_test, y_pred, model_name="AutoEncoder")

# === 畫損失圖 ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AutoEncoder Loss (Train/Val)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
