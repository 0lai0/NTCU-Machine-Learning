# Re-import necessary modules after kernel reset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt

# 資料預處理
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data.drop('Time', axis=1, inplace=True)
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
X = data.drop(columns=['Class']).values
Y = data['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
X_train_auto = X_train[y_train == 0]

# 轉 tensor
X_train_auto_tensor = torch.tensor(X_train_auto, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# AutoEncoder 模型
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        hidden_dim = encoding_dim // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Tanh(),
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, encoding_dim),
            nn.Tanh(),
            nn.Linear(encoding_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 設定
input_dim = X_train.shape[1]
encoding_dim = 17
model = AutoEncoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

val_size = int(0.1 * X_train_auto.shape[0])
train_data = X_train_auto[:-val_size]
val_data = X_train_auto[-val_size:]

# 轉換為 Tensor
train_tensor = torch.tensor(train_data, dtype=torch.float32)
val_tensor = torch.tensor(val_data, dtype=torch.float32)

EPOCHS = 100
BATCH_SIZE = 64
train_losses, val_losses = [], []

# 🔁 訓練迴圈
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(train_tensor.size(0))
    epoch_train_loss = 0

    for i in range(0, len(perm), BATCH_SIZE):
        indices = perm[i:i + BATCH_SIZE]
        batch = train_tensor[indices]
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    # 驗證階段
    model.eval()
    with torch.no_grad():
        val_output = model(val_tensor)
        val_loss = criterion(val_output, val_tensor).item()

    # 記錄
    avg_train_loss = epoch_train_loss / (len(perm) // BATCH_SIZE)
    train_losses.append(epoch_train_loss)
    val_losses.append(val_loss)

    # ✅ 顯示進度
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")
with torch.no_grad():
    X_test_pred = model(X_test_tensor).numpy()
mse = np.mean((X_test - X_test_pred) ** 2, axis=1)

# 評估函數
def evaluation(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{model_name} Evaluation:")
    print("===" * 15)
    print("         Accuracy:", accuracy)
    print("  Precision Score:", precision)
    print("     Recall Score:", recall)
    print("         F1 Score:", f1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, mse)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"\nBest F1 Threshold: {best_threshold:.6f}")
print(f"Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")

# 📊 分類結果
y_pred = (mse > best_threshold).astype(int)
print(best_threshold)

evaluation(y_test, y_pred, model_name=f"AutoEncoder)")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AutoEncoder Loss (Train )")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()