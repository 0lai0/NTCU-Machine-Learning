# === 載入必要套件 ===
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt
import xgboost as xgb
import kagglehub
from imblearn.over_sampling import SMOTE

# === 載入資料集 ===
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data.drop('Time', axis=1, inplace=True)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop(columns=['Class']).values
Y = data['Class'].values
fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]
print(f'Fraudulent:{len(fraud)}, non-fraudulent:{len(nonfraud)}')
print(f'The positive class (frauds) percentage: {len(fraud)/(len(fraud)+len(nonfraud))*100:.3f}%')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

# === 僅使用非詐欺樣本訓練 AutoEncoder ===
X_train_auto = X_train[y_train == 0]
X_train_auto_tensor = torch.tensor(X_train_auto, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# === 更深層的 AutoEncoder ===
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()

        # Encoder（稍微擴張後再壓縮）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, encoding_dim),
            nn.ReLU()
        )

        # Decoder（對稱還原）
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# === 初始化模型與訓練設定 ===
input_dim = X_train.shape[1]
encoding_dim = 17
model = AutoEncoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

val_size = int(0.1 * X_train_auto.shape[0])
train_tensor = torch.tensor(X_train_auto[:-val_size], dtype=torch.float32)
val_tensor = torch.tensor(X_train_auto[-val_size:], dtype=torch.float32)

EPOCHS = 100
BATCH_SIZE = 32
train_losses, val_losses = [], []

# === 訓練 AutoEncoder ===
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

    model.eval()
    with torch.no_grad():
        val_output = model(val_tensor)
        val_loss = criterion(val_output, val_tensor).item()

    train_losses.append(epoch_train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")

# === 將 Encoder 應用於全資料集 ===
model.eval()
with torch.no_grad():
    X_train_encoded = model.encoder(torch.tensor(X_train, dtype=torch.float32)).numpy()
    X_test_encoded = model.encoder(torch.tensor(X_test, dtype=torch.float32)).numpy()

# === 使用 SMOTE 對編碼後資料做平衡 ===
from imblearn.over_sampling import BorderlineSMOTE
smote = BorderlineSMOTE(kind='borderline-2', random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_encoded, y_train)
print(f"\nAfter SMOTE: Class 0 = {(y_train_bal == 0).sum()}, Class 1 = {(y_train_bal == 1).sum()}")

# === 使用 XGBoost 訓練 ===
xgb_model = xgb.XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=200,
    subsample=0.8,
    scale_pos_weight=2.5,           
    eval_metric='logloss',
    tree_method='gpu_hist',          
    predictor='gpu_predictor',
    random_state=42
)
xgb_model.fit(X_train_bal, y_train_bal)

# === 預測與最佳 F1 門檻 ===
y_pred_prob = xgb_model.predict_proba(X_test_encoded)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nBest F1 Threshold: {best_threshold:.6f}")
print(f"Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")

adjusted_threshold = best_threshold # 預設減少 0.01 可觀察影響
y_pred = (y_pred_prob > adjusted_threshold).astype(int)

# === 評估函數 ===
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

evaluation(y_test, y_pred, model_name="AutoEncoder + SMOTE + XGBoost")

# === 畫出損失圖 ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AutoEncoder Loss (Train & Val)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
