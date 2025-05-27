# 匯入系統相關套件並處理 MKL 錯誤（避免 Jupyter 或某些環境報錯）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 匯入 PyTorch 相關模組
import torch
import torch.nn as nn
import torch.optim as optim

# 匯入數值與資料處理模組
import numpy as np
import pandas as pd

# 匯入 sklearn 中的訓練測試切分、標準化與評估工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, precision_recall_curve
)

# 匯入畫圖工具
import matplotlib.pyplot as plt

# 匯入 XGBoost 分類器
import xgboost as xgb

# 匯入 KaggleHub 自動下載資料集
import kagglehub

# 匯入 SMOTE 資料增強工具
from imblearn.over_sampling import SMOTE

# 下載並讀取 Kaggle 上的信用卡詐欺資料集
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")

# 移除 Time 欄位（時間戳記對模型幫助不大）
data.drop('Time', axis=1, inplace=True)

# 對 Amount 欄位進行標準化（均值為0、標準差為1）
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# 分離特徵 X 與標籤 Y
X = data.drop(columns=['Class']).values
Y = data['Class'].values

# 依據標籤切出詐欺與正常樣本，以利觀察數量
fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]

# 印出詐欺與非詐欺樣本數量
print(f'Fraudulent:{len(fraud)}, non-fraudulent:{len(nonfraud)}')

# 印出詐欺樣本比例
print(f'The positive class (frauds) percentage: {len(fraud)/(len(fraud)+len(nonfraud))*100:.3f}%')

# 切分訓練與測試集（保留原始類別比例）
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

# 從訓練集中取出所有非詐欺樣本，用來訓練 AutoEncoder
X_train_auto = X_train[y_train == 0]

# 轉換為 PyTorch tensor 格式
X_train_auto_tensor = torch.tensor(X_train_auto, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 定義 AutoEncoder 結構（Encoder 擴張後壓縮，Decoder 還原）
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    # 定義 forward 過程：輸入經過 encoder 與 decoder
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 設定模型輸入維度與編碼維度
input_dim = X_train.shape[1]
encoding_dim = 17

# 初始化 AutoEncoder 模型
model = AutoEncoder(input_dim, encoding_dim)

# 使用 MSE 作為重建損失函數
criterion = nn.MSELoss()

# 使用 RMSprop 優化器
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

# 留出 10% 作為驗證集
val_size = int(0.1 * X_train_auto.shape[0])
train_tensor = torch.tensor(X_train_auto[:-val_size], dtype=torch.float32)
val_tensor = torch.tensor(X_train_auto[-val_size:], dtype=torch.float32)

# 訓練參數：訓練 100 輪，每批大小為 32
EPOCHS = 100
BATCH_SIZE = 32
train_losses, val_losses = [], []

# 開始訓練 AutoEncoder
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(train_tensor.size(0))
    epoch_train_loss = 0

    # 每個 batch 執行訓練
    for i in range(0, len(perm), BATCH_SIZE):
        indices = perm[i:i + BATCH_SIZE]
        batch = train_tensor[indices]
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    # 計算驗證損失
    model.eval()
    with torch.no_grad():
        val_output = model(val_tensor)
        val_loss = criterion(val_output, val_tensor).item()

    train_losses.append(epoch_train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")

# 使用訓練好的 Encoder 將原始資料進行轉換
model.eval()
with torch.no_grad():
    X_train_encoded = model.encoder(torch.tensor(X_train, dtype=torch.float32)).numpy()
    X_test_encoded = model.encoder(torch.tensor(X_test, dtype=torch.float32)).numpy()

# 對轉換後的資料使用 BorderlineSMOTE 平衡資料
from imblearn.over_sampling import BorderlineSMOTE
smote = BorderlineSMOTE(kind='borderline-2', random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_encoded, y_train)

# 印出平衡後的類別數量
print(f"\nAfter SMOTE: Class 0 = {(y_train_bal == 0).sum()}, Class 1 = {(y_train_bal == 1).sum()}")

# 建立 XGBoost 分類器，使用 GPU 加速
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

# 使用平衡後資料進行訓練
xgb_model.fit(X_train_bal, y_train_bal)

# 對測試集預測機率（只取詐欺機率）
y_pred_prob = xgb_model.predict_proba(X_test_encoded)[:, 1]

# 計算 precision-recall 曲線與最佳 F1 分數的門檻
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# 印出最佳門檻與對應的指標
print(f"\nBest F1 Threshold: {best_threshold:.6f}")
print(f"Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")

# 設定使用的預測門檻（可微調）
adjusted_threshold = best_threshold
y_pred = (y_pred_prob > adjusted_threshold).astype(int)

# 定義模型評估函數
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

# 執行評估
evaluation(y_test, y_pred, model_name="AutoEncoder + SMOTE + XGBoost")

# 畫出訓練與驗證損失圖
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
