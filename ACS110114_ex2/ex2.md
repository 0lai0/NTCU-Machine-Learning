# 挑戰二：融合監督與非監督學習 - 信用卡詐欺檢測

## 技術架構

### 1. 非監督學習：Isolation Forest
- **目的**：識別數據中的異常模式，無需標籤信息
- **原理**：通過隨機選擇特徵和分割點來構建決策樹，異常點更容易被隔離
- **優勢**：能夠發現未知的異常模式，不依賴於標籤數據

### 2. 監督學習：深度神經網路
- **架構**：多層全連接網路 (256 → 128 → 64 → 32 → 1)
- **激活函數**：ReLU + Sigmoid
- **正則化**：Batch Normalization + Dropout (0.3)
- **優勢**：能夠學習複雜的非線性關係

### 3. 融合策略
- 將 Isolation Forest 的異常分數作為額外特徵
- 與原始特徵組合形成增強特徵集
- 利用深度神經網路進行最終分類

## 實現細節

### 數據預處理
```python
# 標準化 Amount 欄位
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# 移除 Time 欄位（時間資訊對模型意義不大）
data = data.drop(['Time'], axis=1)
```

### Isolation Forest 配置
```python
isolation_forest = IsolationForest(
    contamination=0.1,      # 預期異常比例 10%
    random_state=42,        # 確保結果可重現
    n_estimators=100,       # 使用 100 棵樹
    max_samples='auto',     # 自動選擇樣本數
    n_jobs=-1              # 使用所有 CPU 核心
)
```

### 深度神經網路架構
- **輸入層**：30 個特徵（29個原始特徵 + 1個異常分數）
- **隱藏層**：256 → 128 → 64 → 32 神經元
- **輸出層**：1 個神經元（詐欺概率）
- **正則化**：BatchNorm + Dropout 防止過擬合

### 類別不平衡處理
```python
# 計算類別權重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
pos_weight = torch.tensor([class_weights[1]/class_weights[0]])

# 使用帶權重的 BCE 損失
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

## 精確率改善策略

### 問題診斷
原始結果顯示精確率極低（0.0018），表明模型過度預測正類（詐欺交易），導致大量假陽性。

### 改善方案

#### 1. 調整類別權重
```python
# 降低正類權重，避免過度預測
pos_weight = torch.tensor([class_weights[1]/class_weights[0] * 0.1])
```

#### 2. 優化 Isolation Forest 參數
```python
isolation_forest = IsolationForest(
    contamination=0.02,    # 降低至2%，更貼近實際詐欺比例
    n_estimators=200,      # 增加樹的數量提高穩定性
    max_features=0.8,      # 使用80%特徵，增加多樣性
)
```

#### 3. 動態閾值優化
- 使用 PR 曲線找到最佳 F1 閾值
- 提供多種策略：高精確率、平衡策略、最佳F1
- 根據業務需求選擇合適閾值

#### 4. 模型結構簡化
- 減少隱藏層神經元數量防止過擬合
- 調整 Dropout 比例
- 增加梯度裁剪防止梯度爆炸

#### 5. 訓練策略改善
- 增加早停機制
- 使用更強的正則化
- 在驗證過程中使用動態閾值

## 模型優勢

### 1. 異常檢測增強
- Isolation Forest 能夠識別數據中的異常模式
- 異常分數提供了額外的判別信息
- **改善**：降低 contamination 參數，減少假陽性

### 2. 深度學習優勢
- 能夠學習複雜的非線性特徵組合
- 自動特徵提取和表示學習
- **改善**：簡化模型結構，防止過擬合

### 3. 動態閾值優化
- **新增**：不再使用固定的 0.5 閾值
- 基於 PR 曲線找到最佳分類閾值
- 提供多種策略供選擇

### 4. GPU 加速
- 使用 PyTorch 框架支援 GPU 計算
- 大幅提升訓練和推理速度
- 適合處理大規模數據集

### 5. 不平衡數據處理
- **改善**：調整類別權重，避免過度預測
- 關注 F1 分數而非僅僅準確率
- 提供精確率和召回率的平衡選擇

## 實驗設置

### 參數配置
- **隨機種子**：42（確保結果可重現）
- **測試集比例**：30%
- **批次大小**：512
- **學習率**：0.001
- **訓練回合**：100
- **早停機制**：基於 F1 分數


## 預期改進

相比於單純使用 Random Forest 或 KMeans：

1. **特徵增強**：異常分數提供了額外的判別信息
2. **模型複雜度**：深度神經網路能學習更複雜的模式
3. **異常檢測**：Isolation Forest 專門針對異常檢測設計
4. **平衡處理**：更好地處理類別不平衡問題
5. **計算效率**：GPU 加速提供更快的訓練速度

## 執行要求

### GPU 支援
- 自動檢測並使用可用的 GPU
- 如無 GPU 則自動使用 CPU（速度較慢）