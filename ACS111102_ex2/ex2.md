<Isolation Forest 參數調整>
n_estimators=200：設定建構 200 棵樹，提升異常偵測穩定性
contamination=0.0017：明確設定資料中異常（詐欺）比例約 0.17%
random_state=42：固定隨機性以重現結果


<PCA 參數調整>
n_components=12：保留前 12 個主成分，用來濃縮高維特徵資訊，減少雜訊


<XGBoost 分類器參數調整>
n_estimators=200：使用 200 棵樹進行訓練
max_depth=6：限制樹的深度避免過擬合
learning_rate=0.1：學習率設定為 0.1，平衡學習速度與精度
subsample=0.6：每棵樹隨機取用 60% 訓練資料，增加多樣性
colsample_bytree=1.0：每棵樹使用所有特徵（未進行降維）
min_child_weight=5：限制節點最小權重，避免過擬合
gamma=0.5：需達到的最小損失減少量才允許節點分裂
alpha=0.1：L1 正規化，防止過擬合
scale_pos_weight=3：加重少數類（詐欺）樣本的重要性
tree_method='hist'：使用直方圖加速訓練
eval_metric='aucpr'：以 AUC-PR 作為評估指標，更適合不平衡資料