# 信用卡詐欺偵測：監督式 XGBoost 與非監督式 K‑Means 方法

1 專案背景
Kaggle:Credit Card Fraud Detection資料集中，詐欺僅佔約0.17%。在極度不平衡的情況下，同時評估監督式與非監督式模型，可滿足：
有標籤場景：離線訓練 → 線上部署（XGBoost）。
無標籤場景：新市場或早期系統尚未蒐集到足夠詐欺標籤（K‑Means）。
四大指標：Accuracy／Precision／Recall／F1 Score。

2-1 監督式：XGBoost

特性               | 說明                                                            |
|梯度提升樹 (GBDT)  | 擅長捕捉非線性與高階交互特徵。                                    |
|內建處理不平衡     | 透過 `scale_pos_weight` 直接調整少數類權重。                      |
|高效 + 可解釋      | `tree_method="hist"` 加速，feature importance 可用於後續風控審查。|
|成熟生態          | 滿足 MLOps 部署、GPU 加速、交叉驗證。                              |

2‑2 主要參數與設定理由

| 參數值                                       |原因                                                                           
| `max_depth = 6`                             | 適度深度，避免過度擬合，小幅降低 model size。                                             
| `learning_rate = 0.1`                       | 與 `n_estimators=200` 搭配，取得 bias‑variance 折衷。                                
| `n_estimators = 200`                        | 維持充足複雜度；若增減此值需相應調整 learning rate。                                           
| `subsample = 0.8`, `colsample_bytree = 1.0` | Bagging 效果，減少 correlation，提升泛化。                                             
| `scale_pos_weight = 2.5`                    | 依據 $\frac{#\text{neg}}{#\text{pos}}\times0.5$ 經驗設定，提升 Recall 同時維持 Precision。 
| `tree_method = "hist"`                      | 使用直方圖演算法，加速 CPU 訓練。                                                       
| 閾值 `threshold = 0.43`                     | 透過 Precision–Recall 曲線找出 F1 最佳點；將預設 0.5 下調以提升 Recall。                      

2‑3  效果

```
Accuracy  ≈ 0.9997
Precision ≈ 0.944
Recall    ≈ 0.860
F1‑score  ≈ 0.900
```
---

3-1  非監督式：K‑Means


| 特性         | 說明                                             |
| **無須標籤** | 可在資料冷啟動階段使用；僅用正常樣本即可建立基線。    |
| **計算快速** | $\mathcal O(n k d)$；適合作為即時異常偵測前置篩選。 |
| **易於解釋** | 以群中心距離做風控規則（business‑rule friendly）。   |

3‑2  流程與參數

| 步驟 / 參數                  | 值                 | 理由                                                             |
| **訓練資料**                 | 前 1000 筆正常交易  | 快速估計正常分布，避免異常混入。                                   |
| **`n_clusters` 搜尋範圍**    | 2 – 4              | 詐欺行為罕見，過多 k 易將正常拆散。以 **Silhouette score** 選最佳。 |
| **`init = "k-means++"`** |  |                    | 改善初始質心品質，加速收斂。                                       |
| **距離閾值法** (可選)        | 95 percentile      | 以最小群心距離為異常分數，動態調整 Precision/Recall。               |

3‑3  效果（對齊群標籤）

```
Accuracy  ≈ 0.9987
Precision ≈ 0.783
Recall    ≈ 0.365
F1‑score  ≈ 0.498
```
---

小結

* **XGBoost** 透過 `scale_pos_weight` 與閾值調整，在 Precision 與 Recall 取得最佳 F1。
* **K‑Means** 提供零標籤情境的快速 baseline，可透過距離閾值進一步調優。
* 兩者可依資料標籤成熟度與業務風險容忍度彈性選用，或進行集成以獲得更穩健的詐欺偵測。

