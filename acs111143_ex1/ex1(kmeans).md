## 1. 用KMeans訓練模型

### 1 訓練資料篩選
- 使用正常(non-fraud)資料進行 KMeans 訓練  
- 取前 1000 筆正常資料以降低計算成本  

### 2 參數調整
```python
kmeans = KMeans(
    n_clusters=optimal_k,       # 使用輪廓係數決定的最佳分群數
    init='k-means++',           # 智能初始化群集中心，提升穩定性
    n_init=20,                  # 多次初始化選最優，避免局部最佳解
    max_iter=500,               # 設定單次運行的最大迭代次數
    tol=1e-4,                   # 設定收斂的容忍度
    random_state=RANDOM_SEED,   # 固定隨機種子以確保結果可重現 (RANDOM_SEED=42)
    algorithm='elkan'           # 使用 Elkan 演算法提升特定情況下的計算效率
)
```
---

## 3. 評估結果

| 指標         | 值                   |
|--------------|----------------------|
| Accuracy     | 0.9987               |
| Precision    | 0.783                |
| Recall       | 0.365                |
| F1 Score     | 0.498                |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 85295   |
| 1     | 0.78      | 0.36   | 0.50     | 148     |

---

## 4. Summary

雖然它能夠識別出部分詐欺模式（體現在尚可的精確率上），但其召回率相對較低，表明有較多詐欺案件被遺漏。在實際應用中，可能需要探索更複雜的異常偵測技術，或結合監督式學習模型 
