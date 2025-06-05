<Random Forest 模型參數調整>
n_estimators=125
    設定樹的數量為 125，平衡準確度與執行時間

max_depth=20
    限制樹的最大深度，避免過度擬合

min_samples_leaf=2
    每個葉節點至少包含 2 筆樣本，提高模型穩定性

class_weight='balanced'
    自動調整類別權重，處理詐欺樣本數極少的不平衡問題

random_state=42
    固定隨機種子，確保結果可重現

<Isolation Forest 模型參數調整>
n_estimators=1000
    使用較多樹增加異常偵測的穩定性

contamination='auto'
    自動估計資料中的異常比例，影響分類的門檻

random_state=42
    固定隨機性，保持結果一致

threshold = np.percentile(test_scores, 97)
    設定異常分數的第 97 百分位作為詐欺判斷門檻，提高 recall（偵測率）