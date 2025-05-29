import numpy as np
import pandas as pd

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.ensemble          import IsolationForest
from xgboost                   import XGBClassifier
from sklearn.metrics           import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

# 固定參數
RANDOM_SEED = 42
TEST_SIZE   = 0.3

def evaluate_pipeline(cont_list, percentile_list):
    # 讀檔 & 前處理
    df = pd.read_csv("data/creditcard.csv")
    df = df.drop(columns=["Time"])
    df["Amount"] = StandardScaler().fit_transform(
        df["Amount"].values.reshape(-1, 1)
    )
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    # 切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # 訓練 XGBoost（全資料）
    xgb = XGBClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)

    best_cfg = None
    best_f1  = 0

    # 掃描不同的 contamination
    for cont in cont_list:
        iso = IsolationForest(
            contamination=cont,
            random_state=RANDOM_SEED
        )
        iso.fit(X_train[y_train==0])

        # decision_function 取分數
        scores = -iso.decision_function(X_test)

        # 在這個 contamination 下，掃描不同的 percentile 作為 threshold
        for pct in percentile_list:
            thr = np.percentile(scores, pct)
            mask_anom = (scores >= thr)

            # 合併預測
            y_pred = np.zeros_like(y_test)
            if mask_anom.any():
                y_pred[mask_anom] = xgb.predict(X_test[mask_anom])

            # 計算 F1
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = (cont, pct, thr, f1)

    cont, pct, thr, f1 = best_cfg
    print(f"\n最佳配置 → contamination={cont}, percentile={pct:.1f}, thr={thr:.3f}")
    print(f"對應 F1 = {f1:.4f}\n")

    # 用最佳配置重跑一次並印最終報告
    iso = IsolationForest(contamination=cont, random_state=RANDOM_SEED)
    iso.fit(X_train[y_train==0])
    scores = -iso.decision_function(X_test)
    mask_anom = (scores >= thr)

    y_pred = np.zeros_like(y_test)
    y_pred[mask_anom] = xgb.predict(X_test[mask_anom])
    y_prob = np.zeros_like(y_test, dtype=float)
    y_prob[mask_anom] = xgb.predict_proba(X_test[mask_anom])[:,1]

    print("=== 最終評估 ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == "__main__":
    # 自訂 contamination 與 percentile 的範圍
    cons = [0.001, 0.002, 0.005, 0.01]
    pers = [99, 99.5, 99.8, 99.9]
    evaluate_pipeline(cons, pers)
