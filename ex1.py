import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from imblearn.over_sampling    import SMOTE
from sklearn.ensemble          import RandomForestClassifier
from sklearn.cluster           import KMeans
from sklearn.metrics           import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 固定參數
RANDOM_SEED = 42
TEST_SIZE   = 0.3

def supervised_pipeline(X_train, X_test, y_train, y_test):
    """監督式：SMOTE + RandomForest"""
    sm = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
    clf.fit(X_res, y_res)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    print("\n--- 監督式學習：SMOTE + RandomForest ---")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

def unsupervised_pipeline(X_all, y_all):
    """非監督式：KMeans(k=3) 異常偵測"""
    # 全資料標準化
    X_scaled = StandardScaler().fit_transform(X_all)

    k = 3
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED).fit(X_scaled)
    labels = km.labels_

    # 群內多數標籤當預測
    y_pred = np.zeros_like(labels)
    for c in range(k):
        mask = (labels == c)
        majority = pd.Series(y_all[mask]).mode()[0]
        y_pred[mask] = majority

    print("\n--- 非監督式學習：KMeans (k=3) ---")
    print(classification_report(y_all, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_all, y_pred))

def main():
    # 1. 讀檔 & 前處理
    data = pd.read_csv("data/creditcard.csv")
    data = data.drop(columns=['Time'])
    data['Amount'] = StandardScaler().fit_transform(
        data['Amount'].values.reshape(-1,1)
    )

    X = data.drop(columns=['Class']).values
    y = data['Class'].values

    # 2. 切 supervised 的 train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # 3. 執行監督式流程
    supervised_pipeline(X_train, X_test, y_train, y_test)

    # 4. 執行非監督式流程（用全部資料評估）
    unsupervised_pipeline(X, y)

if __name__ == "__main__":
    main()
