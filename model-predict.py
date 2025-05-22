import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve
import joblib

def evaluation(model_name="Model"):
    # general setting. do not change TEST_SIZE
    RANDOM_SEED = 42
    TEST_SIZE = 0.3
    
    # load dataset（from kagglehub）
    data = pd.read_csv(f"./creditcard.csv")
    data['Class'] = data['Class'].astype(int)

    # prepare data
    data = data.drop(['Time'], axis=1)
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    X = np.asarray(data.iloc[:, ~data.columns.isin(['Class'])])
    Y = np.asarray(data.iloc[:, data.columns == 'Class'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    model = joblib.load("random-forest-SEED-42.pkl")
    # predict and print result
    y_pred = model.predict(X_test)
    # 取得模型預測機率（不是 predict，而是 predict_proba）
    y_scores = model.predict_proba(X_test)[:, 1]  # 第二欄是 positive class 的機率
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_scores)

    print(f'\n{model_name} Evaluation:')
    print('===' * 15)
    print('         Accuracy:', accuracy)
    print('  Precision Score:', precision)
    print('     Recall Score:', recall)
    print('         F1 Score:', f1)
    print('            AUPRC:', auprc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

evaluation()

# plt.figure(figsize=(6, 5))
# plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# plt.grid(True)
# plt.show()
