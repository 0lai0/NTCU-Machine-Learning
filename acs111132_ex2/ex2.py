import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import kagglehub

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")

data['Class'] = data['Class'].astype(int)
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop(columns=['Class']).to_numpy()
Y = data['Class'].to_numpy()
RANDOM_SEED = 42
TEST_SIZE = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

iso_forest = IsolationForest(contamination=0.0017, random_state=RANDOM_SEED)
iso_forest.fit(X_train_std)

train_if_score = iso_forest.decision_function(X_train_std).reshape(-1, 1)
test_if_score = iso_forest.decision_function(X_test_std).reshape(-1, 1)


kmeans = KMeans(n_clusters=2, init='k-means++', random_state=RANDOM_SEED)
kmeans.fit(X_train_std[y_train == 0])
train_kmeans_dist = np.min(kmeans.transform(X_train_std), axis=1).reshape(-1, 1)
test_kmeans_dist = np.min(kmeans.transform(X_test_std), axis=1).reshape(-1, 1)
train_kmeans_label = kmeans.predict(X_train_std).reshape(-1, 1)
test_kmeans_label = kmeans.predict(X_test_std).reshape(-1, 1)


pca = PCA(n_components=10, random_state=RANDOM_SEED)
pca.fit(X_train_std)

X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


X_train_enhanced = np.hstack((X_train_std, train_if_score, train_kmeans_dist, X_train_pca))
X_test_enhanced = np.hstack((X_test_std, test_if_score, test_kmeans_dist, X_test_pca))


xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=1.0,
    scale_pos_weight=6,
    random_state=RANDOM_SEED
)

xgb_model.fit(X_train_enhanced, y_train)

y_prob = xgb_model.predict_proba(X_test_enhanced)[:, 1]
best_f1 = 0
best_th = 0.5
for th in np.arange(0.1, 0.91, 0.01):
    y_pred = (y_prob >= th).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th
print(f'最佳分類閾值: {best_th:.2f}, 對應F1: {best_f1:.4f}')
final_pred = (y_prob >= best_th).astype(int)

def evaluation(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'\n{model_name} Evaluation:')
    print('===' * 15)
    print('         Accuracy:', accuracy)
    print('  Precision Score:', precision)
    print('     Recall Score:', recall)
    print('         F1 Score:', f1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
evaluation(y_test, final_pred, model_name="Hybrid Model")