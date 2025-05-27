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

# general setting. do not change TEST_SIZE
RANDOM_SEED = 42
TEST_SIZE = 0.3

# load dataset（from kagglehub）
data = pd.read_csv(f"./creditcard.csv")
data['Class'] = data['Class'].astype(int)

# prepare data
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]
print(f'Fraudulent:{len(fraud)}, non-fraudulent:{len(nonfraud)}')
print(f'the positive class (frauds) percentage: {len(fraud)}/{len(fraud) + len(nonfraud)} ({len(fraud)/(len(fraud) + len(nonfraud))*100:.3f}%)')    

# Extract features and labels
X = np.asarray(data.drop(columns=['Class']))
y = np.asarray(data['Class'])

# Split the dataset into training and testing sets (with stratification)
x_train, x_test, y_train, y_test = train_test_split(
   X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Select a small sample of normal (non-fraud) data for unsupervised training
# n_x_train = x_train[y_train == 0]
# n_x_train = n_x_train[:1000]
# 在 Silhouette Score 中新增包含有詐騙的案例
normal = x_train[y_train == 0][:800]
fraud  = x_train[y_train == 1][:200]
n_x_train = np.vstack([normal, fraud])

scores = []
for k in range(2, 5):
   kmeans = KMeans(n_clusters=k, init='k-means++', random_state=RANDOM_SEED)
   kmeans.fit(n_x_train)
   score = silhouette_score(n_x_train, kmeans.labels_)
   scores.append(score)

optimal_k = np.argmax(scores) + 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=RANDOM_SEED)
kmeans.fit(n_x_train)
y_pred_test = kmeans.predict(x_test)
def align_labels(y_true, y_pred, n_clusters):
   labels = np.zeros_like(y_pred)
   for i in range(n_clusters):
       mask = (y_pred == i)
       if np.sum(mask) > 0:
           labels[mask] = np.bincount(y_true[mask]).argmax()
       else:
           labels[mask] = 0  # Default to normal class
   return labels

y_pred_aligned = align_labels(y_test, y_pred_test, optimal_k)

def evaluation(y_true, y_pred, model_name="Model"):
   accuracy = accuracy_score(y_true, y_pred)
   precision = precision_score(y_true, y_pred, zero_division=0)
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

evaluation(y_test, y_pred_aligned, model_name="KMeans (Unsupervised)")


# 儲存模型到檔案
joblib.dump(kmeans, f"kmeans-{RANDOM_SEED}.pkl")