import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, average_precision_score
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

# 載入資料
data = pd.read_csv("creditcard.csv")
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time'], axis=1)

# 建立 IsolationForest 模型（無監督）
# iso_forest = IsolationForest(
#     n_estimators = 600,
#     contamination = 0.01,
#     random_state = 42
# )
# iso_scores = iso_forest.fit_predict(data.drop(columns=['Class']))  # -1 代表異常，1 代表正常
# anomaly_score = iso_forest.decision_function(data.drop(columns=['Class']))  # 分數越小越可能異常

# 加入為新特徵
# data['isolation_label'] = (iso_scores == -1).astype(int)
# data['anomaly_score'] = anomaly_score

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001)
lof_labels = lof.fit_predict(data.drop(columns=['Class']))
lof_scores = -lof.negative_outlier_factor_  # 越大越異常

data['lof_label'] = (lof_labels == -1).astype(int)
data['lof_score'] = lof_scores


# sns.histplot(data, x='lof_score', hue='Class', bins=100, kde=True, stat='density')
# plt.axvline(0, color='red', linestyle='--', label='Score = 0')
# plt.title('lof_score Distribution by Class')
# plt.xlabel('lof_score')
# plt.ylabel('Density')
# plt.legend()
# plt.show()

# 分離特徵與標籤
X = data.drop(columns=['Class'])
y = data['Class']

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Undersampling
# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

#資料resample後的視覺化比對
# labels = ['Negative', 'Positive']
# original_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
# resampled_counts = [np.sum(y_resampled == 0), np.sum(y_resampled == 1)]

# x = np.arange(len(labels))  # x 軸位置
# width = 0.35

# fig, ax = plt.subplots()
# bar1 = ax.bar(x - width/2, original_counts, width, label='Original')
# bar2 = ax.bar(x + width/2, resampled_counts, width, label='Resampled')

# ax.set_ylabel('Samples')
# ax.set_title('Class Distribution: Original vs Resampled')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
# plt.grid(True, axis='y', linestyle='--', alpha = 0.5)
# plt.show()

# 3. 建立模型 + 記錄 logloss
evals_result = {}  # 用來儲存 logloss 結果

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state = 42,
    scale_pos_weight = (len(y_train[y_train==0]) / len(y_train[y_train==1])),  # class imbalance
    max_depth = 7,
    early_stopping_rounds = 10,
)

#scale_pos_weight=len(y_resampled[y_resampled==0]) / len(y_resampled[y_resampled==1]),  # class imbalance

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
evals_result = model.evals_result()

#early_stopping_rounds=10,

# 4. 繪製學習曲線
epochs = len(evals_result['validation_0']['logloss'])
x_axis = range(epochs)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, evals_result['validation_0']['logloss'], label='Train')
plt.plot(x_axis, evals_result['validation_1']['logloss'], label='Test')
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("XGBoost Learning Curve (Log Loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#比較traning error、testing error
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_error = 1 - recall_score(y_train, y_train_pred)
test_error = 1 - recall_score(y_test, y_test_pred)

print(f'Training error and Testing error:')
print('===' * 15)
print(f"Training Error (recall): {train_error}")
print(f"Testing Error (recall): {test_error}")

# 預測
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

def evaluation(y_true, y_pred, y_prob, model_name="Model"):
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
   print(f"           AUPRC: {average_precision_score(y_test, y_prob)}")
   print("\nClassification Report:")
   print(classification_report(y_true, y_pred))

evaluation(y_test, y_pred, y_prob, model_name="Hybird")

# 評估
#print("Classification Report:\n", classification_report(y_test, y_pred))
#print(f"AUPRC (Average Precision): {average_precision_score(y_test, y_prob):.4f}")
