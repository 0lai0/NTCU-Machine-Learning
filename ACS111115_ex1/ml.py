import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import kagglehub


# general setting. do not change TEST_SIZE
RANDOM_SEED = 42
TEST_SIZE = 0.3

# load dataset（from kagglehub）
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)#0 for nonfraud 1 for fraud

# prepare data
data = data.drop(['Time'], axis=1)#去除time去除time
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))#標準化

fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]
print(f'Fraudulent:{len(fraud)}, non-fraudulent:{len(nonfraud)}')
print(f'the positive class (frauds) percentage: {len(fraud)}/{len(fraud) + len(nonfraud)} ({len(fraud)/(len(fraud) + len(nonfraud))*100:.3f}%)')    
 
X = np.asarray(data.iloc[:, ~data.columns.isin(['Class'])])#選擇非class的值轉成numpy array 且如果原本就是np array時不複製
Y = data['Class'].to_numpy()#最佳化為 Pandas → NumPy 的不複製轉換

# split training set and data set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# build Random Forest model
rf_model = RandomForestClassifier(n_estimators=85
                                  , class_weight='balanced_subsample'
                                  , min_samples_leaf= 2
                                  , oob_score= True
                                  , max_features='sqrt'
                                  , max_depth=25
                                  , min_samples_split= 3
                                  , random_state=RANDOM_SEED)
rf_model.fit(X_train, y_train)

# define evaluation function
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

# predict and print result
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
