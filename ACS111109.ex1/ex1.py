import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import kagglehub

RANDOM_SEED = 42
TEST_SIZE = 0.3

# Load dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)

# Preprocessing
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop(columns=['Class']).to_numpy()
Y = data['Class'].to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Build model
rf_model = RandomForestClassifier(
    n_estimators=85,
    max_depth=25,
    class_weight='balanced_subsample',
    min_samples_split=2,
    max_features='sqrt',
    oob_score=True,
    random_state=RANDOM_SEED
)
rf_model.fit(X_train, y_train)

# Evaluation
def evaluation(y_true, y_pred, model_name="Model"):
    print(f'\n{model_name} Evaluation:')
    print('===' * 15)
    print('Accuracy       :', accuracy_score(y_true, y_pred))
    print('Precision Score:', precision_score(y_true, y_pred))
    print('Recall Score   :', recall_score(y_true, y_pred))
    print('F1 Score       :', f1_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# Predict
y_pred = rf_model.predict(X_test)
evaluation(y_test, y_pred, model_name="Random Forest")
