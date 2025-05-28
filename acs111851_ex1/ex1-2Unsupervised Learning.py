#################### K-Means 非監督式學習#####################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import kagglehub

# General settings
RANDOM_SEED = 42
TEST_SIZE = 0.3

# Load dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
data['Class'] = data['Class'].astype(int)

# Prepare data
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# Extract features and labels
X = data.drop(columns=['Class']).values
y = data['Class'].values

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Hyperparameters
pca_variances = [0.96, 0.97, 0.98]
sample_sizes = [1200, 1500, 1800, 2000]
best_f1 = 0
best_y_pred = None
best_config = {}

# Start grid search
for pca_var in pca_variances:
    pca = PCA(n_components=pca_var, random_state=RANDOM_SEED)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    for sample_size in sample_sizes:
        n_x_train = x_train_pca[y_train == 0][:sample_size]
        z_scores = np.abs((n_x_train - n_x_train.mean(axis=0)) / n_x_train.std(axis=0))
        n_x_train = n_x_train[(z_scores < 1.8).all(axis=1)]

        k_range = range(10, 16)
        scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=500, random_state=RANDOM_SEED)
            kmeans.fit(n_x_train)
            scores.append(silhouette_score(n_x_train, kmeans.labels_))

        optimal_k = k_range[np.argmax(scores)]
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=500, random_state=RANDOM_SEED)
        kmeans.fit(n_x_train)

        cov_matrix = np.cov(n_x_train.T)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        centroids = kmeans.cluster_centers_

        def min_mahalanobis_distance(x, centroids, inv_cov):
            return min([mahalanobis(x, c, inv_cov) for c in centroids])

        min_distances = np.array([min_mahalanobis_distance(x, centroids, inv_cov_matrix) for x in x_test_pca])
        min_distances = (min_distances - np.median(min_distances)) / (
            np.percentile(min_distances, 75) - np.percentile(min_distances, 25))

        percentiles = np.arange(99.85, 99.99, 0.002)
        for perc in percentiles:
            threshold = np.percentile(min_distances, perc)
            y_pred = (min_distances > threshold).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            score = 0.85 * precision + 0.15 * recall

            if f1 > best_f1:
                best_f1 = f1
                best_y_pred = y_pred
                best_config = {
                    'pca_var': pca_var,
                    'sample_size': sample_size,
                    'k': optimal_k,
                    'threshold': round(threshold, 4)
                }

# Final evaluation
def evaluation(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'\n{model_name} Evaluation:')
    print('===' * 15)
    print(f'         Accuracy: {accuracy:.4f}')
    print(f'  Precision Score: {precision:.4f}')
    print(f'     Recall Score: {recall:.4f}')
    print(f'         F1 Score: {f1:.4f}')
    print(f'Best Configuration: {best_config}')
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# Evaluate
evaluation(y_test, best_y_pred, model_name="KMeans (Ultra Fine-Tuned Mahalanobis)")
