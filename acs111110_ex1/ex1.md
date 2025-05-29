監督是學習(random forest)

參數調整
rf_model = RandomForestClassifier(
    n_estimators=130
    max_depth=20
    min_samples_leaf=2
    class_weight='balanced'
    random_state=RANDOM_SEED
    )


output

Model Evaluation:
=============================================
         Accuracy: 0.9996488887328394
  Precision Score: 0.9416666666666667
     Recall Score: 0.8308823529411765
         F1 Score: 0.8828125

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.94      0.83      0.88       136

    accuracy                           1.00     85443
   macro avg       0.97      0.92      0.94     85443
weighted avg       1.00      1.00      1.00     85443


--------------------------------------------------------------------------
非監督學習(Kmeans)

參數調整

調整range大小
k_range = range(2, 15)
調整kmeans參數
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1000, max_iter=500, random_state=RANDOM_SEED)
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=1000, max_iter=500, random_state=RANDOM_SEED)

新增PCA (Principal Component Analysis)(降維技術)
pca = PCA(n_components=26,random_state=RANDOM_SEED)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


output

KMeans (Unsupervised) Evaluation:
=============================================
         Accuracy: 0.9987594068560327
  Precision Score: 0.7916666666666666
     Recall Score: 0.38513513513513514
         F1 Score: 0.5181818181818182

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.79      0.39      0.52       148

    accuracy                           1.00     85443
   macro avg       0.90      0.69      0.76     85443
weighted avg       1.00      1.00      1.00     85443