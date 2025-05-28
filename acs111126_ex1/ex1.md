# EX1:  Random Forest(Supervised Learning) and KMeans (Unsupervised Learning)

## Dataset

[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.
The dataset is highly imbalanced, with only 0.173% of transactions being fraudulent (492 fraudulent vs. 284,315 non-fraudulent).
After analysing the data set, the false positive and false negative rate is also relatively low.

## Random Forest Model (Supervised Learning)

- **Parameters**:
  - `max_depth=16` controls tree complexity(depth). added max_depth to 16.
  - `n_estimators=120` increases the number of trees. changed n_estimators from 100 to 120.
  - `max_features=10` limits the number of features considered at each split. added max_features to 10.

I have tried many different parameters but ended up simply using max_depth and max_features for most simple impelementation.

### Output

Random Forest (Supervised) Evaluation:
=============================================
         Accuracy: 0.9996605924417448
  Precision Score: 0.9495798319327731
     Recall Score: 0.8308823529411765
         F1 Score: 0.8862745098039215

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.95      0.83      0.89       136

    accuracy                           1.00     85443
   macro avg       0.97      0.92      0.94     85443
weighted avg       1.00      1.00      1.00     85443

Overview: Sightly better F1 Score, fine tuning parameter is pain considering the time it takes to run. (overall 6 hrs total work time)

## KMeans (Unsupervised Learning)

- **PCA**:
  - `n_components=17` reduce dimensionality. fine tuned to 17.
It is said that one part of the data has not been through pca before.

- **Parameters**:
  - `n_estimators=k` taken sample size.
  - `init='k-means++'` greedy k-means++ algorithm.
  - `n_init=10` cluster center update rate. added n_init to 10.


### Output

KMeans (Unsupervised) Evaluation:
=============================================
         Accuracy: 0.9987477031471274
  Precision Score: 0.7887323943661971
     Recall Score: 0.3783783783783784
         F1 Score: 0.5114155251141552

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.79      0.38      0.51       148

    accuracy                           1.00     85443
   macro avg       0.89      0.69      0.76     85443
weighted avg       1.00      1.00      1.00     85443

Overview: Sightly better F1 Score, KMean in this dataset does not perform too well. still took a lot of work fine tuning considering changing certain parameters or training data broke the results compeletly. (overall 4 hrs total work time)


## References
Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html (not the only source but the only one ended up in the final product)

KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html (not the only source but the only one ended up in the final product)

additional references: use of llms and search engines to check for grammar and recommmend parameters to try. did try using the code provided but llms are way too stupid without proper prompt engineering and direction provided. (I installed kagglehub on my machine 4 times ICANT)