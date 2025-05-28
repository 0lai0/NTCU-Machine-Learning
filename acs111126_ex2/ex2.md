# EX1:  isolation+XGBoost

## Dataset

[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.
The dataset is highly imbalanced, with only 0.173% of transactions being fraudulent (492 fraudulent vs. 284,315 non-fraudulent).
After analysing the data set, the false positive and false negative rate is also relatively low.

## isolation+XGBoost

### isolation

- **Parameters**:
  - `contamination=0.016` how isolated the data is. added contamination to 0.016 after fine tuning.

### XGBoost

- **Parameters**:
  - `max_depth=6` controls tree complexity(depth). added max_depth to 6.
  - `n_estimators=120` increases the number of trees. changed n_estimators from 100 to 120 taken form ex1, and still works.
  - `learning_rate=0.1` reduce step size to not over fitt. added learning_rate to 0.1 after fine tuning, adding this helped, i tried changing this for a long time.
  - `subsample=0.8`randomly sample data. added subsample to 0.8 after fine tuning, adding this helped.
  - `scale_pos_weight=2.5` adds more weight to help with the imbalanced dataset. added scale_pos_weigh to 2.5 after fine tuning.

After trying for a long time, adding and adjusting parameters, this is the best i've got, some parameters like scale_pos_weight is very helpful and is taken in advice form other students who told me.

### Output
isolation+XGBoost Evaluation:
=============================================
         Accuracy: 0.9997074072773662
  Precision Score: 0.9512195121951219
     Recall Score: 0.8602941176470589
         F1 Score: 0.9034749034749034

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.95      0.86      0.90       136

    accuracy                           1.00     85443
   macro avg       0.98      0.93      0.95     85443
weighted avg       1.00      1.00      1.00     85443

Overview: Sightly better F1 Score, would have taken more time if not for other people who helped, also as I said the false positive and false negative rate is also relatively low so isolation forest did not work well for me, I don't kmow how others approched this problem but I mainly focused on improving xgboost. (overall 6 hrs total work time)



## References
Isoation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html (not the only source but the only one ended up in the final product)

XGBoost: https://xgboost.readthedocs.io/en/latest/python/python_api.html (parameters)

additional references: use of llms and search engines to check for grammar and recommmend parameters to try. did try using the code provided but llms are way too stupid without proper prompt engineering and direction provided. (the documentation is way more helpful than any llm, also other people who helped with discussing which parameters worked for them <3) 