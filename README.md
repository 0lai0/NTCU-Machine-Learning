# NTCU Machine Learning Assignment Repository
**NTCU-Machine-Learning** repository for the Machine Learning course at NTCU. 
This repository is used for submitting assignments related to machine learning projects, with a focus on **Credit Card Fraud Detection**.

## Project Overview
This assignment focuses on building a machine learning model for **credit card fraud detection**. 
You will use a dataset to train and evaluate models, applying techniques such as data preprocessing, feature scaling, and classification algorithms (e.g., Random Forest) or clustering (e.g., KMeans).

**Objectives:**
- Load and preprocess the dataset.
- Train a machine learning model to detect fraudulent transactions.
- Evaluate the model using metrics like accuracy, precision, recall, F1-score, ROC AUC, and confusion matrix.

## Setup Instructions
To set up your environment and work on the assignment, follow these steps:

### 1. Fork the Repository
- Fork the `NTCU-Machine-Learning` repository to your GitHub account.
- Clone your forked repository to your local machine:
  ```bash
  git clone <your-forked-repo-url>
  ```

### 2. Install Git
Ensure Git is installed on your system:
- **Windows/Mac**: Download and install Git from [git-scm.com](https://git-scm.com).
- **Ubuntu/Linux**:
  ```bash
  sudo apt update
  sudo apt install git
  ```

## Submission Guidelines
1. **Fork and Clone**: Fork this repository and clone it to your local machine.
2. **Create a Branch**: Create a branch for your assignment (e.g., `assignment-<your-student-id>`).
3. **Implement Your Code**: Modify the provided code template (see [Code Structure](#code-structure)) to complete the assignment.
4. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Submit assignment for <your-student-id>"
   git push origin <your-branch-name>
   ```
5. **Create a Pull Request**: Submit a pull request from your forked repository to the main repository for review.
6. **File Naming**: Name your main script as `<ex_number>.py`.
7. **Create a New Folder**: Remeber put your each file into `<your-student-id>_<ex_number>`.

**Important**:
- Do not modify the `TEST_SIZE` (set to `0.3`) or `RANDOM_SEED` (set to `42`) in the code.
- Ensure your code is well-documented with comments explaining your approach.
- Submit your pull request before the deadline.

## Dataset
The dataset for this assignment is available via **KaggleHub**. Use the following code to load it:
```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
data = pd.read_csv(f"{path}/creditcard.csv")
```

- The dataset contains credit card transaction data with features like transaction amount, time, and anonymized features (`V1` to `V28`).
- The target variable is `Class` (0 for non-fraudulent, 1 for fraudulent).

### Tasks
1. Preprocess the data (e.g., handle missing values, scale features using `StandardScaler`).
2. Split the dataset into training (70%) and testing (30%) sets.
3. Train a classification model (e.g., `RandomForestClassifier`) or a clustering model (e.g., `KMeans`).
4. Evaluate the model using the following metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC AUC score
   - Confusion Matrix
   - (For clustering) Silhouette Score

## Evaluation Metrics
Your model will be evaluated based on:
- **Correctness**: Does the code run without errors and produce the expected outputs?
- **Performance**: How well does your model perform on the test set (based on the metrics above)?
- **Code Quality**: Is the code well-organized, commented, and easy to understand?
- **Documentation**: Include a brief explanation of your approach in the pull request description.

## Contact
For questions or issues, contact the teaching assistant via:
- Email: [bcs113116@gm.ntcu.edu.tw]
- Email: [bcs113115@gm.ntcu.edu.tw]
---
