# Member 6 – Machine Learning Model Development and Evaluation

## 1. Overview

This folder contains the individual work completed by **Member 6** for the IT2011 – Artificial Intelligence and Machine Learning group assignment.

The main responsibility of Member 6 is **Machine Learning Model Development and Evaluation**.

The work focuses on:

* Logistic Regression
* Support Vector Machine (SVM)
* PCA + Logistic Regression
* PCA + SVM
* Model evaluation
* 5-fold stratified cross-validation
* Hyperparameter tuning using GridSearchCV
* Model comparison
* Confusion matrix analysis
* ROC-AUC evaluation
* Selection of the final model

The dataset used is the **Breast Cancer Wisconsin (Original)** dataset.

---

## 2. Dataset

The Breast Cancer Wisconsin (Original) dataset contains measurements describing characteristics of breast cell nuclei.

### Dataset Information

* Original observations: **699**
* Predictive features: **9**
* Target variable: **Class**
* Original class labels:

  * `2` = Benign
  * `4` = Malignant
* Missing values: **16**, all in `Bare_Nuclei`
* Exact duplicate observations removed: **8**
* Final modelling observations: **691**

The target variable was encoded as:

```text
0 = Benign
1 = Malignant
```

The dataset was divided using a stratified train-test split:

```text
Training set: 552 observations
Test set: 139 observations
```

A `random_state` of 42 was used to make the split reproducible.

---

## 3. Preprocessing

The modelling notebook uses a scikit-learn `Pipeline` to keep preprocessing and model training together.

The main preprocessing steps are:

### Missing Value Handling

Missing values represented by `?` were converted to `NaN`.

The missing `Bare_Nuclei` values are handled using:

```python
SimpleImputer(strategy="median")
```

The imputer is included inside the modelling pipeline to prevent data leakage during cross-validation.

### Feature Scaling

`StandardScaler()` is used before Logistic Regression, SVM, and PCA.

Scaling is useful because:

* Logistic Regression can benefit from standardized features.
* SVM is sensitive to feature scale.
* PCA is affected by feature scale.

### PCA

PCA is applied in the PCA-based models.

Four principal components are used to reduce the original 9 features while retaining approximately **85.35% of the variance** based on the PCA analysis.

---

## 4. Machine Learning Models

Four main modelling approaches were evaluated.

### 4.1 Logistic Regression

A Logistic Regression pipeline was created using:

```text
Median Imputation
        ↓
Standard Scaling
        ↓
Logistic Regression
```

Logistic Regression was evaluated as a baseline model and also tuned using `GridSearchCV`.

---

### 4.2 Support Vector Machine

An SVM pipeline was created using an RBF kernel for the baseline model:

```text
Median Imputation
        ↓
Standard Scaling
        ↓
SVM
```

The SVM model was also tuned using `GridSearchCV`.

---

### 4.3 PCA + Logistic Regression

PCA was added before Logistic Regression:

```text
Median Imputation
        ↓
Standard Scaling
        ↓
PCA
        ↓
Logistic Regression
```

Four principal components were used.

---

### 4.4 PCA + SVM

PCA was combined with SVM:

```text
Median Imputation
        ↓
Standard Scaling
        ↓
PCA
        ↓
SVM
```

This model was also evaluated using cross-validation and hyperparameter tuning.

---

## 5. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* 5-fold Stratified Cross-Validation

### Why These Metrics Were Used

**Accuracy** measures the overall proportion of correct predictions.

**Precision** measures how many observations predicted as malignant were actually malignant.

**Recall** measures how many actual malignant cases were correctly identified.

Recall is particularly important in this problem because a false negative means a malignant case was predicted as benign.

**F1-score** provides a balance between precision and recall.

**ROC-AUC** measures how well the model separates the two classes across different classification thresholds.

---

## 6. Cross-Validation

A **5-fold Stratified Cross-Validation** approach was used.

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Stratification helps maintain a similar benign/malignant class distribution in each fold.

The cross-validation results for the PCA + SVM model were:

| Metric    | Mean ± Standard Deviation |
| --------- | ------------------------: |
| Accuracy  |           0.9714 ± 0.0104 |
| Precision |           0.9453 ± 0.0225 |
| Recall    |           0.9742 ± 0.0162 |
| F1-score  |           0.9593 ± 0.0142 |
| ROC-AUC   |           0.9889 ± 0.0086 |

The relatively high recall indicates that the PCA + SVM model identified most malignant cases across the validation folds.

---

## 7. Hyperparameter Tuning

`GridSearchCV` was used to search for better model configurations.

### Logistic Regression

The following parameters were tested:

```python
C = [0.01, 0.1, 1, 10, 100]
solver = ["liblinear", "lbfgs"]
class_weight = [None, "balanced"]
```

### SVM

The following parameters were tested:

```python
C = [0.1, 1, 10, 100]
kernel = ["linear", "rbf"]
gamma = ["scale", "auto"]
class_weight = [None, "balanced"]
```

The tuning process used **5-fold cross-validation** and F1-score as the main selection metric.

---

## 8. Final Model Comparison

The models were compared using the test-set performance.

| Model                                |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ------------------------------------ | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression – Baseline       |     0.9640 |     0.9778 |     0.9167 |     0.9462 |     0.9970 |
| Logistic Regression – Tuned          |     0.9712 |     0.9583 |     0.9583 |     0.9583 |     0.9961 |
| SVM – Baseline                       |     0.9712 |     0.9783 |     0.9375 |     0.9574 |     0.9922 |
| SVM – Tuned                          |     0.9712 |     0.9583 |     0.9583 |     0.9583 |     0.9970 |
| PCA + Logistic Regression – Baseline | **0.9784** | **0.9787** | **0.9583** | **0.9684** | **0.9966** |
| PCA + Logistic Regression – Tuned    |     0.9712 |     0.9583 |     0.9583 |     0.9583 |     0.9970 |
| PCA + SVM – Baseline                 |     0.9712 |     0.9583 |     0.9583 |     0.9583 |     0.9881 |
| PCA + SVM – Tuned                    |     0.9712 |     0.9583 |     0.9583 |     0.9583 |     0.9968 |

---

## 9. Selected Model

The **PCA + Logistic Regression – Baseline** model was selected as the final model.

It achieved:

* Accuracy: **97.84%**
* Precision: **97.87%**
* Recall: **95.83%**
* F1-score: **96.84%**
* ROC-AUC: **99.66%**

Its test confusion matrix was:

```text
[[90, 1],
 [ 2, 46]]
```

This means:

* True Negatives = 90
* False Positives = 1
* False Negatives = 2
* True Positives = 46

The model was selected because it achieved the highest test accuracy, precision, and F1-score among the evaluated approaches while maintaining strong recall and ROC-AUC.

The model should **not** be considered clinically ready because this project uses a historical dataset and is an academic machine learning study.

---

## 10. Data Leakage Prevention

To reduce the risk of data leakage, preprocessing operations such as:

* Missing-value imputation
* Feature scaling
* PCA

were included inside scikit-learn pipelines.

This ensures that preprocessing parameters are learned from the appropriate training data during model evaluation and cross-validation.

---

## 11. Results

The notebook generates outputs related to:

* Model performance comparison
* Cross-validation results
* Confusion matrices
* ROC curves
* Model evaluation metrics

These outputs support the comparison and selection of the final model.

---

## 12. Limitations

The project has several limitations:

* The dataset is historical.
* The dataset contains only 699 original observations.
* The dataset is relatively small compared with modern medical datasets.
* The data contains missing values.
* The dataset has some class imbalance.
* The models are evaluated on a single train-test split in addition to cross-validation.
* High performance on this dataset does not guarantee performance on new hospitals, populations, or modern clinical data.
* The model should not be used as a standalone medical diagnosis system.

---

## 13. Technologies and Libraries

The following Python libraries were used:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / VS Code

Important scikit-learn components include:

```text
Pipeline
SimpleImputer
StandardScaler
LogisticRegression
SVC
PCA
train_test_split
StratifiedKFold
cross_validate
GridSearchCV
```

---

## 14. Folder Structure

```text
MEMBER_06/
├── README.md
├── data/
│   ├── raw/
│   │   └── breast-cancer-wisconsin.data
│   └── external/
├── notebook.ipynb
└── results/
    ├── eda_visualizations/
    ├── logs/
    └── outputs/
```

---

## 15. Summary

Member 6 focused on the **Machine Learning Model Development and Evaluation** phase.

Multiple machine learning approaches were implemented and compared, including Logistic Regression, SVM, PCA + Logistic Regression, and PCA + SVM.

The models were evaluated using test-set metrics, confusion matrices, ROC-AUC, and 5-fold stratified cross-validation. Hyperparameter tuning was also performed using `GridSearchCV`.

Based on the final test-set comparison, **PCA + Logistic Regression – Baseline** achieved the strongest overall performance and was selected as the final model.

The results demonstrate that the evaluated machine learning approaches can classify the two classes in this dataset effectively, while the limitations of the historical dataset mean that the results should not be interpreted as evidence of clinical readiness.
