# IT2011 Artificial Intelligence and Machine Learning
## Breast Cancer Classification Using Machine Learning

### Group Project – Year 2 Semester 1 (2026)

This repository contains the implementation of our IT2011 Artificial Intelligence and Machine Learning group project.

The project uses the **Breast Cancer Wisconsin (Original)** dataset to develop and compare multiple machine learning models for binary breast cancer classification.

The main objective is to classify a breast cancer sample as:

- **Benign**
- **Malignant**

---

## 1. Dataset

**Dataset:** Breast Cancer Wisconsin (Original)

**Source:** UCI Machine Learning Repository

The original dataset contains:

- 699 observations
- 9 predictive features
- 1 sample ID column
- 1 class/target column
- 16 missing values in the `Bare_Nuclei` feature
- Missing values represented by `?`

Original target classes:

- `2` = Benign
- `4` = Malignant

For machine learning, the target is mapped as:

- `0` = Benign
- `1` = Malignant

During data quality investigation, 8 exact duplicate observations were identified and removed, resulting in 691 observations for modelling.

The original dataset is stored in:

`data/raw/breast-cancer-wisconsin.data`

---

## 2. Group Members and Models

| IT Number | Member | Machine Learning Model |
|---|---|---|
| IT25102909 | Charuka M.G.D.D.V | Decision Tree |
| IT25101082 | Perera G.N.N | K-Nearest Neighbors (KNN) |
| IT25104028 | Niluminda B.G.E.S | Random Forest |
| IT25103989 | Ahameth M.U.Y | Logistic Regression |
| IT25101976 | Undugodage J.C.A | PCA + Logistic Regression |
| IT25101990 | Amasha S.M.N | Support Vector Machine (SVM) |

Each member implemented and evaluated an individual machine learning model. Baseline and tuned versions were evaluated using appropriate classification metrics and cross-validation.

---

## 3. Individual Notebooks

The six individual implementation notebooks are stored in the `notebooks/` directory:

- `IT25102909_Decision_Tree.ipynb`
- `IT25101082_KNN.ipynb`
- `IT25104028_Random_Forest.ipynb`
- `IT25103989_Logistic_Regression.ipynb`
- `IT25101976_PCA_Logistic_Regression.ipynb`
- `IT25101990_SVM.ipynb`

Each notebook contains the code, explanations, model implementation, tuning, evaluation, and generated outputs for the corresponding member.

---

## 4. Group Pipeline

The integrated group-level analysis is available in:

`group_pipeline.ipynb`

This notebook brings together the final results of all six machine learning approaches and compares their performance.

The comparison considers metrics including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- False negatives
- Cross-validation performance

---

## 5. Final Model Comparison

The six selected tuned models were compared using the common held-out test set.

| Model | Accuracy | Recall | F1-score | ROC-AUC | False Negatives |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.9281 | 0.8333 | 0.8889 | 0.9666 | 8 |
| KNN | 0.9640 | 0.9167 | 0.9462 | 0.9823 | 4 |
| Random Forest | 0.9784 | 0.9792 | 0.9691 | 0.9934 | 1 |
| Logistic Regression | 0.9712 | 0.9583 | 0.9583 | 0.9961 | 2 |
| PCA + Logistic Regression | 0.9712 | 0.9583 | 0.9583 | 0.9970 | 2 |
| SVM | 0.9712 | 0.9583 | 0.9583 | 0.9970 | 2 |

In this experiment, **Random Forest showed the strongest overall held-out classification performance**, with an accuracy of 0.9784, recall of 0.9792, F1-score of 0.9691, and only one false negative.

These results are experimental and should not be interpreted as evidence of clinical readiness. External validation would be required before considering real-world clinical use.

---

## 6. Repository Structure

```text
Group_ID/
|
|-- README.md
|-- group_pipeline.ipynb
|
|-- data/
|   |-- raw/
|   |   `-- breast-cancer-wisconsin.data
|   |
|   `-- external/
|
|-- notebooks/
|   |-- IT25102909_Decision_Tree.ipynb
|   |-- IT25101082_KNN.ipynb
|   |-- IT25104028_Random_Forest.ipynb
|   |-- IT25103989_Logistic_Regression.ipynb
|   |-- IT25101976_PCA_Logistic_Regression.ipynb
|   `-- IT25101990_SVM.ipynb
|
`-- results/
    |-- eda_visualizations/
    |-- logs/
    `-- outputs/

## 7. EDA Visualizations

The main exploratory data analysis plots are stored in:

`results/eda_visualizations/`

These include:

- Class distribution
- Feature histograms
- Feature boxplots
- Class-wise feature boxplots
- Correlation matrix

---

## 8. Final Outputs

Final group comparison results are stored in:

`results/outputs/`

These include:

- Model comparison tables
- F1-score ranking
- Recall ranking
- ROC-AUC ranking
- False-negative comparison
- Final selected model information
- Group comparison visualizations

---

## 9. How to Run the Project

1. Clone or download the repository.
2. Create and activate a Python virtual environment.
3. Install the required Python libraries.
4. Confirm that the dataset exists at: `data/raw/breast-cancer-wisconsin.data`
5. Open Jupyter Notebook or JupyterLab.
6. Run the required individual notebook from the `notebooks/` directory.
7. Run `group_pipeline.ipynb` to view the final six-model comparison.

### Main Python Libraries

The main Python libraries used in this project include:

- pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter

---

## 10. Reproducibility and Data Leakage Prevention

For the final machine learning evaluation, preprocessing operations such as missing-value imputation and feature scaling are incorporated into scikit-learn pipelines where applicable.

These preprocessing steps are fitted using the appropriate training data or training folds rather than using information from the held-out test set. This helps reduce data leakage during model evaluation.

Stratified train/test splitting and stratified cross-validation are used to maintain class proportions during evaluation.

---

## 11. Ethical Considerations

Breast cancer classification is a high-impact healthcare application. Particular attention should therefore be given to **false-negative predictions**, where a malignant case is incorrectly classified as benign.

The dataset is relatively small and historical, which limits the generalizability of the results. Model performance on this dataset does not guarantee equivalent performance on modern or diverse clinical populations.

The models developed in this project are intended for **academic experimentation and evaluation only** and are **not clinical diagnostic systems**.

---

## 12. Streamlit Prediction Application

A Streamlit-based user interface was developed to demonstrate how the final machine learning solution can be used in an interactive application.

The application uses the **tuned Random Forest pipeline**, which achieved the strongest overall held-out performance in the final group comparison.

### Final Random Forest Performance

| Metric | Result |
|---|---:|
| Accuracy | 97.84% |
| Precision | 95.92% |
| Recall | 97.92% |
| F1-Score | 96.91% |
| ROC-AUC | 99.34% |
| False Negatives | 1 |

### Application Features

- Accepts all 9 predictive features from the Breast Cancer Wisconsin (Original) dataset.
- Uses feature values on the original 1–10 scale.
- Loads the saved tuned Random Forest pipeline.
- Applies the same preprocessing and model used during final evaluation.
- Predicts **Benign** or **Malignant**.
- Displays Random Forest class-probability outputs.
- Displays the entered sample for transparency.
- Shows final model performance information.
- Includes a clear educational and non-clinical-use disclaimer.

### Application Structure

```text
app/
└── app.py

models/
└── random_forest_pipeline.pkl

requirements.txt
```

### Install Application Dependencies

From the project root:

```bash
python -m pip install -r requirements.txt
```

### Run the Application

```bash
python -m streamlit run app/app.py
```

Streamlit will start the application locally, normally at:

```text
http://localhost:8501
```

### Important Disclaimer

This application is an **academic AI/ML demonstration only**. The model was trained using the historical Breast Cancer Wisconsin (Original) dataset and has not been clinically validated. The predictions and class-probability outputs must not be interpreted as medical diagnoses, clinical risk estimates, or treatment recommendations.