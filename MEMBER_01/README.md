# IT2011 – Breast Cancer Wisconsin (Original)

## Progress Review I – Member 1

### 1. Overview

This folder contains the work completed by Member 1 for the IT2011 Artificial Intelligence and Machine Learning project.

The project uses the **Breast Cancer Wisconsin (Original)** dataset for a binary classification problem.

Member 1 is mainly responsible for **data loading, data quality investigation, missing value handling, and duplicate investigation/removal**.

---

### 2. Dataset

**Dataset:** Breast Cancer Wisconsin (Original)

The original dataset contains:

* 699 observations
* 9 predictive features
* 1 sample ID column
* 1 class/target column
* Missing values represented by `?`

The dataset contains two classes:

* `2` – Benign
* `4` – Malignant

The raw dataset is stored in:

```text
data/raw/breast-cancer-wisconsin.data
```

---

### 3. Member 1 Responsibilities

The following data preprocessing and data quality tasks were performed:

1. Loaded the original dataset.
2. Assigned meaningful column names.
3. Inspected the dataset shape and data types.
4. Investigated missing values.
5. Identified `?` values in the `Bare_Nuclei` feature.
6. Converted `?` values to `NaN`.
7. Converted `Bare_Nuclei` to a numeric data type.
8. Investigated the distribution of missing values.
9. Investigated exact duplicate observations.
10. Removed exact duplicate observations.
11. Applied median imputation to the remaining missing values.
12. Performed a final data quality check.

---

### 4. Main Findings

The initial dataset contained:

```text
699 rows
11 columns
```

A total of **16 missing values** were identified.

All missing values were found in:

```text
Bare_Nuclei
```

The missing values were originally represented using:

```text
?
```

After converting them to `NaN`, the missing values were handled using **median imputation**.

The median was selected because `Bare_Nuclei` has a skewed distribution, and the median is less affected by extreme values than the mean.

During duplicate investigation, **8 exact duplicate observations** were identified and removed.

Therefore:

```text
699 original observations
        ↓
8 exact duplicate observations removed
        ↓
691 observations
```

The final cleaned dataset contains:

```text
691 rows
11 columns
0 missing values
0 exact duplicate rows
```

---

### 5. Important Note About Repeated Sample IDs

Repeated `Sample_ID` values were not automatically treated as duplicates.

A row was removed only when the **entire row contained the same information** as another row.

If the same `Sample_ID` appeared with different feature values, the observations were retained.

This avoids incorrectly removing potentially valid observations.

---

### 6. Notebook

The complete implementation can be found in:

```text
notebook.ipynb
```

The notebook contains the code and outputs for the data quality investigation and preprocessing steps described above.

---

### 7. Results

Generated outputs are stored in:

```text
results/outputs/
```

The results folder contains the outputs produced during the analysis.

---

### 8. Folder Structure

```text
MEMBER_01/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   └── breast-cancer-wisconsin.data
│   │
│   └── external/
│
├── notebook.ipynb
│
└── results/
    ├── eda_visualizations/
    ├── logs/
    └── outputs/
```

---

### 9. Reproducibility

To reproduce the analysis:

1. Place the original dataset in `data/raw/`.
2. Open `notebook.ipynb`.
3. Run the notebook cells in order.
4. The generated outputs will be saved in the `results/` directory.

---

### 10. Data Leakage Note

The direct median imputation demonstrated during the data-cleaning stage is used for the preprocessing investigation.

For the final machine learning models, preprocessing is performed using a **scikit-learn Pipeline**, with imputation fitted only on the appropriate training data/folds. This helps prevent data leakage during model evaluation.
