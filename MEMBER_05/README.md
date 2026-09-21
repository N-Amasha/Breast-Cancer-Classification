# IT2011 – AI and Machine Learning

## Member 5 – PCA and Dimensionality Reduction

### 1. Overview

This folder contains the individual work completed by **Member 5** for the IT2011 Artificial Intelligence and Machine Learning group assignment.

The main responsibility of Member 5 is **Feature Engineering through Principal Component Analysis (PCA) and Dimensionality Reduction**.

The work investigates whether the original features contain overlapping information and whether the number of features can be reduced while retaining most of the important information in the dataset.

---

### 2. Dataset

The project uses the **Breast Cancer Wisconsin (Original)** dataset.

The original dataset contains:

* **699 observations**
* **9 predictive features**
* **1 target/class variable**
* Feature values ranging from **1 to 10**
* Missing values represented using `?`

The nine predictive features are:

1. Clump Thickness
2. Uniformity of Cell Size
3. Uniformity of Cell Shape
4. Marginal Adhesion
5. Single Epithelial Cell Size
6. Bare Nuclei
7. Bland Chromatin
8. Normal Nucleoli
9. Mitoses

The original class values are:

* `2` → Benign
* `4` → Malignant

For machine learning processing, the target can be represented as:

* `0` → Benign
* `1` → Malignant

---

### 3. Main Objective

The main objective of this notebook is to apply **Principal Component Analysis (PCA)** to reduce the dimensionality of the nine original predictor features.

The analysis focuses on:

* Standardizing the features before PCA
* Applying PCA
* Measuring explained variance
* Measuring cumulative explained variance
* Selecting an appropriate number of principal components
* Creating a reduced dataset
* Investigating PCA loadings

---

### 4. Why PCA Was Used

The EDA showed that several predictor variables have relatively strong correlations.

For example, **Uniformity of Cell Size** and **Uniformity of Cell Shape** have a strong correlation.

This indicates that some features contain overlapping information.

PCA can transform the original correlated features into a smaller number of new variables called **principal components**.

This can:

* Reduce the number of dimensions
* Reduce redundancy
* Preserve a large amount of the variation in the data
* Make the data easier to visualize and process
* Potentially make machine learning models more efficient

PCA does not simply remove individual original features. Instead, it creates new features that are combinations of the original features.

---

### 5. Standardization Before PCA

The features were standardized using `StandardScaler` before applying PCA.

This is important because PCA is based on variance.

Standardization puts the features onto a comparable scale so that features with larger variation do not dominate the PCA results.

The workflow used is:

```text
Original Features
       ↓
Handle Missing Values
       ↓
Standardize Features
       ↓
Apply PCA
       ↓
Calculate Explained Variance
       ↓
Select Principal Components
       ↓
Create Reduced Dataset
```

For the final machine learning implementation, preprocessing should be performed inside a pipeline so that transformations are fitted only on the training data.

---

### 6. Missing Values

The `Bare_Nuclei` feature contains missing values represented by `?`.

For the standalone PCA demonstration in this notebook, complete observations are used when demonstrating PCA calculations.

This should not be interpreted as permanently deleting missing observations from the final machine learning dataset.

For the final ML models, missing values are handled using median imputation inside the preprocessing pipeline to avoid data leakage.

---

### 7. Explained Variance

PCA calculates how much of the total variation in the dataset is captured by each principal component.

The project results show approximately:

| Principal Component | Explained Variance |
| ------------------- | -----------------: |
| PC1                 |             65.56% |
| PC2                 |              8.62% |
| PC3                 |              6.00% |
| PC4                 |              5.17% |
| PC5                 |              4.11% |

The first four principal components retain approximately **85.35%** of the total variance.

This means that the nine original features can be represented using four principal components while retaining most of the variation in the data.

---

### 8. Selection of Four Components

Four principal components were selected as a practical balance between dimensionality reduction and information retention.

The cumulative explained variance is approximately:

* 1 PC → 65.56%
* 2 PCs → 74.18%
* 3 PCs → 80.18%
* 4 PCs → 85.35%
* 5 PCs → 89.47%

Therefore, **4 components** were selected for the reduced representation used in the project.

This reduces the feature space from:

```text
9 original features
        ↓
4 principal components
```

This represents a reduction of approximately **55.6% in the number of features**.

---

### 9. PCA Loadings

PCA loadings describe how strongly the original features contribute to each principal component.

The loading analysis helps interpret which original variables have stronger contributions to the different principal components.

The principal components are not individual original variables. Each component is created from a combination of the original standardized features.

Therefore, PCA improves dimensionality reduction but makes the resulting features less directly interpretable than the original variables.

---

### 10. Notebook Contents

The notebook contains the following main sections:

1. Project title and introduction
2. Required library imports
3. Dataset loading
4. Column name assignment
5. Missing-value preparation
6. Conversion of `Bare_Nuclei` to numeric
7. Definition of predictor features
8. Preparation of data for PCA
9. Explanation of dimensionality reduction
10. Standardization using `StandardScaler`
11. Application of PCA
12. Explained variance analysis
13. Cumulative explained variance analysis
14. Explained variance visualization
15. Cumulative variance visualization
16. Selection of four principal components
17. Creation of the four-component dataset
18. PCA loading analysis
19. Loading visualization
20. Saving results
21. Final PCA summary

---

### 11. Results

The generated PCA results are stored in the following locations:

```text
results/
├── eda_visualizations/
│   ├── explained_variance.png
│   └── cumulative_explained_variance.png
│
├── logs/
│
└── outputs/
    ├── pca_explained_variance.csv
    ├── pca_cumulative_variance.csv
    ├── pca_4_components.csv
    └── pca_loadings.csv
```

#### Output files

**`pca_explained_variance.csv`**

Contains the proportion of variance explained by each principal component.

**`pca_cumulative_variance.csv`**

Contains the cumulative variance retained as additional principal components are included.

**`pca_4_components.csv`**

Contains the transformed dataset using the selected four principal components.

**`pca_loadings.csv`**

Contains the contribution/loadings of the original features to the principal components.

---

### 12. Visualizations

The notebook produces visualizations to support the PCA analysis.

#### Explained Variance Plot

Shows the amount of variance explained by each principal component.

#### Cumulative Explained Variance Plot

Shows how much total variance is retained when additional components are included.

These visualizations help justify the selection of four principal components.

---

### 13. Key Findings

The main findings from the PCA analysis are:

* The original dataset contains **9 predictor features**.
* Several features contain overlapping information because of their correlations.
* Standardization was performed before PCA.
* PC1 explains approximately **65.56%** of the variance.
* The first four components retain approximately **85.35%** of the variance.
* Four components were selected as a balance between dimensionality reduction and information retention.
* PCA reduced the feature space from **9 features to 4 components**.
* PCA components are combinations of the original features.
* PCA loadings were examined to understand the contribution of the original variables.

---

### 14. Tools and Libraries

The following Python libraries were used:

* **Pandas** – data loading and manipulation
* **NumPy** – numerical operations
* **Matplotlib** – visualization
* **Seaborn** – visualization
* **Scikit-learn** – standardization and PCA

Main functions/classes used include:

```python
pd.read_csv()
StandardScaler()
PCA()
plt.bar()
plt.plot()
sns.heatmap()
```

---

### 15. Folder Structure

```text
MEMBER_05/
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
    │   ├── explained_variance.png
    │   └── cumulative_explained_variance.png
    │
    ├── logs/
    │
    └── outputs/
        ├── pca_explained_variance.csv
        ├── pca_cumulative_variance.csv
        ├── pca_4_components.csv
        └── pca_loadings.csv
```

---

### 16. Reproducibility

To reproduce the analysis:

1. Place the original dataset inside:

```text
data/raw/
```

2. Open:

```text
notebook.ipynb
```

3. Make sure the required Python libraries are installed.

4. Run the notebook cells from top to bottom.

5. The generated PCA tables and visualizations will be saved inside the `results/` directory.

---

### 17. Member Responsibility

Member 5 is primarily responsible for:

* PCA
* Dimensionality reduction
* Explained variance analysis
* Cumulative explained variance
* Principal component selection
* PCA loadings
* PCA visualizations
* PCA-related outputs

The work is part of the group's overall machine learning pipeline and connects with the feature selection, scaling, and model development stages completed by the other members.

---

### 18. Conclusion

PCA was applied to investigate whether the nine original predictor features could be represented using fewer dimensions.

The analysis showed that the first four principal components retain approximately **85.35% of the total variance**. Therefore, four components provide a reasonable reduced representation of the original nine features.

The PCA results can then be used as an alternative feature representation for machine learning models and compared with models trained using the original features.
