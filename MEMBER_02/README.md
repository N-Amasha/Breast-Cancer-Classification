# IT2011 – Breast Cancer Wisconsin (Original)

## Progress Review I – Member 2

### 1. Overview

This folder contains the work completed by Member 2 for the IT2011 Artificial Intelligence and Machine Learning project.

The project uses the **Breast Cancer Wisconsin (Original)** dataset for a binary classification problem.

Member 2 is mainly responsible for **Exploratory Data Analysis (EDA)**.

---

### 2. Dataset

**Dataset:** Breast Cancer Wisconsin (Original)

The original dataset contains:

* 699 observations
* 9 predictive features
* 1 sample ID column
* 1 class/target column

The target classes are:

* `2` – Benign
* `4` – Malignant

The raw dataset is stored in:

```text id="4xq7wv"
data/raw/breast-cancer-wisconsin.data
```

---

### 3. Member 2 Responsibilities

The following EDA tasks were performed:

1. Loaded and inspected the dataset.
2. Investigated the class distribution.
3. Calculated descriptive statistics for the predictive features.
4. Created histograms to study feature distributions.
5. Created boxplots to investigate feature spread and potential statistical outliers.
6. Created a correlation matrix to identify relationships between predictive features.
7. Compared feature distributions between benign and malignant classes.
8. Interpreted the main findings from the EDA.

---

### 4. Class Distribution

The original dataset contains more benign cases than malignant cases.

The class distribution is approximately:

* **Benign (Class 2): 458 observations – 65.5%**
* **Malignant (Class 4): 241 observations – 34.5%**

Therefore, the dataset is moderately imbalanced.

The class distribution was visualized using a count plot.

---

### 5. Descriptive Statistics

Descriptive statistics were calculated for the nine predictive features.

The analysis included:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* Median
* 75th percentile
* Maximum

The features generally have values ranging from **1 to 10**.

These statistics were used to understand the central tendency and spread of the predictive features.

---

### 6. Histograms

Histograms were created to examine the distribution of the predictive features.

The analysis showed that many features have a higher concentration of observations at lower values. Some features, including **Bare_Nuclei** and the uniformity-related features, showed noticeable skewness.

The histograms are stored in:

```text id="2r9h0q"
results/eda_visualizations/
```

---

### 7. Boxplots and Outlier Analysis

Boxplots were used to examine the median, spread, and potential statistical outliers of the predictive features.

Some statistical outliers were observed. However, the feature values remained within the valid range of **1 to 10**.

Therefore, the observations were not automatically removed as outliers because unusual values may still represent valid observations and potentially useful information for classification.

---

### 8. Correlation Analysis

A correlation matrix was created using the nine predictive features.

Several strong positive relationships were identified.

The strongest relationship was approximately:

```text id="8jgnjp"
Uniformity_Cell_Size
        ↕
Uniformity_Cell_Shape

Correlation ≈ 0.91
```

Other relatively strong relationships were also observed between several uniformity, chromatin, nucleoli, and cell-size features.

These correlations indicate that some features contain overlapping information. This finding was useful for the later feature engineering and dimensionality reduction stages.

---

### 9. Feature vs Class Analysis

Feature distributions were compared between the benign and malignant classes.

Noticeable differences were observed in several features, particularly:

* Uniformity Cell Size
* Uniformity Cell Shape
* Bare Nuclei
* Clump Thickness

These features showed different patterns between the two classes and may therefore provide useful information for classification.

The analysis does not mean that an individual feature alone can diagnose cancer. The features are considered together by the machine learning models.

---

### 10. EDA Visualizations

The main visualizations produced during the analysis include:

```text id="e9c5iz"
results/
└── eda_visualizations/
    ├── class_distribution.png
    ├── feature_histograms.png
    ├── feature_boxplots.png
    ├── correlation_matrix.png
    └── class_feature_boxplots.png
```

---

### 11. Main EDA Findings

The EDA provided the following important findings:

* The dataset contains more benign cases than malignant cases.
* Several features have distributions concentrated toward lower values.
* Some features show skewed distributions.
* Statistical outliers are present, but the values remain within the valid 1–10 range.
* Several predictive features are strongly correlated.
* Uniformity Cell Size and Uniformity Cell Shape have the strongest observed correlation, approximately 0.91.
* Several features show noticeable distribution differences between benign and malignant cases.

These findings helped guide the later preprocessing, feature engineering, and machine learning stages.

---

### 12. Notebook

The complete implementation and visualizations can be found in:

```text id="82k4mj"
notebook.ipynb
```

---

### 13. Folder Structure

```text id="z4s1ja"
MEMBER_02/
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

### 14. Reproducibility

To reproduce the EDA:

1. Place the original dataset in `data/raw/`.
2. Open `notebook.ipynb`.
3. Run the notebook cells in order.
4. The EDA visualizations and statistical outputs will be generated in the `results/` directory.
