# Predicting Diabetes Risk Using Machine Learning

> **A Comparative Analysis of Nine Supervised Classifiers on the CDC BRFSS 2015 Dataset**
>
> *CSP 571 — Machine Learning · Spring 2026*
>
> **Authors:** Yaswanth Moapda · Srujith Borra · Uday Ankam

---

## Abstract

Diabetes mellitus affects over 37 million Americans, with approximately 25% remaining undiagnosed until complications develop. This study evaluates whether machine learning models trained exclusively on self-reported survey data from the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 can achieve clinically meaningful predictive accuracy. We train and compare nine supervised classifiers — Logistic Regression, Lasso, Decision Tree, Random Forest, GBM, XGBoost, SVM, Neural Network, and k-NN — on a class-balanced subset of 67,920 records. Gradient Boosted Machines (GBM) and XGBoost achieve the highest discrimination (AUC ≈ 0.83), but Logistic Regression follows closely (AUC = 0.820), suggesting the underlying feature-outcome relationship is predominantly linear.

**Keywords:** Diabetes prediction · Supervised learning · Gradient boosting · Medical screening · BRFSS

---

## 1. Dataset

| Property | Value |
|---|---|
| **Source** | [UCI ML Repository — CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) |
| **File** | `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` |
| **Raw records** | 70,692 |
| **After preprocessing** | 67,920 |
| **Train / Test** | 47,544 / 20,376 (stratified 70/30) |
| **Features** | 21 (14 binary + 7 continuous) |
| **Target** | `Diabetes_binary` (0 = No, 1 = Yes) |

### Predictors

- **Binary (14):** HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex
- **Continuous (7):** BMI, GenHlth, MentHlth, PhysHlth, Age, Education, Income (z-score standardized)

---

## 2. Project Structure

```
diabetes-ml/
│
├── Diabetes_Complete.R          # Main R pipeline (run this)
├── Diabetes_Complete.Rmd        # R Markdown notebook version
│
├── Diabetes_Report.docx         # Research-paper style report
├── Diabetes_Presentation.pptx   # 10-slide presentation
│
├── data/
│   └── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│
├── README.md
└── .gitignore
```

---

## 3. Reproducing the Analysis

### Prerequisites

```r
install.packages(c(
  "data.table", "dplyr", "ggplot2", "tidyr", "scales", "tibble",
  "caret", "glmnet", "randomForest", "gbm", "e1071", "nnet",
  "xgboost", "class", "rpart", "rpart.plot",
  "pROC", "corrplot", "gridExtra", "grid", "png"
))
```

### Setup

1. Download the dataset from the UCI link above
2. Place the CSV on your Desktop (or update `CSV_PATH` at the top of the script)

### Run

```r
setwd("~/Desktop")
source("Diabetes_Complete.R")
```

The pipeline will produce **15 plots** in **3 windows** in the RStudio plot pane (use the ←/→ arrows to navigate) and print the model comparison table to the console.

---

## 4. Methodology

### Preprocessing Pipeline

| Step | Result | Justification |
|---|---|---|
| Duplicate removal | 1,635 rows removed → 69,057 | Prevents data leakage between train/test |
| Class balancing | 67,920 rows (50/50) | Eliminates majority-class bias in threshold-sensitive classifiers |
| Factor encoding | 14 binary features | Ensures tree-based models treat them categorically |
| Z-score standardization | 7 continuous features | Required for distance- and gradient-based algorithms |
| Stratified 70/30 split | 47,544 train / 20,376 test | Preserves class balance in both partitions |

### Models

The nine models span hypothesis classes deliberately chosen to test different inductive biases:

| Model | Class | Training Data | Configuration |
|---|---|---|---|
| Logistic Regression | Parametric · Linear | Full (47,544) | Binomial GLM, all 21 features |
| Lasso (glmnet) | Parametric · Regularized | Full (47,544) | α=1, λ by 5-fold CV |
| Decision Tree (rpart) | Tree-based · Interpretable | Full (47,544) | cp=0.001, maxdepth=6 |
| Random Forest | Tree-based · Bagging | Full (47,544) | 300 trees, mtry=4 |
| GBM ⭐ | Tree-based · Boosting | Full (47,544) | 500 trees, depth=3, η=0.05 |
| XGBoost | Tree-based · Reg. Boosting | Full (47,544) | 400 rounds, η=0.05, depth=4 |
| SVM | Kernel · Margin-based | 10% (4,756) | Linear, C=1 |
| Neural Network | Non-parametric · Deep | 10% (4,756) | 1 hidden layer, 8 units, decay=0.01 |
| k-NN | Distance · Local | 10% (4,756) | k=21 |

> **Note on subsampling:** SVM, Neural Network, and k-NN were trained on 10% of the training data due to the O(n²) memory cost of the SVM kernel matrix and the O(n_train × n_test) inference cost of k-NN. Their results represent lower bounds on full-data performance.

---

## 5. Results

### Performance Comparison (Test Set, n = 20,376)

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---|---|---|---|---|
| **GBM** ⭐ | **0.7486** | **0.7892** | 0.7081 | **0.7584** | **0.8266** |
| **XGBoost** | 0.7482 | 0.7861 | 0.7104 | 0.7576 | 0.8255 |
| Logistic | 0.7456 | 0.7673 | 0.7239 | 0.7510 | 0.8197 |
| Lasso | 0.7454 | 0.7675 | 0.7233 | 0.7509 | 0.8197 |
| SVM | 0.7442 | 0.7629 | 0.7256 | 0.7489 | 0.8183 |
| RandomForest | 0.7411 | 0.7847 | 0.6974 | 0.7519 | 0.8135 |
| NeuralNet | 0.7266 | 0.7487 | 0.7046 | 0.7325 | 0.7983 |
| DecisionTree | 0.7180 | 0.7345 | 0.7015 | 0.7226 | 0.7895 |
| kNN | 0.6892 | 0.6914 | 0.6870 | 0.6899 | 0.7548 |

### Key Findings

1. **GBM and XGBoost lead** — but the margin over Logistic Regression is only 0.007 AUC
2. **Linearity of signal** — Logistic Regression's near-equivalent performance reveals the feature-outcome relationship is predominantly linear
3. **Lasso retained all features** — λ ≈ 8e-4 confirms no feature redundancy
4. **k-NN underperforms** — curse of dimensionality on 21-feature space
5. **EDA validated** — RF importance scores match EDA findings (GenHlth, BMI, Age, HighBP, Income)
6. **GBM is best for deployment** — highest sensitivity and lowest false negatives (2,148)

---

## 6. Visualizations

The pipeline generates 15 plots organized into 3 windows:

| Window | Plots | Content |
|---|---|---|
| 1 | 01–05 | Class balance · BMI · Age · Correlation · GenHlth |
| 2 | 06–10 | Boxplots · Binary features · Income · BMI vs GenHlth · Education |
| 3 | 11–15 | ROC curves · RF importance · Comparison · Sens/Spec · Confusion matrix |

---

## 7. Limitations

1. **Subsampling for SVM, NN, k-NN** — results are lower bounds on full-data performance
2. **50/50 class balance** ≠ ~14% population prevalence — probability recalibration required for deployment
3. **Self-report bias** — all features are self-reported via telephone interview

## 8. Future Work

- Full-data training for SVM, Neural Network, and k-NN
- Hyperparameter optimization via grid search or Bayesian methods
- Ensemble stacking combining GBM, XGBoost, and Logistic Regression
- Threshold optimization for clinical sensitivity targets
- Probability calibration via Platt scaling or isotonic regression
- Validation on more recent BRFSS cycles (2020+)

---

## References

1. Centers for Disease Control and Prevention. *National Diabetes Statistics Report.* 2022.
2. CDC. *Behavioral Risk Factor Surveillance System: 2015 Survey Data.* 2016.
3. UCI Machine Learning Repository. *CDC Diabetes Health Indicators Dataset.* 2023.
4. Tibshirani, R. Regression shrinkage and selection via the lasso. *JRSS-B*, 58(1):267–288, 1996.
5. Breiman, L. Random forests. *Machine Learning*, 45(1):5–32, 2001.
6. Friedman, J. H. Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5):1189–1232, 2001.
7. Chen, T., & Guestrin, C. XGBoost: A scalable tree boosting system. *KDD '16*, 785–794, 2016.
8. Cortes, C., & Vapnik, V. Support-vector networks. *Machine Learning*, 20(3):273–297, 1995.
9. Cover, T., & Hart, P. Nearest neighbor pattern classification. *IEEE Trans. Information Theory*, 13(1):21–27, 1967.

---

## License

Created for academic purposes as part of CSP 571 — Machine Learning, Spring 2026.
