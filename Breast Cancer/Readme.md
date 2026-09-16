# 🩺 Breast Cancer Prediction using Logistic Regression

A Machine Learning classification project that predicts whether a breast tumor is **Benign** or **Malignant** using diagnostic features of breast cancer cells.

The project uses **Logistic Regression** to classify tumors based on measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and other cell characteristics.

---

## 📌 Project Overview

Breast cancer diagnosis can be formulated as a **binary classification problem**, where the model predicts one of two classes:

* **Benign (B)**
* **Malignant (M)**

In this project, the original diagnosis labels are converted into numerical values:

```text
1 → Benign
0 → Malignant
```

The notebook performs data loading, exploratory analysis, preprocessing, feature selection, train-test splitting, model training, prediction, and accuracy evaluation.

---

## 🎯 Objective

The main objective of this project is to build a Machine Learning classification model capable of predicting breast cancer diagnosis from numerical diagnostic measurements.

### Problem Type

**Binary Classification**

### Target Variable

`diagnosis`

| Original Value | Encoded Value | Meaning   |
| -------------- | ------------: | --------- |
| B              |             1 | Benign    |
| M              |             0 | Malignant |

The label mapping is explicitly performed in the notebook using:

```python
data['diagnosis'] = data['diagnosis'].map({"B": 1, "M": 0})
```

---

## 📊 Dataset

The project loads the dataset from:

```text
data.csv
```

The dataset contains:

* **569 observations**
* **33 columns initially**
* 1 target column: `diagnosis`
* 1 identifier column: `id`
* 1 completely empty column: `Unnamed: 32`
* 30 final input features used by the model

## The notebook shows that all 569 observations have values for the actual diagnostic and feature columns, while `Unnamed: 32` contains **569 missing values**.

## 🔍 Dataset Features

The dataset contains measurements grouped into three categories:

### Mean Features

Examples include:

* `radius_mean`
* `texture_mean`
* `perimeter_mean`
* `area_mean`
* `smoothness_mean`
* `compactness_mean`
* `concavity_mean`
* `concave points_mean`
* `symmetry_mean`
* `fractal_dimension_mean`

### Standard Error Features

Examples include:

* `radius_se`
* `texture_se`
* `perimeter_se`
* `area_se`
* `smoothness_se`
* `compactness_se`
* `concavity_se`
* `concave points_se`
* `symmetry_se`
* `fractal_dimension_se`

### Worst Features

Examples include:

* `radius_worst`
* `texture_worst`
* `perimeter_worst`
* `area_worst`
* `smoothness_worst`
* `compactness_worst`
* `concavity_worst`
* `concave points_worst`
* `symmetry_worst`
* `fractal_dimension_worst`

## The notebook ultimately uses **30 numerical features** after removing the target, ID, and empty column.

## 📈 Class Distribution

The dataset contains:

| Diagnosis |   Count |
| --------- | ------: |
| Benign    |     357 |
| Malignant |     212 |
| **Total** | **569** |

This indicates that the dataset contains more benign cases than malignant cases.

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed.

### 1. Load Dataset

```python
data = pd.read_csv('data.csv')
```

### 2. Encode Target Variable

The categorical diagnosis values are converted into numerical labels:

```python
data['diagnosis'] = data['diagnosis'].map({
    "B": 1,
    "M": 0
})
```

### 3. Separate Features and Target

```python
X = data.drop(columns='diagnosis', axis=1)
Y = data['diagnosis']
```

### 4. Remove Empty Column

The `Unnamed: 32` column contains no values and is removed:

```python
X = X.drop(columns='Unnamed: 32', axis=1)
```

### 5. Remove ID Column

The patient/sample identifier is removed because it is not used as a predictive feature:

```python
X = X.drop(columns='id', axis=1)
```

After preprocessing:

```text
X → 569 rows × 30 features
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
```

### Dataset Split

| Dataset      | Samples | Features |
| ------------ | ------: | -------: |
| Full Dataset |     569 |       30 |
| Training Set |     455 |       30 |
| Testing Set  |     114 |       30 |

The notebook confirms these dimensions directly.

---

## 🤖 Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as the classification algorithm.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, Y)
```

Logistic Regression is suitable for this problem because the target contains two classes: Benign and Malignant.

The notebook uses the default `LogisticRegression()` configuration. During training, scikit-learn produced a convergence warning because the solver reached the default maximum of 100 iterations.

---

## 🔮 Prediction

After training, predictions are generated on the test dataset:

```python
y_pred = model.predict(X_test)
```

The model produces binary predictions corresponding to the encoded diagnosis classes.

---

## 📊 Model Performance

The model is evaluated using **Accuracy Score**.

```python
score = accuracy_score(y_test, y_pred)

print("Accuracy Score:", score)
```

### Result

**Accuracy: 96.49%**

```text
Accuracy Score: 0.9649122807017544
```

### Performance Summary

| Model               | Metric   |      Score |
| ------------------- | -------- | ---------: |
| Logistic Regression | Accuracy | **96.49%** |

---

## 🔄 Machine Learning Workflow

```text
                ┌──────────────────┐
                │    data.csv      │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Load Dataset     │
                │    Pandas        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Data Exploration │
                │ head/info/nulls  │
                │ describe/count   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Encode Diagnosis │
                │ B → 1            │
                │ M → 0            │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Remove ID &      │
                │ Empty Column     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Feature Matrix X │
                │ 569 × 30         │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Train/Test Split │
                │ 80% / 20%        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Logistic         │
                │ Regression       │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    Prediction    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Accuracy Score   │
                │     96.49%       │
                └──────────────────┘
```

---

## 🛠️ Technologies Used

### Programming Language

* Python 3.13.6

### Libraries

* **NumPy** — Numerical computing
* **Pandas** — Data manipulation and analysis
* **Scikit-learn** — Machine Learning

  * `train_test_split`
  * `LogisticRegression`
  * `accuracy_score`

The notebook metadata reports Python version **3.13.6**.

---

## 📁 Project Structure

```text
Breast-Cancer-Prediction/
│
├── Breast Cancer.ipynb
├── data.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project

```bash
cd Breast-Cancer-Prediction
```

### 3. Install Dependencies

```bash
pip install numpy pandas scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
Breast Cancer.ipynb
```

Make sure `data.csv` is located in the appropriate working directory before executing the notebook.

---

## 🧠 Key Machine Learning Concepts Practiced

This project demonstrates several fundamental Machine Learning concepts:

* Binary classification
* Data loading with Pandas
* Exploratory Data Analysis
* Missing-value inspection
* Categorical label encoding
* Feature-target separation
* Feature selection
* Train-test splitting
* Logistic Regression
* Model prediction
* Accuracy evaluation

---

## ⚠️ Important Note

The notebook demonstrates a Machine Learning workflow for educational purposes. The reported **96.49% accuracy** is based on the specific train-test split used in the notebook and should not be interpreted as a clinical diagnostic performance measure.

The notebook also shows a Logistic Regression convergence warning with the default `max_iter=100`; additional preprocessing such as feature scaling and/or increasing the iteration limit could be explored in future versions.

---





## 👨‍💻 Author

**Aryan Dongre**

B.Tech — Artificial Intelligence & Data Science

---

## ⭐ Acknowledgement

This project was created as part of Machine Learning practice and learning.

If you found this project useful, consider giving the repository a ⭐ on GitHub.
