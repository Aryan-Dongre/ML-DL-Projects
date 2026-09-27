# 🩺 Diabetes Prediction using SVM

A Machine Learning classification project that uses a **Support Vector Machine (SVM)** to predict whether a person is diabetic based on medical diagnostic measurements.

The project implements the complete machine learning workflow, including data loading, preprocessing, train-test splitting, feature scaling, SVM model training, accuracy evaluation, and a predictive system for making predictions on new input data.

---

## 📌 Project Overview

Diabetes is a common health condition that can be predicted using various medical measurements.

In this project, a **Support Vector Machine (SVM)** classifier is trained on the `diabetes.csv` dataset to predict whether a person is diabetic.

The target variable is:

```text
Outcome
```

where:

```text
0 → Not Diabetic
1 → Diabetic
```

The model uses the following medical features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

## 🎯 Objective

The main objective of this project is to develop a machine learning model that can classify a person as:

```text
Not Diabetic
```

or

```text
Diabetic
```

based on their medical measurements.

---

## 📊 Dataset

The project uses:

```text
diabetes.csv
```

The dataset contains medical diagnostic measurements along with the `Outcome` column indicating whether the person is diabetic.

---

## 📋 Features

The model uses the following input features:

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Skin fold thickness |
| `Insulin` | Insulin level |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age of the person |

### Target Variable

```text
Outcome
```

Target values:

| Value | Meaning |
|---:|---|
| `0` | Not Diabetic |
| `1` | Diabetic |

---

# 🧠 Why SVM?

The project uses a **Support Vector Machine (SVM)** for classification.

SVM is a supervised machine learning algorithm that is particularly useful for classification problems.

The basic idea of SVM is to find a **decision boundary (hyperplane)** that separates different classes.

For this project, the SVM tries to find a boundary that separates:

```text
Not Diabetic (0)
        vs
Diabetic (1)
```

---

## 🔎 How SVM Works

Imagine the data points as two different groups:

```text
       Not Diabetic          Diabetic

           ● ●
        ●  ● ●

--------------------------  ← Decision Boundary

                       ● ●
                    ●  ● ●
```

SVM tries to find a decision boundary that separates the classes while maximizing the distance between the boundary and the closest data points.

These closest points are called **Support Vectors**.

```text
Support Vector
      ↓
      ●

      |<---- Margin ---->|

--------------------------  ← Decision Boundary

      |<---- Margin ---->|

      ●
      ↑
Support Vector
```

The larger the margin, the better the separation between the classes in many cases.

---

## ⚡ Why SVM is Used in This Project

SVM is suitable for this project because:

### 1. Classification Problem

The target contains two classes:

```text
0 → Not Diabetic
1 → Diabetic
```

Therefore, this is a binary classification problem.

---

### 2. Works Well with Numerical Features

The dataset contains numerical medical measurements such as:

```text
Glucose
BMI
BloodPressure
Insulin
Age
```

SVM can work effectively with numerical feature data.

---

### 3. Feature Scaling is Important

SVM is sensitive to the scale of input features.

For example:

```text
Age       → values around 20–80
Glucose   → values around 50–200
Insulin   → values can be much larger
```

If the features are not scaled, features with larger numerical values can have a greater influence on the model.

Therefore, this project uses:

```python
StandardScaler()
```

to standardize the features.

---

### 4. Linear Kernel

The model uses:

```python
svm.SVC(kernel='linear')
```

A linear kernel attempts to separate the two classes using a linear decision boundary.

This makes the model relatively simple and easy to understand for a first SVM classification project.

---

# 🔄 Machine Learning Workflow

The complete workflow is:

```text
Load Dataset
      ↓
Explore Dataset
      ↓
Check Missing Values
      ↓
Separate Features and Target
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Create SVM Model
      ↓
Train Model
      ↓
Make Predictions
      ↓
Calculate Accuracy
      ↓
Build Predictive System
```

---

# 🧹 Data Preprocessing

## 1. Load Dataset

The dataset is loaded using Pandas:

```python
import pandas as pd

df = pd.read_csv("diabetes.csv")
```

The first few rows are inspected using:

```python
df.head()
```

---

## 2. Check Missing Values

Missing values are checked using:

```python
df.isnull().sum()
```

This helps identify whether any columns contain missing values.

---

## 3. Separate Features and Target

The target column `Outcome` is separated from the input features:

```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
```

Therefore:

```text
X → Input Features
y → Target
```

---

# ✂️ Train-Test Split

The dataset is divided into training and testing sets using:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Configuration

| Parameter | Value |
|---|---:|
| Test Size | 20% |
| Training Size | 80% |
| Random State | 42 |

Using `random_state=42` makes the split reproducible.

---

# 📏 Feature Scaling

The project uses `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

This is important because the test data should not influence the scaling parameters learned from the training data.

---

# 🤖 SVM Model

The Support Vector Machine classifier is created using:

```python
from sklearn import svm

model = svm.SVC(
    kernel="linear"
)
```

The model uses a **linear kernel**.

---

## 🏋️ Model Training

The SVM is trained using:

```python
model.fit(
    X_train_scaled,
    y_train
)
```

During training, the model learns a decision boundary that separates the two classes.

---

# 🔮 Making Predictions

After training, predictions are made using:

```python
y_pred = model.predict(X_test_scaled)
```

The predictions contain:

```text
0 → Not Diabetic
1 → Diabetic
```

---

# 📈 Model Evaluation

The project evaluates the model using **accuracy**.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_pred,
    y_test
)

print(
    "The Accuracy : ",
    accuracy * 100
)
```

### Accuracy

Accuracy represents the percentage of test samples that were classified correctly.

```text
Accuracy =
Correct Predictions
------------------- × 100
Total Predictions
```

---

# 🔬 Predictive System

The notebook also includes a simple predictive system that allows a new person's medical measurements to be passed to the trained model.

Example input:

```python
input_data = (
    6,
    148,
    72,
    35,
    0,
    33.6,
    0.627,
    50
)
```

These values correspond to:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
```

---

## 🧮 Preparing New Input

The input is converted into a NumPy array:

```python
import numpy as np

input_array = np.asarray(input_data)
```

It is then reshaped into a single sample containing 8 features:

```python
input_array = input_array.reshape(1, -1)
```

---

## 📏 Scaling New Data

Because the SVM was trained using scaled data, the new input must also be scaled using the same scaler:

```python
std_data = scaler.transform(input_array)
```

The scaled input should then be passed to the model:

```python
prediction = model.predict(std_data)
```

---

## ⚠️ Important Correction

The original notebook contains:

```python
std_data = scaler.transform(input_array)

prediction = model.predict(input_array)
```

This should be changed to:

```python
std_data = scaler.transform(input_array)

prediction = model.predict(std_data)
```

### Why?

The SVM was trained using:

```python
X_train_scaled
```

Therefore, new input data must also be transformed using the same `StandardScaler`.

The correct pipeline is:

```text
Raw Input
    ↓
StandardScaler
    ↓
Scaled Input
    ↓
SVM
    ↓
Prediction
```

---

# 🩺 Prediction Output

The prediction can be converted into a readable result:

```python
if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
```

The final output will be either:

```text
The person is not diabetic
```

or:

```text
The person is diabetic
```

> **Note:** This project is an educational machine-learning project and should not be used as a medical diagnosis or substitute for professional medical advice.

---

# 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Support Vector Machine**
- **Jupyter Notebook**

---

# 📦 Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
```

---

# 📂 Project Structure

A suggested GitHub repository structure:

```text
Diabetes-Prediction/
│
├── Diabetes Prediction.ipynb
│
├── diabetes.csv
│
├── README.md
│
└── requirements.txt
```

---

# 🚀 How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Diabetes-Prediction.git
```

Replace `your-username` with your GitHub username.

---

## 2. Navigate to the Project

```bash
cd Diabetes-Prediction
```

---

## 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or create a `requirements.txt` file:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Then run:

```bash
pip install -r requirements.txt
```

---

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Diabetes Prediction.ipynb
```

Make sure:

```text
diabetes.csv
```

is located in the same directory as the notebook.

---

# 🔄 Complete Project Pipeline

```text
                  Diabetes Dataset
                         │
                         ▼
                  Data Exploration
                         │
                         ▼
                Check Missing Values
                         │
                         ▼
                Feature / Target Split
                         │
                         ▼
                  Train-Test Split
                         │
                         ▼
                   StandardScaler
                         │
                         ▼
                 Scaled Training Data
                         │
                         ▼
                  Linear SVM (SVC)
                         │
                         ▼
                     Training
                         │
                         ▼
                   Test Prediction
                         │
                         ▼
                      Accuracy
                         │
                         ▼
                 New Patient Input
                         │
                         ▼
                     Scaling
                         │
                         ▼
                  SVM Prediction
                         │
                         ▼
              Diabetic / Not Diabetic
```

---

# 🎓 Key Learning Outcomes

Through this project, the following concepts were implemented and practiced:

- Binary Classification
- Support Vector Machine
- Linear Kernel
- Decision Boundary
- Support Vectors
- Margin
- Feature Scaling
- StandardScaler
- Train-Test Split
- Model Training
- Model Prediction
- Accuracy Score
- NumPy Arrays
- Predictive Systems
- Scikit-learn

---

# 🔮 Future Improvements

The project can be improved by adding:

### 📊 More Evaluation Metrics

Instead of using only accuracy:

- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

### 🤖 Model Comparison

The SVM model can be compared with:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Naive Bayes
- XGBoost

### ⚙️ SVM Hyperparameter Tuning

Different SVM configurations can be tested:

```text
C
Kernel
Gamma
```

For example:

```python
svm.SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
```

This can be compared with the current linear SVM.

### 🌐 Deployment

The trained model could be deployed using:

- Flask
- FastAPI
- Streamlit

A web application could allow users to enter the medical measurements and receive the model's predicted class.

---

# 💡 Key Takeaways

This project demonstrates how an SVM can be used to solve a binary classification problem.

The most important pipeline is:

```text
Raw Medical Data
       ↓
Feature Scaling
       ↓
Linear SVM
       ↓
Learn Decision Boundary
       ↓
Class Prediction
```

Feature scaling is especially important for SVM because the algorithm relies on distances and margins between data points.

---

# 👨‍💻 Author

## Aryan Dongre

**B.Tech — Artificial Intelligence & Data Science**

### Interests

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Data Science
- AI Engineering
- Backend Development

---

# ⭐ Acknowledgement

This project was developed as part of my learning journey in:

**Machine Learning • Classification • Support Vector Machines • Scikit-learn**

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

# 📜 License

This project is intended for **educational and learning purposes**.
