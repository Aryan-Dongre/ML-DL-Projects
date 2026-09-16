# 🪙 Gold Price Prediction

A Machine Learning project that predicts **gold prices (`GLD`)** using financial market indicators such as the S&P 500 (`SPX`), crude oil (`USO`), silver (`SLV`), and the EUR/USD exchange rate.

The project explores the relationship between these financial indicators and gold prices using **Random Forest Regression** and **Linear Regression**.

---

## 📌 Project Overview

Gold prices are influenced by several economic and financial factors. This project uses historical financial data to build regression models capable of estimating gold prices from related market indicators.

The notebook covers:

* Data loading and inspection
* Exploratory Data Analysis
* Statistical analysis
* Correlation analysis
* Feature and target selection
* Train-test splitting
* Random Forest Regression
* Linear Regression
* Gold price prediction
* R² score evaluation
* Actual vs predicted visualization

---

## 📊 Dataset

The project uses:

```text
gld_price_data.csv
```

The dataset contains **2,290 records and 6 columns**.

### Dataset Features

| Feature   | Description                      |
| --------- | -------------------------------- |
| `Date`    | Date of the observation          |
| `SPX`     | S&P 500 stock market index       |
| `GLD`     | Gold ETF price — Target Variable |
| `USO`     | United States Oil Fund price     |
| `SLV`     | Silver ETF price                 |
| `EUR/USD` | EUR to USD exchange rate         |

The dataset contains **no missing values** across the six columns.

---

## 🎯 Target Variable

The target variable is:

```text
GLD
```

`GLD` represents the gold-price-related value that the models attempt to predict.

The `Date` column and target `GLD` are removed from the input features:

```python
X = data.drop(['GLD', 'Date'], axis=1)
y = data['GLD']
```

Therefore, the model uses:

```text
SPX
USO
SLV
EUR/USD
```

as input features.

---

## 🔍 Exploratory Data Analysis

The notebook performs several initial data-analysis operations, including:

* `head()` to inspect the dataset
* `shape` to determine dataset dimensions
* `isnull().sum()` to check missing values
* `describe()` for statistical summaries
* Data type inspection
* Correlation analysis
* Data visualization

### Dataset Size

```text
Rows    : 2290
Columns : 6
```

---

## 📈 Statistical Summary

Some descriptive statistics from the dataset include:

| Variable  |    Mean | Minimum | Maximum |
| --------- | ------: | ------: | ------: |
| `SPX`     | 1654.32 |  676.53 | 2872.87 |
| `GLD`     |  122.73 |   70.00 |  184.59 |
| `USO`     |   31.84 |    7.96 |  117.48 |
| `SLV`     |   20.08 |    8.85 |   47.26 |
| `EUR/USD` |    1.28 |    1.04 |    1.60 |

---

## 🤖 Machine Learning Models

Two regression approaches are implemented in the notebook.

### 1. Random Forest Regression

The primary Random Forest model is created using:

```python
regressor = RandomForestRegressor(n_estimators=100)
```

The model contains **100 decision trees** and is trained on the training dataset.

Predictions are generated using:

```python
y_pred = regressor.predict(X_test)
```

### 2. Linear Regression

The notebook also implements a Linear Regression model:

```python
model = LinearRegression()
```

The trained model generates gold-price predictions on the test dataset.

---

## ✂️ Train-Test Split

The dataset is divided into training and testing data using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces:

```text
Training samples : 1832
Testing samples  : 458
Total samples    : 2290
```

---

## 📊 Model Performance

### Random Forest Regression

The Random Forest model achieved:

```text
R² Score: 0.9899833337132089
```

or approximately:

### **98.998% R² Score**

on the test dataset.

### Linear Regression

The Linear Regression model achieved:

```text
R² Score: 0.8975640982991402
```

or approximately:

### **89.756% R² Score**

on the test dataset.

---

## 📋 Model Comparison

| Model                    |    R² Score |
| ------------------------ | ----------: |
| Random Forest Regression | **0.98998** |
| Linear Regression        | **0.89756** |

## These scores are the results recorded in the notebook's test-set evaluation.

## 📉 Visualization

The notebook includes visualization for comparing:

```text
Actual Gold Price
        vs
Predicted Gold Price
```

This helps visually examine how closely the model's predictions follow the actual target values.

---

## 🔄 Project Workflow

```text
Historical Financial Dataset
          ↓
      Load Dataset
          ↓
    Data Inspection
          ↓
   Missing Value Check
          ↓
 Exploratory Data Analysis
          ↓
  Correlation Analysis
          ↓
 Feature & Target Selection
          ↓
    Train-Test Split
          ↓
 ┌──────────────────────┐
 │                      │
 ▼                      ▼
Random Forest       Linear Regression
 │                      │
 ▼                      ▼
Predictions          Predictions
 │                      │
 └──────────┬───────────┘
            ↓
       R² Evaluation
            ↓
      Visualization
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation and analysis
* **NumPy** — Numerical computing
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualization
* **Scikit-learn** — Machine learning

## The notebook imports Pandas, NumPy, Matplotlib and Seaborn for analysis and uses scikit-learn's `train_test_split`, `LinearRegression`, `RandomForestRegressor`, and metrics functionality.

## 📁 Project Structure

```text
Gold-Price-Prediction/
│
├── Gold Price.ipynb
├── gld_price_data.csv
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
cd Gold-Price-Prediction
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the Notebook

```text
Gold Price.ipynb
```

Run the notebook cells sequentially to reproduce the analysis and model results.

---

## 🎓 Key Learning Outcomes

This project demonstrates practical understanding of:

* Regression problems
* Financial dataset analysis
* Exploratory Data Analysis
* Correlation analysis
* Feature selection
* Train-test splitting
* Random Forest Regression
* Linear Regression
* Model prediction
* R² evaluation
* Actual vs predicted visualization

---

## 🔮 Future Improvements

Possible improvements include:

* Add more financial and macroeconomic indicators
* Perform feature scaling and transformation experiments
* Tune Random Forest hyperparameters
* Compare additional regression algorithms
* Use cross-validation
* Perform time-series-specific validation
* Add MAE and RMSE comparisons
* Build an interactive gold-price prediction application
* Deploy the trained model as an API
* Develop a time-series forecasting version using models designed specifically for sequential data

---

## ⚠️ Disclaimer

This project is created for **educational and Machine Learning practice purposes**.

The predictions generated by this model should not be considered financial advice or a guarantee of future gold prices. Financial markets are affected by many factors that may not be represented in this dataset.

---

## 👨‍💻 Author

**Aryan Dongre**

This project is part of my Machine Learning practice and demonstrates the application of regression techniques to a financial price-prediction problem.
