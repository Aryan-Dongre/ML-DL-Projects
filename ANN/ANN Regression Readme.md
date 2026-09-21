# ⚡ ANN Regression — Power Plant Energy Prediction

A Deep Learning regression project that uses an **Artificial Neural Network (ANN)** built with **PyTorch** to predict the electrical energy output of a Combined Cycle Power Plant.

---

## 📌 Project Overview

This project uses historical power plant data to predict the **net hourly electrical energy output (`PE`)** based on four environmental and operational features:

- **AT** — Ambient Temperature
- **V** — Exhaust Vacuum
- **AP** — Ambient Pressure
- **RH** — Relative Humidity

A feed-forward **Artificial Neural Network** is implemented using **PyTorch** and trained using the **Adam optimizer** with **Mean Squared Error (MSE)** as the loss function.

The project demonstrates the complete workflow of a neural-network-based regression problem, including data preprocessing, feature scaling, model creation, training, validation, model saving, and evaluation.

---

## 🎯 Objective

The main objective of this project is to build a neural network capable of predicting the power plant's electrical energy output using environmental and operational measurements.

### Input Features

```text
AT
V
AP
RH
```

### Target

```text
PE
```

---

## 📊 Dataset

The project uses the `Powerplant.csv` dataset.

### Dataset Information

| Property | Value |
|---|---:|
| Total Observations | 9,568 |
| Input Features | 4 |
| Target Variable | PE |
| Missing Values | None |
| Training Samples | 7,654 |
| Testing Samples | 1,914 |

---

## 📋 Features

| Feature | Description |
|---|---|
| `AT` | Ambient Temperature |
| `V` | Exhaust Vacuum |
| `AP` | Ambient Pressure |
| `RH` | Relative Humidity |
| `PE` | Net Hourly Electrical Energy Output |

The four input variables are used by the ANN to predict `PE`.

---

## 🧠 Artificial Neural Network Architecture

The project uses a fully connected feed-forward Artificial Neural Network.

```text
                Input
                  │
                  ▼
          ┌───────────────┐
          │  4 Features   │
          │ AT, V, AP, RH  │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ Linear (4 → 6)│
          └───────┬───────┘
                  │
                  ▼
             ReLU
                  │
                  ▼
          ┌───────────────┐
          │ Linear (6 → 6)│
          └───────┬───────┘
                  │
                  ▼
             ReLU
                  │
                  ▼
          ┌───────────────┐
          │ Linear (6 → 1)│
          └───────┬───────┘
                  │
                  ▼
          Predicted PE
```

### Model Structure

```python
ANN(
    Linear(4, 6),
    ReLU(),
    Linear(6, 6),
    ReLU(),
    Linear(6, 1)
)
```

---

## ⚙️ Technologies Used

The following technologies and Python libraries were used:

- **Python**
- **PyTorch**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

---

## 🔄 Machine Learning Workflow

The project follows the following workflow:

```text
Load Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Check Missing Values
     │
     ▼
Separate Features and Target
     │
     ▼
Train-Test Split
     │
     ▼
Feature Scaling
     │
     ▼
Convert Data to PyTorch Tensors
     │
     ▼
Create TensorDataset
     │
     ▼
Create DataLoader
     │
     ▼
Build ANN Model
     │
     ▼
Train Model
     │
     ▼
Validate Model
     │
     ▼
Save Best Model
     │
     ▼
Evaluate Model
```

---

## 🧹 Data Preprocessing

### 1. Load Dataset

The power plant dataset is loaded using Pandas.

```python
import pandas as pd

data = pd.read_csv("Powerplant.csv")
```

---

### 2. Separate Features and Target

The input features are separated from the target variable.

```python
X = data.drop("PE", axis=1)
y = data["PE"]
```

The model uses the following features:

```text
AT
V
AP
RH
```

to predict:

```text
PE
```

---

### 3. Train-Test Split

The dataset is divided into training and testing datasets using an 80:20 split.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces:

- **Training samples:** 7,654
- **Testing samples:** 1,914

---

### 4. Feature Scaling

The input features are standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Feature scaling helps the neural network train more effectively by putting the input features on a similar scale.

---

## 🔢 Converting Data into PyTorch Tensors

The scaled data is converted into PyTorch tensors before being used by the neural network.

```python
import torch

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1, 1)
```

---

## 📦 TensorDataset and DataLoader

PyTorch's `TensorDataset` and `DataLoader` are used to perform mini-batch training.

```python
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

### Batch Size

```text
32
```

The training data is processed in batches of 32 samples.

---

## 🏗️ Building the ANN Model

The neural network is implemented using PyTorch's `nn.Module`.

The architecture contains:

- Input layer with 4 neurons
- First hidden layer with 6 neurons
- ReLU activation
- Second hidden layer with 6 neurons
- ReLU activation
- Output layer with 1 neuron

The final output represents the predicted power output.

---

## 🔥 Model Training

The model is trained for:

```text
100 Epochs
```

### Loss Function

The project uses **Mean Squared Error (MSE)**:

```python
criterion = nn.MSELoss()
```

MSE measures the average squared difference between the actual and predicted values.

---

### Optimizer

The **Adam optimizer** is used to update the model's weights.

```python
optimizer = torch.optim.Adam(
    model.parameters()
)
```

Adam is commonly used for training neural networks because it adapts the learning rate for individual parameters during optimization.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---:|
| Problem Type | Regression |
| Framework | PyTorch |
| Epochs | 100 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | MSELoss |
| Activation Function | ReLU |
| Input Neurons | 4 |
| Hidden Layer 1 | 6 neurons |
| Hidden Layer 2 | 6 neurons |
| Output Neurons | 1 |

---

## 📉 Training and Validation Loss

Training and validation losses are tracked during model training.

The loss curves help visualize how the model learns over the epochs.

```python
plt.figure(figsize=(8, 8))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
```

The graph can be used to observe:

- Training progress
- Validation performance
- Model convergence
- Possible overfitting

---

## 💾 Model Saving

During training, the best-performing model is saved to:

```text
best_model.pt
```

The `.pt` file contains the trained PyTorch model parameters.

This allows the trained model to be reused later without training it again.

---

## 📈 Model Evaluation

After training, the model is evaluated on the test dataset.

The project uses **Mean Squared Error (MSE)** and **R² Score** to evaluate regression performance.

### Mean Squared Error

```text
MSE ≈ 18.62
```

### R² Score

```text
R² ≈ 93.49%
```

### Performance Summary

| Metric | Result |
|---|---:|
| Training MSE | ~20.40 |
| Validation MSE | ~18.62 |
| Testing MSE | ~18.62 |
| R² Score | ~93.49% |

The R² score indicates that the trained ANN captures a substantial portion of the variation in the target variable.

---

## 📊 Results

The trained neural network successfully learns the relationship between:

```text
Ambient Temperature
        +
Exhaust Vacuum
        +
Ambient Pressure
        +
Relative Humidity
        ↓
Electrical Energy Output
```

The model achieves an R² score of approximately **93.49%** on the evaluated test data.

---

## 📂 Project Structure

```text
ANN-Regression/
│
├── ANN for Regression(1).ipynb
│
├── Powerplant.csv
│
├── best_model.pt
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ANN-Regression.git
```

Replace `your-username` with your GitHub username.

---

### 2. Navigate to the Project Directory

```bash
cd ANN-Regression
```

---

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

Alternatively, if the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
ANN for Regression(1).ipynb
```

Run the notebook cells sequentially.

---

## 📝 Requirements

The project requires the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
jupyter
```

You can install them using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch jupyter
```

---

## 🔮 Future Improvements

The project can be further improved by:

- Performing hyperparameter tuning
- Experimenting with different ANN architectures
- Increasing or decreasing the number of neurons
- Adding more hidden layers
- Experimenting with different activation functions
- Implementing Dropout
- Implementing Early Stopping
- Comparing ANN with Linear Regression
- Comparing ANN with Random Forest Regression
- Adding additional regression metrics
- Building a prediction interface
- Deploying the model using Flask or FastAPI
- Creating a web application for real-time prediction

---

## 🎓 Key Learning Outcomes

Through this project, the following concepts were implemented and practiced:

- Artificial Neural Networks
- Regression using Deep Learning
- PyTorch
- `nn.Module`
- Neural network layers
- ReLU activation
- Forward propagation
- Mean Squared Error
- Adam optimizer
- Backpropagation
- Mini-batch training
- Tensor conversion
- `TensorDataset`
- `DataLoader`
- Feature scaling
- Train-test splitting
- Training and validation
- Loss visualization
- Model evaluation
- R² score
- PyTorch model saving

---

## 💡 Key Takeaways

This project demonstrates how a relatively simple feed-forward ANN can be used for a real-world regression problem.

The overall process can be summarized as:

```text
Data
 ↓
Preprocessing
 ↓
Scaling
 ↓
PyTorch Tensors
 ↓
ANN
 ↓
Training
 ↓
Validation
 ↓
Evaluation
 ↓
Prediction
```

---

## 👨‍💻 Author

### Aryan Dongre

**B.Tech — Artificial Intelligence & Data Science**

Interested in:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Data Science
- AI Engineering
- Backend Development

---

## ⭐ Acknowledgement

This project was developed as part of my learning journey in **Machine Learning, Deep Learning, Artificial Neural Networks, and PyTorch**.

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for **educational and learning purposes**.
