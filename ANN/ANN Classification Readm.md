# 🧠 ANN for Classification — PyTorch

A Deep Learning classification project that implements an **Artificial Neural Network (ANN)** using **PyTorch** to classify data from the **Date Fruit Dataset**.

The project covers the complete workflow of a classification problem, including data loading, preprocessing, label encoding, feature scaling, train-test splitting, PyTorch tensor conversion, DataLoader creation, ANN model development, training, and evaluation.

---

## 📌 Project Overview

This project demonstrates how an Artificial Neural Network can be used for a multi-class classification problem.

The dataset used in this project is:

```text
10 DateFruit_Dataset.csv
```

The target column is:

```text
Class
```

The input features are all columns except `Class`.

The classification model is implemented using **PyTorch** and consists of multiple fully connected layers with **ReLU activation functions**.

---

## 🎯 Objective

The main objective of this project is to build and train an Artificial Neural Network that can classify observations into different fruit classes based on the available numerical features.

### Input

All columns except:

```text
Class
```

are used as input features.

### Target

```text
Class
```

The target labels are converted from categorical values into numerical labels using `LabelEncoder`.

---

## 📊 Dataset

The project uses the **Date Fruit Dataset** stored in:

```text
10 DateFruit_Dataset.csv
```

The dataset is loaded using Pandas:

```python
df = pd.read_csv("10 DateFruit_Dataset.csv")
```

The notebook performs basic dataset inspection using:

```python
df.head()
df.isnull().sum()
df.info()
df.shape
```

---

## 🧹 Data Preprocessing

### 1. Separate Features and Target

The target column `Class` is removed from the input features:

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

Therefore:

```text
X → Input Features
y → Target Class
```

---

### 2. Check Classes

The unique class values are inspected using:

```python
df["Class"].unique()
```

---

### 3. Encode Target Labels

Since the target contains categorical class labels, `LabelEncoder` is used to convert them into numerical values.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
```

The encoded labels can then be used by PyTorch during classification training.

---

## ✂️ Train-Test Split

The dataset is divided into training and testing data using `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split Configuration

| Parameter | Value |
|---|---:|
| Test Size | 20% |
| Training Size | 80% |
| Random State | 42 |

The `random_state=42` ensures that the same split can be reproduced when the notebook is executed again.

---

## 📏 Feature Scaling

The input features are standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then used to transform both training and testing data.

Feature scaling is important for neural networks because it helps keep input features on comparable scales and can improve training stability.

---

## 🔥 PyTorch Implementation

The project uses **PyTorch** to build and train the Artificial Neural Network.

The following PyTorch modules are used:

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
```

---

## 🔢 Converting Data into Tensors

The scaled feature values are converted into PyTorch tensors.

```python
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)
```

### Tensor Data Types

| Data | Tensor Type |
|---|---|
| Input Features | `torch.float32` |
| Target Labels | `torch.long` |

The target uses `torch.long` because the model uses `CrossEntropyLoss` for classification.

---

## 📦 TensorDataset

The feature and target tensors are combined using `TensorDataset`.

```python
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)
```

This makes it easier to work with the data using PyTorch's data-loading utilities.

---

## 🚚 DataLoader

`DataLoader` is used to load the data in batches.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)
```

### DataLoader Configuration

| Parameter | Training | Testing |
|---|---:|---:|
| Batch Size | 32 | 32 |
| Shuffle | Yes | No |

Training data is shuffled to help prevent the model from learning patterns based on the order of the training samples.

---

# 🧠 Artificial Neural Network

The ANN is created using PyTorch's `nn.Module`.

```python
class ANN(nn.Module):

    def __init__(self):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        return self.model(x)
```

---

## 🏗️ Model Architecture

The architecture used in the notebook is:

```text
Input Features
      │
      ▼
┌─────────────────┐
│ Linear Layer    │
│ Input → 64      │
└────────┬────────┘
         │
         ▼
      ReLU
         │
         ▼
┌─────────────────┐
│ Linear Layer    │
│ 64 → 64         │
└────────┬────────┘
         │
         ▼
      ReLU
         │
         ▼
┌─────────────────┐
│ Output Layer    │
│ 64 → 7          │
└────────┬────────┘
         │
         ▼
   Class Scores
```

### Layer Summary

| Layer | Configuration |
|---|---|
| Input Layer | `X.shape[1] → 64` |
| Activation | ReLU |
| Hidden Layer | `64 → 64` |
| Activation | ReLU |
| Output Layer | `64 → 7` |

The final layer produces **7 output values**, corresponding to the seven classification outputs configured in the notebook.

---

## ⚡ Activation Function — ReLU

The model uses the **Rectified Linear Unit (ReLU)** activation function after each hidden layer.

```python
nn.ReLU()
```

ReLU introduces non-linearity into the neural network, allowing the model to learn more complex relationships in the data.

---

## 🎯 Loss Function

The project uses:

```python
criteria = nn.CrossEntropyLoss()
```

`CrossEntropyLoss` is commonly used for multi-class classification problems where the model outputs class scores and the target contains integer class labels.

---

## 🚀 Optimizer

The model uses the **Adam optimizer**:

```python
optimizer = optim.Adam(
    model.parameters()
)
```

The optimizer updates the weights and biases of the neural network during training.

---

## 🔥 Model Training

The model is trained for:

```python
epochs = 100
```

The training loop performs the following operations for every batch:

```text
Input Batch
    ↓
Forward Pass
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Update Parameters
    ↓
Next Batch
```

The main training process in the notebook is:

```python
for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for xb, yb in train_loader:

        optimizer.zero_grad()

        outputs = model(xb)

        loss = criteria(outputs, yb)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
```

The average training loss for each epoch is printed during training.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---:|
| Problem Type | Multi-Class Classification |
| Framework | PyTorch |
| Epochs | 100 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Hidden Layers | 2 |
| Hidden Neurons | 64, 64 |
| Activation | ReLU |
| Output Neurons | 7 |
| Train-Test Split | 80:20 |
| Random State | 42 |

---

## 📈 Model Evaluation

After training, the model is switched to evaluation mode:

```python
model.eval()
```

Gradient calculation is disabled during evaluation:

```python
with torch.no_grad():
```

The model then generates predictions for the test data.

The notebook calculates the number of correct predictions and total samples to obtain classification accuracy.

```text
Accuracy = Correct Predictions / Total Predictions × 100
```

The final accuracy is printed using:

```python
print("accuracy: ", correct / total * 100)
```

---

## 📊 Evaluation Metrics

The current notebook evaluates the model using:

### Accuracy

Accuracy measures the proportion of correctly classified samples.

```text
Accuracy =
Correct Predictions
------------------- × 100
Total Predictions
```

The notebook does not include additional evaluation metrics such as:

- Precision
- Recall
- F1-Score
- Confusion Matrix

These can be added in future improvements.

---

## 🔄 Complete Workflow

The complete project workflow can be summarized as:

```text
Date Fruit Dataset
        │
        ▼
Load CSV
        │
        ▼
Explore Dataset
        │
        ▼
Check Missing Values
        │
        ▼
Separate X and y
        │
        ▼
Encode Class Labels
        │
        ▼
Train-Test Split
        │
        ▼
StandardScaler
        │
        ▼
Convert to PyTorch Tensors
        │
        ▼
TensorDataset
        │
        ▼
DataLoader
        │
        ▼
Build ANN
        │
        ▼
CrossEntropyLoss
        │
        ▼
Adam Optimizer
        │
        ▼
Train for 100 Epochs
        │
        ▼
Evaluate on Test Data
        │
        ▼
Calculate Accuracy
```

---

## 📂 Project Structure

A suggested GitHub repository structure is:

```text
ANN-Classification/
│
├── ANN for Classification.ipynb
│
├── 10 DateFruit_Dataset.csv
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ANN-Classification.git
```

Replace `your-username` with your GitHub username.

---

### 2. Navigate to the Project

```bash
cd ANN-Classification
```

---

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn torch jupyter
```

---

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
ANN for Classification.ipynb
```

Make sure the following file is in the same directory as the notebook:

```text
10 DateFruit_Dataset.csv
```

Then execute the notebook cells sequentially.

---

## 📦 Requirements

The main dependencies used in the project are:

```text
Python
Pandas
NumPy
Scikit-learn
PyTorch
Jupyter Notebook
```

You can create a `requirements.txt` file containing:

```text
pandas
numpy
scikit-learn
torch
jupyter
```

Then install them using:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

The current project can be extended in several ways.

### 📊 Better Evaluation

Add:

- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report

For example:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        predictions
    )
)
```

---

### 📈 Training Visualization

Track training loss for every epoch and visualize the learning curve.

```text
Epoch
  │
  │\
  │ \
  │  \
  │   \
  │    \
  └──────────────
      Loss
```

This can help identify whether the model is learning properly or overfitting.

---

### 🧠 Model Improvements

Possible improvements include:

- Hyperparameter tuning
- Increasing or decreasing hidden-layer neurons
- Adding more hidden layers
- Trying different activation functions
- Adding Dropout
- Implementing Early Stopping
- Trying different learning rates
- Experimenting with different batch sizes

---

### 🔍 Model Comparison

The ANN can also be compared with traditional machine-learning classification algorithms such as:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors

This would provide a useful comparison between traditional ML and Deep Learning approaches.

---

## ⚠️ Note About the Current Evaluation Code

The notebook currently contains:

```python
predicted = torch.max(outputs, 1)

correct += (predicted == yb).sum().item()
```

`torch.max(outputs, 1)` returns both the maximum values and their indices. For obtaining the predicted class labels, the indices should be used.

A typical implementation is:

```python
_, predicted = torch.max(outputs, 1)

correct += (predicted == yb).sum().item()
```

This correction should be applied if the current evaluation code does not produce the expected accuracy.

---

## 🎓 Key Learning Outcomes

Through this project, the following concepts were implemented and practiced:

- Artificial Neural Networks
- Multi-class Classification
- PyTorch
- `nn.Module`
- `nn.Sequential`
- Linear Layers
- ReLU Activation
- Cross Entropy Loss
- Adam Optimizer
- Forward Propagation
- Backpropagation
- Train-Test Split
- Label Encoding
- Feature Scaling
- StandardScaler
- PyTorch Tensors
- TensorDataset
- DataLoader
- Mini-Batch Training
- Model Evaluation
- Classification Accuracy

---

## 💡 Key Takeaways

This project demonstrates the complete pipeline for implementing a classification neural network using PyTorch:

```text
Raw Data
   ↓
Data Preprocessing
   ↓
Label Encoding
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
PyTorch Tensors
   ↓
DataLoader
   ↓
Artificial Neural Network
   ↓
CrossEntropyLoss
   ↓
Adam Optimizer
   ↓
Training
   ↓
Testing
   ↓
Accuracy
```

The project provides a practical introduction to using **PyTorch for multi-class classification** and demonstrates how neural networks can be trained using batches of tensor data.

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

This project was developed as part of my learning journey in:

**Machine Learning • Deep Learning • Artificial Neural Networks • PyTorch**

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

