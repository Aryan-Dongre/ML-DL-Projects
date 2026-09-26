# 🖼️ CNN Image Classification with PyTorch

A Deep Learning image classification project that implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images from the **CIFAR-10 dataset**.

The project covers the complete workflow of building a CNN, including dataset loading, image preprocessing, normalization, DataLoader creation, convolutional layers, pooling, flattening, fully connected layers, model training, and evaluation.

---

## 📌 Project Overview

This project uses a Convolutional Neural Network to classify images into **10 different categories** from the CIFAR-10 dataset.

The CNN learns visual features from images through multiple convolutional and pooling layers and then uses fully connected layers to perform the final classification.

The model is implemented using **PyTorch** and trained using:

- Convolutional Layers
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Cross Entropy Loss
- Adam Optimizer

---

## 🎯 Objective

The main objective of this project is to build a CNN capable of classifying CIFAR-10 images into their respective categories.

### Input

Each CIFAR-10 image has:

```text
32 × 32 × 3
```

where:

- `32` → Image Height
- `32` → Image Width
- `3` → RGB Color Channels

PyTorch represents the image as:

```text
3 × 32 × 32
```

---

## 📊 Dataset

The project uses the **CIFAR-10 dataset**, which is automatically downloaded using `torchvision`.

```python
from torchvision.datasets import CIFAR10
```

The dataset is loaded using:

```python
trainset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
```

---

## 🏷️ CIFAR-10 Classes

The CIFAR-10 dataset contains 10 image classes:

```text
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck
```

The CNN therefore contains **10 output neurons** in the final classification layer.

---

## 🧹 Image Preprocessing

Before feeding images into the CNN, the images are transformed using `torchvision.transforms`.

```python
transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

### 1. ToTensor

```python
transform.ToTensor()
```

This converts the image into a PyTorch tensor and scales pixel values into the range:

```text
[0, 1]
```

---

### 2. Normalization

The images are then normalized using:

```python
transform.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

This transforms the image values approximately into:

```text
[-1, 1]
```

The preprocessing pipeline can therefore be represented as:

```text
Original Image
      ↓
ToTensor()
      ↓
Pixel Values [0, 1]
      ↓
Normalize()
      ↓
Values approximately [-1, 1]
      ↓
CNN
```

---

## 📦 DataLoader

PyTorch `DataLoader` is used to load images in batches.

```python
trainloader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

testloader = DataLoader(
    testset,
    batch_size=64
)
```

### DataLoader Configuration

| Parameter | Training | Testing |
|---|---:|---:|
| Batch Size | 64 | 64 |
| Shuffle | Yes | No |

Training data is shuffled to prevent the model from depending on the original order of the dataset.

---

# 🧠 Convolutional Neural Network

The CNN is built using PyTorch's `nn.Module`.

The architecture contains three convolutional blocks followed by fully connected layers.

---

## 🏗️ CNN Architecture

The overall architecture is:

```text
Input Image
3 × 32 × 32
      │
      ▼
Conv2D
3 → 32
      │
      ▼
ReLU
      │
      ▼
MaxPool
      │
      ▼
32 × 16 × 16
      │
      ▼
Conv2D
32 → 64
      │
      ▼
ReLU
      │
      ▼
MaxPool
      │
      ▼
64 × 8 × 8
      │
      ▼
Conv2D
64 → 128
      │
      ▼
ReLU
      │
      ▼
MaxPool
      │
      ▼
128 × 4 × 4
      │
      ▼
Flatten
      │
      ▼
2048 Features
      │
      ▼
Linear
2048 → 256
      │
      ▼
ReLU
      │
      ▼
Linear
256 → 10
      │
      ▼
Class Prediction
```

---

## 🔲 Convolutional Layers

The model contains three convolutional layers.

### First Convolution

```python
nn.Conv2d(
    3,
    32,
    kernel_size=3,
    padding=1
)
```

This converts:

```text
3 channels → 32 channels
```

The image size remains:

```text
32 × 32
```

before pooling.

After Max Pooling:

```text
32 × 32 → 16 × 16
```

---

### Second Convolution

```python
nn.Conv2d(
    32,
    64,
    kernel_size=3,
    padding=1
)
```

This converts:

```text
32 channels → 64 channels
```

After Max Pooling:

```text
16 × 16 → 8 × 8
```

---

### Third Convolution

```python
nn.Conv2d(
    64,
    128,
    kernel_size=3,
    padding=1
)
```

This converts:

```text
64 channels → 128 channels
```

After Max Pooling:

```text
8 × 8 → 4 × 4
```

---

## 🔻 Max Pooling

The model uses:

```python
nn.MaxPool2d(2, 2)
```

with:

```text
Kernel Size = 2 × 2
Stride = 2
```

Max pooling reduces the spatial dimensions of the feature maps.

The dimensionality changes approximately as follows:

```text
32 × 32
    ↓
16 × 16
    ↓
8 × 8
    ↓
4 × 4
```

At the same time, the number of feature channels increases:

```text
3 → 32 → 64 → 128
```

---

## 🔥 ReLU Activation

After every convolutional layer, the model uses:

```python
nn.ReLU()
```

ReLU introduces non-linearity and allows the CNN to learn complex visual patterns.

The convolutional block follows this pattern:

```text
Convolution
     ↓
ReLU
     ↓
Max Pooling
```

---

# 🔄 Flattening

After the third convolutional block, the output has the shape:

```text
128 × 4 × 4
```

The feature maps are flattened before being passed to the fully connected layers.

```python
x = x.view(x.size(0), -1)
```

The number of features becomes:

```text
128 × 4 × 4 = 2048
```

Therefore, the first fully connected layer receives:

```text
2048 input features
```

---

# 🧮 Fully Connected Layers

The CNN uses two fully connected layers.

### First Fully Connected Layer

```python
nn.Linear(4 * 4 * 128, 256)
```

This converts:

```text
2048 → 256
```

followed by:

```python
nn.ReLU()
```

---

### Output Layer

```python
nn.Linear(256, 10)
```

The final layer produces:

```text
10 output values
```

corresponding to the 10 CIFAR-10 classes.

---

## 🧩 Complete Model Code

The CNN architecture implemented in the notebook is:

```python
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(

            # 1st Layer
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 2nd Layer
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 3rd Layer
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(

            nn.Linear(
                4 * 4 * 128,
                256
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                10
            )
        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc_layers(x)

        return x
```

---

# ⚙️ Loss Function

The project uses:

```python
criterion = nn.CrossEntropyLoss()
```

`CrossEntropyLoss` is suitable for multi-class classification problems.

The model produces class scores for all 10 CIFAR-10 classes, and the loss function compares these scores with the actual class labels.

---

# 🚀 Optimizer

The model uses the **Adam optimizer**:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

### Learning Rate

```text
0.001
```

Adam updates the model's weights and biases during backpropagation.

---

# 🔥 Model Training

The CNN is trained for:

```python
epochs = 10
```

The training loop follows:

```text
Input Images
     ↓
Forward Propagation
     ↓
Model Output
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Update Weights
     ↓
Next Batch
```

The training code follows this structure:

```python
for epoch in range(epochs):

    epoch_training_loss = 0.0

    for images, labels in trainloader:

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(
            output,
            labels
        )

        loss.backward()

        optimizer.step()

        epoch_training_loss += loss.item()
```

The average training loss is printed after each epoch.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---:|
| Problem Type | Image Classification |
| Dataset | CIFAR-10 |
| Framework | PyTorch |
| Epochs | 10 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Convolutional Layers | 3 |
| Fully Connected Layers | 2 |
| Activation | ReLU |
| Output Classes | 10 |

---

# 📊 Model Evaluation

After training, the model is evaluated using the test dataset.

Gradients are disabled during evaluation using:

```python
with torch.no_grad():
```

The model predicts the class with the highest output score:

```python
_, predicted = torch.max(
    outputs,
    1
)
```

The number of correct predictions is then calculated:

```python
correct_labels += (
    (predicted == labels)
    .sum()
    .item()
)
```

Finally, accuracy is calculated using:

```python
accuracy = (
    correct_labels /
    total_labels
) * 100
```

---

## 📈 Evaluation Metric

The notebook evaluates the CNN using:

### Accuracy

```text
Accuracy =
Correct Predictions
------------------- × 100
Total Predictions
```

The final accuracy is printed after evaluating all test batches.

> The notebook calculates the accuracy during execution; the README intentionally does not hard-code an accuracy value because the provided notebook does not contain a stored final result.

---

# 🔄 Complete CNN Workflow

The complete project workflow can be summarized as:

```text
CIFAR-10 Dataset
       ↓
Load Dataset
       ↓
ToTensor()
       ↓
Normalize Images
       ↓
Create DataLoader
       ↓
Input Image
3 × 32 × 32
       ↓
Conv2D
       ↓
ReLU
       ↓
MaxPool
       ↓
Conv2D
       ↓
ReLU
       ↓
MaxPool
       ↓
Conv2D
       ↓
ReLU
       ↓
MaxPool
       ↓
Flatten
       ↓
Fully Connected Layer
       ↓
ReLU
       ↓
Output Layer
       ↓
10 Class Scores
       ↓
Prediction
       ↓
Accuracy
```

---

# 📐 Feature Map Transformation

The spatial dimensions change throughout the CNN as follows:

```text
Input
3 × 32 × 32

       ↓ Conv2D

32 × 32 × 32

       ↓ MaxPool

32 × 16 × 16

       ↓ Conv2D

64 × 16 × 16

       ↓ MaxPool

64 × 8 × 8

       ↓ Conv2D

128 × 8 × 8

       ↓ MaxPool

128 × 4 × 4

       ↓ Flatten

2048

       ↓ Fully Connected

256

       ↓ Output Layer

10
```

This shows how the CNN gradually:

- Increases the number of feature channels
- Reduces spatial dimensions
- Extracts increasingly complex visual features
- Converts extracted features into class predictions

---

# 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Jupyter Notebook**

### Main PyTorch Components

```python
torch
torch.nn
torch.optim
torchvision
torchvision.datasets
torchvision.transforms
torch.utils.data.DataLoader
```

---

# 📂 Project Structure

A suggested GitHub repository structure is:

```text
CNN-Image-Classification/
│
├── CNN.ipynb
│
├── data/
│   └── CIFAR-10 dataset
│
├── README.md
│
└── requirements.txt
```

The CIFAR-10 dataset is automatically downloaded into the `data` directory when the notebook is executed.

---

# 🚀 How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/CNN-Image-Classification.git
```

Replace `your-username` with your GitHub username.

---

## 2. Navigate to the Project

```bash
cd CNN-Image-Classification
```

---

## 3. Install Dependencies

Install the required packages:

```bash
pip install torch torchvision numpy jupyter
```

---

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
CNN.ipynb
```

Run the cells sequentially.

The CIFAR-10 dataset will automatically be downloaded because the notebook uses:

```python
download=True
```

---

# 📦 Requirements

The main dependencies are:

```text
Python
PyTorch
Torchvision
NumPy
Jupyter Notebook
```

A `requirements.txt` file can contain:

```text
torch
torchvision
numpy
jupyter
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# 🔮 Future Improvements

The current CNN can be improved in several ways.

### 🧠 Model Improvements

- Add Batch Normalization
- Add Dropout
- Increase the number of convolutional layers
- Experiment with different kernel sizes
- Experiment with different numbers of filters
- Add additional fully connected layers

### ⚡ Training Improvements

- Train for more epochs
- Experiment with different learning rates
- Use a learning-rate scheduler
- Use GPU acceleration
- Implement early stopping

### 🖼️ Data Augmentation

The model can be improved using image augmentation techniques such as:

```text
Random Crop
Random Horizontal Flip
Random Rotation
Color Jitter
```

This can help the model generalize better to unseen images.

### 📊 Better Evaluation

Additional metrics can be added:

- Confusion Matrix
- Precision
- Recall
- F1-Score
- Per-class Accuracy

### 🔍 Model Visualization

Future versions can visualize:

- CNN feature maps
- Convolution filters
- Training loss
- Validation loss
- Confusion matrix
- Correct and incorrect predictions

---

# 🎓 Key Learning Outcomes

Through this project, the following concepts were implemented and practiced:

- Convolutional Neural Networks
- Image Classification
- CIFAR-10 Dataset
- PyTorch
- Torchvision
- Convolutional Layers
- Kernel / Filter
- Padding
- Stride
- ReLU Activation
- Max Pooling
- Feature Maps
- Flattening
- Fully Connected Layers
- Cross Entropy Loss
- Adam Optimizer
- Forward Propagation
- Backpropagation
- Batch Training
- DataLoader
- Image Normalization
- Model Evaluation
- Classification Accuracy

---

# 💡 Key Takeaways

This project demonstrates the basic architecture and workflow of a CNN for image classification.

The model learns visual representations through convolutional layers:

```text
Image
  ↓
Convolution
  ↓
Feature Extraction
  ↓
Pooling
  ↓
Feature Compression
  ↓
More Convolution
  ↓
Flatten
  ↓
Fully Connected Layers
  ↓
Class Prediction
```

The project provides practical experience with building and training a CNN from scratch using **PyTorch and CIFAR-10**.

---

# 👨‍💻 Author

## Aryan Dongre

**B.Tech — Artificial Intelligence & Data Science**

### Interests

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- AI Engineering
- Backend Development

---

# ⭐ Acknowledgement

This project was developed as part of my learning journey in:

**Machine Learning • Deep Learning • Computer Vision • Convolutional Neural Networks • PyTorch**

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

# 📜 License

This project is intended for **educational and learning purposes**.
