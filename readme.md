# DDA3020 Machine Learning Codes

## Introduction

In order to avoid ownership disputes, this repository has removed the original notebook files and replaced them with scripts containing the relevant content. It does not contain any code that can be directly executed to obtain results.

For example, the original code block could be:

```python
def gradient_descent(X, y):
    raise NotImplementedError("You need to fill in the blank here.")
```

The block in this project is reshaped to:

```python
def gradient_descent(X:ndarray, y:ndarray):
    m, n = X.shape
    self.initialize_parameters(n)

    for i in range(self.num_iterations):
        # Complete your implementation here.
        # Forward pass (sigmoid function provided)
        fx = self.sigmoid(np.dot(X, self.weights) + self.bias)
        # Compute cost
        cost = -1 / m * np.sum(y * np.log(fx) + (1 - y) * np.log(1 - fx))
        # Compute gradients
        gradient = 1 / m * np.dot((fx-y), X)  
        # Update weights through gradient descent
        self.weights -= self.learning_rate * gradient
        # Print the cost every 100 iterations
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
```

## Contents

- `HW1`: Polynomial fitting curve, hyperparameter tuning

- `HW2`: Gradient Descent, Support Vector Machine

- `HW3`: Neural Network, Decision Trees and Embedding

- `HW4`: PCA, GMM, and AUC

## Setup

```bash
torch==2.2.1+cu121
torchaudio==2.2.1+cu121
torchmetrics==1.2.0
torchsummary==1.5.1
torchvision==0.17.1+cu121
numpy==1.23.0
pandas-bokeh==0.5.5
pandas-datareader==0.10.0
pandas-market-calendars==4.1.4
pandas-stubs==2.2.1.240316
matplotlib==3.8.2
seaborn==0.13.1
sklearn==0.0.post5
```
