import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def gradient_descent(self, X, y):
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