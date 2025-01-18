from HW1.project1.metric import get_MSE
import numpy as np
from HW1.project1.linear_reg import estimate_w
from numpy import ndarray
import matplotlib.pyplot as plt

def grid_search(x:ndarray, y:ndarray, n_up_bnd:int=12, lambda_list:list=[]):
    """
    Implement a grid search by changing the polynomial degree $n$ as well as the regularization parameter λ.
    """
    n_values = range(0, n_up_bnd+1)
    mse_values = np.ndarray([])
    n_grid_values = np.ndarray([])
    lambda_grid_values = np.ndarray([])

    best_mse = np.inf
    for n_grid in n_values:
        for lambda_grid in lambda_list:
            # generate X matrix based on the degree
            X_grid = np.column_stack([x ** i for i in range(0, n_grid + 1)])
            # estimate w for current n and lambda
            w_grid = estimate_w(X_grid, y, lambda_grid)
            # calculate estimated y
            y_grid = X_grid @ w_grid
            mse = get_MSE(y, y_grid, 2)

            mse_values = np.append(mse_values, mse)
            n_grid_values = np.append(n_grid_values, n_grid)
            lambda_grid_values = np.append(lambda_grid_values, lambda_grid)

            if mse < best_mse:
                best_mse = mse
                best_n = n_grid
                best_lambda = lambda_grid
                best_w = w_grid

    print(f'(1) Solution of w = {best_w}, with shape = {np.shape(best_w)}')
    print(f'(2) Best MSE = {best_mse.round(2)}, with n = {best_n}, lambda = {best_lambda}')
    return best_mse, best_n, best_lambda, best_w

def display_grid(best_mse:ndarray, n_grid_values:ndarray,
                lambda_grid_values:ndarray, mse_values:ndarray):
    print('3D plots for grid research:')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(n_grid_values, lambda_grid_values, mse_values)
    best_point_index = mse_values.index(best_mse)
    ax.scatter(n_grid_values[best_point_index], lambda_grid_values[best_point_index], mse_values[best_point_index], c='r', marker='o', s=100, label='Best MSE')
    ax.set_xlabel('x-axis: n')
    ax.set_ylabel('y-axis: lambda')
    ax.set_zlabel('z-axis: MSE')
    plt.legend()
    plt.show()