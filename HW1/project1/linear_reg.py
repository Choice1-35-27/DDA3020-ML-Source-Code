import pandas as np
from numpy import ndarray

def estimate_w(X:ndarray, y:ndarray, lambda_reg:float) -> ndarray:
    """
    calculate the weights of linear regression
    """
    XTX = X.T @ X
    # # create normalized item lambda_reg * I, shape of lambda_I is equal to the inverse matrix of X.T @ X
    lambda_I = lambda_reg * np.identity(XTX.shape[0])
    result = np.linalg.inv(XTX + lambda_I) @ X.T @ y
    return result

def simple_fit_linear_reg(x:ndarray, y:ndarray, n:int=3, lambda_reg:float=0.0) -> ndarray:
    """
    given dimension n and lambda coefficient, return the estimated y
    """
    X = np.column_stack((x**i for i in range(n+1)))
    w = estimate_w(X, y, lambda_reg)
    print(f"When n={n}, lambda=0 , w =", w)
    y_estimated = X @ w
    return y_estimated
