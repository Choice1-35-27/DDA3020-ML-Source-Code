import copy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def gradient_descent(X, y, w_in, b_in, gradient_function, alpha, num_iters): 
    '''
    w_in: w*, the initial weights w
    b_in: b*, the initial intercept b
    alpha: learning rate
    num_iters: how many times for iteration no matter reach the tol or not 
    '''
    w = copy.deepcopy(w_in) 
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient
        dj_db,dj_dw = gradient_function(X, y, w, b)   
        # Update Parameters
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
    return w, b

def compute_gradient_matrix(X, y, w, b): 
    m,n = X.shape
    f_wb = X @ w + b              
    e = f_wb - y                
    dj_dw  = (1/m) * (X.T @ e)    
    dj_db  = (1/m) * np.sum(e)    
    return dj_db,dj_dw

def run_gradient_descent(X,y,iterations, alpha, if_print=False):
    m, n = X.shape
    # initialize
    initial_w = np.zeros(n)
    initial_b = 0.0
    # gradient descent
    w_out, b_out = gradient_descent(X ,y, initial_w, initial_b, compute_gradient_matrix, alpha, iterations)
    if if_print:
        print(f"w: \n {w_out}, \n b = {b_out:0.2f}")
    return(w_out, b_out)

def cost(X, y, w, b):
    cost = 0
    for i in range(X.shape[0]):                                
        f_wb_i = X[i] @ w + b       
        cost = cost + (f_wb_i - y[i]) ** 2              
    cost = cost / (2*X.shape[0])
    return(np.squeeze(cost))

def calculate_rmse(y, y_pred):
    # calculate RMSE
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def denormalize_predictions(predictions, min_value, max_value):
    return predictions * (max_value - min_value) + min_value

def run_linear_regression(x_data, target, splits, size, iters, alpha):
    seed_list = [78, 25, 36, 15, 58, 12, 73, 18, 48, 34]
    for i in range(splits):
        print('trail time {}/{}'.format(i + 1, splits))
        print(f'learning rate = {alpha}, iteration = {iters}, seed = {seed_list[i]}')
        print('-------------------------------------------------------')
        X_train, X_test, y_train, y_test = train_test_split(x_data, target, test_size = size, random_state=seed_list[i])
        w, b = run_gradient_descent(X_train, y_train, iters, alpha, if_print=True)

        y_pred_train = X_train @ w + b
        y_pred = X_test @ w + b

        # # denormalize y
        # y_train_de = denormalize_predictions(y_train, y_train.min(), y_train.max())
        # y_test_de = denormalize_predictions(y_test, y_test.min(), y_test.max())
        # y_pred_train_de = denormalize_predictions(y_pred_train, y_pred_train.min(), y_pred_train.max())
        # y_pred_de = denormalize_predictions(y_pred, y_pred.min(), y_pred.max())

        print('-------------------------------------------------------')
        print(f"Train RMSE = {calculate_rmse(y_train, y_pred_train)}")
        print(f"Test RMSE = {calculate_rmse(y_test, y_pred)}")
        # print(f"Denormalize Train RMSE = {calculate_rmse(y_train_de, y_pred_train_de)}")
        # print(f"Denormalize Test RMSE = {calculate_rmse(y_test_de, y_pred_de)}")
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(y_test)), y_test, label='Ground True Value', color='steelblue')
        plt.plot(np.arange(len(y_test)), y_pred, label='Predicted Value', color='firebrick')
        plt.xlabel('data on test set')
        plt.ylabel('RMSE for test')
        plt.title(f'Error Curve for  trial {i+1} / 10')
        plt.legend()
        plt.show()