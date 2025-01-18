from linear_regression import (run_linear_regression, run_gradient_descent, 
                               calculate_rmse)
from preprocessing import prepare_normalized_x_and_y
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

train_nor, y1_df_nor, y2_df_nor = prepare_normalized_x_and_y()

def pred_future_max_temp(train_nor, y1_df_nor, splits=10, size=0.2, iters=100, alpha=1e-1):
    run_linear_regression(train_nor, y1_df_nor, splits=splits, size=size, iters=iters, alpha=alpha)

def tunning_alpha(x_data, target, splits, size, iters, alpha, flag):
    seed_list = [78, 25, 36, 15, 58, 12, 73, 18, 48, 34]
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    train_rmse_list = []  
    test_rmse_list = []   

    for i in range(splits):
        X_train, X_test, y_train, y_test = train_test_split(x_data, target, test_size=size, random_state=seed_list[i])
        w, b = run_gradient_descent(X_train, y_train, iters, alpha, if_print=False)

        y_pred_train = X_train @ w + b
        y_pred = X_test @ w + b
        
        train_rmse = calculate_rmse(y_train, y_pred_train)
        test_rmse = calculate_rmse(y_test, y_pred)
        
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)

        row, col = divmod(i, 5)
        ax = axes[row, col]
        ax.plot(np.arange(len(y_test)), y_test, color='steelblue')
        ax.plot(np.arange(len(y_test)), y_pred, color='firebrick')
        ax.set_xlabel(f'Trial {i + 1}')
        ax.set_ylabel(f'RMSE {test_rmse:.2f}')
        ax.set_title(f'LR Trial {i + 1}')

    avg_train_rmse = sum(train_rmse_list) / splits
    avg_test_rmse = sum(test_rmse_list) / splits
    if flag == 'high':
        print(f'Under learning rate = {alpha}, prediciting next day max temperature')
    elif flag == 'low':
        print(f'Under learning rate = {alpha}, prediciting next day min temperature')
    print(f"Average Train RMSE = {avg_train_rmse:.2f}")
    print(f"Average Test RMSE = {avg_test_rmse:.2f}")

    plt.tight_layout()
    plt.show()

def tunning_iterstep(x_data, target, splits, size, iters, alpha, flag):
    seed_list = [78, 25, 36, 15, 58, 12, 73, 18, 48, 34]
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    train_rmse_list = []  
    test_rmse_list = []   

    for i in range(splits):
        X_train, X_test, y_train, y_test = train_test_split(x_data, target, test_size=size, random_state=seed_list[i])
        w, b = run_gradient_descent(X_train, y_train, iters, alpha, if_print=False)

        y_pred_train = X_train @ w + b
        y_pred = X_test @ w + b
        
        train_rmse = calculate_rmse(y_train, y_pred_train)
        test_rmse = calculate_rmse(y_test, y_pred)
        
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)

        row, col = divmod(i, 5)
        ax = axes[row, col]
        ax.plot(np.arange(len(y_test)), y_test, color='steelblue')
        ax.plot(np.arange(len(y_test)), y_pred, color='firebrick')
        ax.set_xlabel(f'Trial {i + 1}')
        ax.set_ylabel(f'RMSE {test_rmse:.2f}')
        ax.set_title(f'LR Trial {i + 1}')

    avg_train_rmse = sum(train_rmse_list) / splits
    avg_test_rmse = sum(test_rmse_list) / splits
    if flag == 'high':
        print(f'Under iteration steps = {iters}, prediciting next day max temperature')
    elif flag == 'low':
        print(f'Under iteration steps = {iters}, prediciting next day min temperature')
    print(f"Average Train RMSE = {avg_train_rmse:.2f}")
    print(f"Average Test RMSE = {avg_test_rmse:.2f}")

    plt.tight_layout()
    plt.show()

def tuning_pipeline():
    tunning_alpha(train_nor, y1_df_nor, splits=10, size=0.2, iters=100, alpha=1e-1, flag='high')
    tunning_alpha(train_nor, y1_df_nor, splits=10, size=0.2, iters=100, alpha=1e-2, flag='high')
    tunning_alpha(train_nor, y1_df_nor, splits=10, size=0.2, iters=100, alpha=1e-3, flag='high')
    tunning_alpha(train_nor, y2_df_nor, splits=10, size=0.2, iters=100, alpha=1e-1, flag='low')
    tunning_alpha(train_nor, y2_df_nor, splits=10, size=0.2, iters=100, alpha=1e-2, flag='low')
    tunning_alpha(train_nor, y2_df_nor, splits=10, size=0.2, iters=100, alpha=1e-3, flag='low')
    tunning_iterstep(train_nor, y1_df_nor, splits=10, size=0.2, iters=100, alpha=1e-1, flag='high')
    tunning_iterstep(train_nor, y1_df_nor, splits=10, size=0.2, iters=1000, alpha=1e-1, flag='high')
    tunning_iterstep(train_nor, y1_df_nor, splits=10, size=0.2, iters=10000, alpha=1e-1, flag='high')
    tunning_iterstep(train_nor, y2_df_nor, splits=10, size=0.2, iters=100, alpha=1e-1, flag='low')
    tunning_iterstep(train_nor, y2_df_nor, splits=10, size=0.2, iters=1000, alpha=1e-1, flag='low')
    tunning_iterstep(train_nor, y2_df_nor, splits=10, size=0.2, iters=10000, alpha=1e-1, flag='low')