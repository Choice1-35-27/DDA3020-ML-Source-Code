import numpy as np
import matplotlib.pyplot as plt

global x_real_g, y_real_g
global x, y

def f(x, best_n, best_w):
    '''
    best_n: obtain in (4) which offers the least MSE
    X_best: corresponding polynomial matrix X under best degree n
    '''
    X_best = np.column_stack([x ** j for j in range(0, best_n+1)])
    return X_best @ best_w

def load():
    import pickle as pkl
    with open('HW1\Data\ground truth function', 'rb') as read_file:
        x_real_g, y_real_g = pkl.load(read_file)
        read_file.close()
    return x_real_g, y_real_g

def display(x_left, x_right):
    '''
    x_left: int, the lower bound of range x
    x_right: int, the upper bound of range x
    '''
    # generate points for specific x range and corresponding y calculated by f
    x_test = np.arange(x_left, x_right, 0.1)
    y_test = f(x_test)

    plt.plot(x_real_g, y_real_g, color='C0', label='ground true function: g')
    plt.scatter(x, y, color='C1', label='sample points')
    plt.plot(x_test, y_test, color='C2', label='estimate function: f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()