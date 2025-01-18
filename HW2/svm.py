from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

x_train = pd.read_csv('HW2\Data\x_train.csv')
x_test = pd.read_csv('HW2\Data\x_test.csv')
y_train = pd.read_csv('HW2\Data\y_train.csv')['target']
y_test = pd.read_csv('HW2\Data\y_test.csv')['target']

classifiers = {}
support_vectors_indices = {}
class_labels = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

for class_label in class_labels:
    # Create binary classification labels for the current class
    y_train_binary = (y_train == class_label).astype(int)
    y_test_binary = (y_test == class_label).astype(int)


def train(model):
    svm = model
    # train the model for each class
    svm.fit(x_train, y_train_binary)
    # get predictions on training and testing set
    train_predictions = svm.predict(x_train)
    test_predictions = svm.predict(x_test)
    # calculate training and testing errors
    train_error = 1 - accuracy_score(y_train_binary, train_predictions)
    test_error = 1 - accuracy_score(y_test_binary, test_predictions)
    # set w 
    if svm.kernel == 'linear':
        w = svm.coef_
    else:
        # we don't need to estimate weights for non-linear kernels
        w = np.nan

    # Save the classifier and support vector indices
    classifiers[class_labels[class_label]] = svm
    support_vectors_indices = svm.support_
    sorted_ind = sorted(support_vectors_indices)

    slack_variables = []

    for i in support_vectors_indices:
        yt_i = y_train_binary[i]
        slack_variable = max(0, 1 - yt_i * svm.decision_function(x_train)[i])
        slack_variables.append(slack_variable)
        
    slack = [round(num, 5) for num in slack_variables]
    
    return train_error, test_error, w, svm.intercept_, sorted_ind, slack

def output(model, slack_variable=False):
    train_error, test_error, w, b, sv, slack= train(model=model)
    print(f"{class_labels[class_label]} training error: {train_error.round(3)}, testing error: {test_error.round(3)},")
    print(f"w_of_{class_labels[class_label]}: {w}, b_of_{class_labels[class_label]}: {b.round(3)},")
    print(f"support_vector_indices_of_{class_labels[class_label]}: {sv},")
    if slack_variable:
        print(f"slack_variable_{class_labels[class_label]}: {slack}")
    print('\n')

def standard_svm(C=1e5):
    model = SVC(kernel='linear', C=C)
    return model

def poly_svm(n):
    model = SVC(kernel='poly', degree=n, decision_function_shape='ovr', C=1)
    return model

def rbf_svm():
    model = SVC(kernel='rbf', gamma=1, C=1)
    return model

def sigmoid_svm():
    model = SVC(kernel='sigmoid', gamma=1, C=1)
    return model

def run():
    print('Q2.2.1 Calculation using Standard SVM Model:')
    model1 = standard_svm()
    for class_label in class_labels:
        output(model=model1, slack_variable=False)

    model2 = standard_svm(C=.2)
    print('Q2.2.2 Calculate using SVM with Slack Variables (C = 0.2 x t, where t = 1, 2, ... , 5):')
    for t in range(1, 6):
        C2 = round(0.2 * t, 2)
        print(f'C: {C2},')
        for class_label in class_labels:
            # Create binary classification labels for the current class
            y_train_binary = (y_train == class_label).astype(int)
            y_test_binary = (y_test == class_label).astype(int)
            output(model=model2, slack_variable=True)
        print('-----------------------------------------')

    poly2 = poly_svm(2)
    poly3 = poly_svm(3)
    rbf = rbf_svm()
    sigmoid = sigmoid_svm()

    print('Q2.2.3 Calculate using SVM with Kernel Functions and Slack Variables:')

    kernel_list = [poly2, poly3, rbf, sigmoid]
    ouputlist = ['(a)2nd-order Polynomial Kernel:', '(b) 3rd-order Polynomial Kernel:',
                '(c) Radial Basis Function Kernel with σ = 1:',
                '(d) Sigmoidal Kernel with σ = 1:']

    for index, kernel_name in enumerate(kernel_list):
        print(ouputlist[index])
        for class_label in class_labels:
            # Create binary classification labels for the current class
            y_train_binary = (y_train == class_label).astype(int)
            y_test_binary = (y_test == class_label).astype(int)
            output(model=kernel_name, slack_variable=True)
        print('-----------------------------------------')