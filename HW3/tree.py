import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import numpy as np

def set_global_random_seed(seed:int=42):
    np.random.seed(seed)

seed = set_global_random_seed(42)

def load():
    data = pd.read_csv('diabetes.csv')
    data = data.astype(int)
    print((data==0).sum())
    return data

def plot_dist_histograms(data):
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))

    axes[0, 0].hist(data['Glucose'])
    axes[0, 0].set_title('Glucose Distribution')
    axes[0, 0].set_xlabel('Glucose')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].hist(data['BloodPressure'])
    axes[0, 1].set_title('Blood Pressure Distribution')
    axes[0, 1].set_xlabel('Blood Pressure')
    axes[0, 1].set_ylabel('Count')

    axes[1, 0].hist(data['SkinThickness'])
    axes[1, 0].set_title('Skin Thickness Distribution')
    axes[1, 0].set_xlabel('Skin Thickness')
    axes[1, 0].set_ylabel('Count')

    axes[1, 1].hist(data['BMI'])
    axes[1, 1].set_title('BMI Distribution')
    axes[1, 1].set_xlabel('BMI')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

def feature_engineering(data):
    data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())
    data = data[(data[['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']] != 0).all(axis=1)]
    return data

def split_data(data):
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=seed)
    print('training set shape: ', X_train.shape)
    print('training set shape: ', y_train.shape)
    print('testing set shape: ', X_test.shape)
    print('training set shape: ', y_train.shape)
    return X_train, X_test, y_train, y_test

def decision_tree(X_train, X_test, y_train, y_test):
    classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=8, random_state=114514)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred

def get_auc_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def get_tree(features):
    plt.figure(figsize=(25, 20))
    classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=8, random_state=seed)
    plot_tree(classifier, feature_names=features.columns, class_names=['No', 'Yes'], filled=True)

def ensemble(X_train, X_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(max_features=50, n_estimators=100,random_state=seed)
    rf_classifier.fit(X_train, y_train)
    y_pred_ens = rf_classifier.predict(X_test)
    print(classification_report(y_test, y_pred_ens))
    return y_pred_ens

def main():
    data = load()
    plot_dist_histograms(data)
    data = feature_engineering(data)
    plot_dist_histograms(data)
    X_train, X_test, y_train, y_test = split_data(data)
    y_pred = decision_tree(X_train, X_test, y_train, y_test)
    get_auc_curve(y_test, y_pred)
    get_tree(X_train)
    y_pred_ens = ensemble(X_train, X_test, y_train, y_test)
    get_auc_curve(y_test, y_pred_ens)