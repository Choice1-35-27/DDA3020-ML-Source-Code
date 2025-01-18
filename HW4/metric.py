import numpy as np

def silhouette_coef(X, labels):
    n_samples = len(X)
    sc = np.zeros(n_samples)

    for i in range(n_samples):
        # calculate distance to cluster
        samples_in_same = X.values[labels == labels[i]]
        a = np.mean(np.linalg.norm(X.values[i] - samples_in_same, axis=1))
        
        # calculate distance to other clusters
        samples_other = set(np.unique(labels)) - set([labels[i]])
        b_values = []
        for j in samples_other:
            if np.any(labels == j):
                b = np.mean(np.linalg.norm(X.values[i] - X.values[labels == j], axis=1))
                b_values.append(b)
        
        if len(b_values) > 0:
            b = np.min(b_values)
        else:
            b = 0
        
        sc[i] = (b - a) / max(a, b)
    
    return np.mean(sc)

def rand_index(labels_true, labels_pred):
    n_samples = len(labels_true)
    tp, tn = 0, 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if (labels_true[i] == labels_true[j]) and (labels_pred[i] == labels_pred[j]):
                tp += 1
            elif (labels_true[i] != labels_true[j]) and (labels_pred[i] != labels_pred[j]):
                tn += 1
    
    return (tp + tn) / (n_samples * (n_samples - 1) / 2)

def entropy(labels):
    unique_labels, count = np.unique(labels, return_counts=True)
    prob = count / len(labels)
    return -np.sum(prob * np.log2(prob))

# Mutual information
def mutual_info(labels_pred, labels_true):
    unique_labels_true, count_true = np.unique(labels_true, return_counts=True)
    unique_labels_pred, count_pred = np.unique(labels_pred, return_counts=True)

    # calculate joint probability
    joint_prob = np.zeros((len(unique_labels_true), len(unique_labels_pred)))
    for i, label_true in enumerate(unique_labels_true):
        for j, label_pred in enumerate(unique_labels_pred):
            joint_prob[i, j] = np.sum((labels_true == label_true) & (labels_pred == label_pred)) / len(labels_true)
    
    mi = 0
    for i, label_true in enumerate(unique_labels_true):
        for j, label_pred in enumerate(unique_labels_pred):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (count_true[i] * count_pred[j]))
    
    return mi

# NMI
def normalize_mutual_info(labels_pred, labels_true):
    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)
    mi = mutual_info(labels_pred, labels_true)
    
    if h_true == 0 or h_pred == 0:
        return np.nan
    else:
        nmi = mi / np.sqrt(h_true * h_pred)
    return nmi