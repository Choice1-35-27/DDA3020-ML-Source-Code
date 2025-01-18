import numpy as np
from scipy.stats import multivariate_normal

def k_means(X, n_clusters, max_iter=100):
    cen_idx = np.random.choice(X.index, n_clusters, replace=False)
    center_id = X.loc[cen_idx].values

    for _ in range(max_iter):
        # calculate distance to center
        dist = np.linalg.norm(X.values[:, np.newaxis] - center_id, axis=-1)
        # assign cluster
        labels = np.argmin(dist, axis=1)
        # update center
        new_center_id = np.array([X.values[labels == i].mean(axis=0) for i in range(n_clusters)])
        # stop if no change
        if np.allclose(center_id, new_center_id):
            break
        center_id = new_center_id
    
    return labels

def gmm(X, n_clusters, max_iter=100):
    # Randomly initialize model parameters
    n_samples, n_features = X.shape
    means = X.sample(n_clusters, replace=False).values
    covariances = [np.eye(n_features) for _ in range(n_clusters)]
    weights = np.ones(n_clusters) / n_clusters

    for _ in range(max_iter):
        # E-step: Calculate posterior probabilities
        posterior = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            posterior[:, i] = weights[i] * multivariate_normal.pdf(X, means[i], covariances[i])
        posterior /= np.sum(posterior, axis=1, keepdims=True)

        # M-step: Update model parameters
        N = np.sum(posterior, axis=0)
        means = np.dot(posterior.T, X) / N[:, np.newaxis]
        for i in range(n_clusters):
            diff = X - means[i]
            covariances[i] = np.dot((diff.T * posterior[:, i]), diff) / N[i]
        weights = N / n_samples

    # Assign labels based on posterior probabilities
    labels = np.argmax(posterior, axis=1)

    return labels