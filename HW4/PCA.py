import numpy as np

def pca():
    X = np.array([
    [1, 0, 2, -3, -2],
    [0, 1, -3, -2, -3],
    [1, 2, 1, 3, -2],
    [-1, 1, 2, 3, -1],
    [1, 0, 1, -1, 1],
    [2, 3, -1, 1, -2],
    [-2, 3, -3, 2, 3],
    [-2, -2, 2, 3, -2],
    [-2, -2, 1, -3, -3],
    [-3, 2, 0, -1, -2]
    ])

    # Step 1: Calculate the mean of each column
    mean = np.mean(X, axis=0)

    # Step 2: Subtract the mean from each column
    centered_X = X - mean

    # Step 3: Calculate the covariance matrix
    covariance_matrix = np.cov(centered_X.T)

    # Step 4: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    for i in range(len(eigenvalues)):
        print('特征值:', eigenvalues[i])
        print('特征向量:', eigenvectors[:, i])

    # Step 5: Sort eigenvalues in descending order and choose eigenvectors as principal components
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices]

    # Step 6: Normalize the principal components to unit length
    unit_length_principal_components = principal_components / np.linalg.norm(principal_components, axis=0)
    for i in range(len(eigenvalues)):
        print(f'the {i+1}th principal component: {unit_length_principal_components[i]}')
    # Calculate the projection of each data point on the chosen principal components
    projections = np.dot(centered_X, unit_length_principal_components[:, :2])

    print(projections)
