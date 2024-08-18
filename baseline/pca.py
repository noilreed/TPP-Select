import numpy as np
from sklearn.decomposition import PCA


def deviation(x, eigenvalues, feature_vectors):
    d = 0.0
    for eigenvalue, feature_vector in zip(eigenvalues, feature_vectors):
        d += np.sum(np.inner(x, feature_vector) ** 2) / eigenvalue
    return d


def pca_pred(timestamp, timestamp_dims, mark, size):
    pca = PCA(n_components=1)
    X = [_ for _ in zip(timestamp, mark, timestamp_dims)]
    pca.fit(X)
    devs = [deviation(x, pca.explained_variance_, pca.components_) for x in X]
    # pred_exo_order = np.argsort(devs)[::-1]
    pred_exo_order = np.argsort(devs)
    return pred_exo_order[:size]
