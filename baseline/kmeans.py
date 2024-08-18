from sklearn.cluster import KMeans
import numpy as np


def Kmeans_anomaly(timestamps, timestamp_dims, mark):
    X = [_ for _ in zip(timestamps, mark, timestamp_dims)]
    kmeans = KMeans(n_clusters=2).fit(X)
    kmeans_y = kmeans.predict(X)
    return kmeans_y


def kmeans_pred(timestamps, timestamp_dims, mark, size):
    X = [_ for _ in zip(timestamps, mark, timestamp_dims)]
    kmeans = KMeans(n_clusters=2).fit(X)
    kmeans_y = kmeans.predict(X)
    c1 = np.where(kmeans_y == 1)[0]
    c2 = np.where(kmeans_y == 0)[0]
    c = c1 if c1.size < c2.size else c2
    if len(c) > size:
        print("len(c[:size]): %s" % len(c[:size]))
        return c[:size]
    else:
        np.random.shuffle(c2)
        print("len(np.concatenate([c, c2[:(len(timestamps) - c1.size)]])): %s" % len(np.concatenate([c, c2[:(len(timestamps) - c1.size)]])))
        return np.concatenate([c, c2[:(len(timestamps) - c1.size)]])
