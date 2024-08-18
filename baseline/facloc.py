from apricot.functions.facilityLocation import FacilityLocationSelection
import time, pickle
import numpy as np


def facloc_anomaly(timestamps, timestamp_dims, mark):
    facloc = FacilityLocationSelection(int(0.7 * len(timestamps)))
    X = np.array([_ for _ in zip(timestamps, mark, timestamp_dims)])
    result = facloc.fit_transform(X)
    endo_time = np.array([r[0] for r in result])
    endo_mask = np.array([True if t in endo_time else False for t in timestamps])
    return np.argsort(endo_mask)


def facloc_pred(timestamps, timestamp_dims, mark, size):
    facloc = FacilityLocationSelection(size)
    X = np.array([_ for _ in zip(timestamps, mark, timestamp_dims)])
    result = facloc.fit_transform(X)
    exo_time = np.array([r[0] for r in result])
    exo_mask = np.array([True if t in exo_time else False for t in timestamps])
    return np.where(exo_mask == True)[0]
