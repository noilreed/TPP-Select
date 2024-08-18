import numpy as np

from model import time_estimator, mark_estimator


def improvements(timestamps, timestamp_dims, mark, edge, omega, v, n_dim, sentiments, T, stochastic_size):
    T = np.max(timestamps)
    mat_excition_time = time_estimator.calc_excition(timestamps, timestamp_dims, None, omega, T)
    mat_excition_mark = mark_estimator.calc_excition(timestamps, timestamp_dims, None, mark, v)
    score = np.full(timestamps.size, -np.inf)

    stochastic_mask = np.full(timestamps.size, False)
    stochastic_mask[np.random.choice(timestamps.size, stochastic_size, replace=False)] = True

    for i, t, d, m in zip(range(len(timestamps)), timestamps, timestamp_dims, mark):
        if not stochastic_mask[i]:
            continue
        cur_endo_mask = np.full(len(timestamps), True, dtype=np.bool)
        cur_endo_mask[i] = False
        _, fn_time_u = time_estimator.optimize_exo(cur_endo_mask, timestamp_dims, d, n_dim, omega, edge,
                                                   mat_excition_time, 0)
        _, fn_mark_u = mark_estimator.optimize_exo(cur_endo_mask, timestamp_dims, mark, d, n_dim, edge,
                                                   mat_excition_mark, 0, sentiments)
        score[i] = fn_time_u + fn_mark_u

    return score


def em_pred(timestamps, timestamp_dims, mark, size, edge, omega, v, n_dim, sentiments, T, stochastic_size=300):
    score_improvement = improvements(timestamps, timestamp_dims, mark, edge, omega, v, n_dim, sentiments, T, stochastic_size)
    max_improvement = np.argsort(score_improvement)[::-1]
    return max_improvement[:size]
