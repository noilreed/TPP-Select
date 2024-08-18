import copy
import time

import numpy as np

from model import time_estimator as time_estimator, mark_estimator as mark_estimator


# Algorithm 1 that only considers time influence
def time_greedy(timestamps, timestamp_dims, iters, omega, n_dim, T, penalty_time, edge, stochastic_size=300,
                verbose=False):
    if stochastic_size is None or stochastic_size > timestamps.size:
        stochastic_size = timestamps.size
    cur_endo_mask = np.full(len(timestamps), True)
    size = len(timestamps)
    exo_idx = None
    mat_excition_time = time_estimator.calc_excition(timestamps, timestamp_dims, None, omega, T)

    fn = np.empty(n_dim)
    for u in range(n_dim):
        _, fn_u = time_estimator.optimize_exo(cur_endo_mask, timestamp_dims, u, n_dim, omega, edge,
                                              mat_excition_time, penalty_time)
        fn[u] = fn_u
    exo_idxs = []
    for i in range(iters):
        if verbose:
            print("Iteration %s started.." % i)
            start = time.time()
        if exo_idx is not None:
            u = timestamp_dims[exo_idx]
            _, fn_u = time_estimator.optimize_exo(cur_endo_mask, timestamp_dims, u, n_dim, omega, edge,
                                                  mat_excition_time, penalty_time)
            fn[u] = fn_u
        stochastic_mask = np.full(timestamps.size, False)
        stochastic_mask[np.random.choice(timestamps.size, stochastic_size, replace=False)] = True
        incr_fn = np.full(size, -np.inf)
        for j, q in enumerate(timestamps):
            if not cur_endo_mask[j] or not stochastic_mask[j]:
                continue
            to_dim = timestamp_dims[j]
            _cur_endo_mask = copy.copy(cur_endo_mask)
            _cur_endo_mask[j] = False
            _, _fn = time_estimator.optimize_exo(_cur_endo_mask, timestamp_dims, to_dim, n_dim, omega, edge,
                                                 mat_excition_time, penalty_time)
            incr_fn[j] = _fn - fn[to_dim]
        exo_idx = np.argmax(incr_fn)
        cur_endo_mask[exo_idx] = False
        exo_idxs += [exo_idx]
        if verbose:
            end = time.time()
            print("Iteration %s ended.. takes %.2f, selected exogenous message: %s" % (i, end - start, exo_idx))
    return np.array(exo_idxs)


# Algorithm 1 that only considers mark influence
def mark_greedy(timestamps, timestamp_dims, mark, iters, v, n_dim, penalty_mark, edge, sentiments, stochastic_size=300,
                verbose=False):
    if stochastic_size is None or stochastic_size > timestamps.size:
        stochastic_size = timestamps.size
    cur_endo_mask = np.full(len(timestamps), True)
    size = len(timestamps)
    exo_idx = None
    mat_excition_mark = mark_estimator.calc_excition(timestamps, timestamp_dims, None, mark, v)

    fn = np.empty(n_dim)
    for u in range(n_dim):
        _, fn_time_u = mark_estimator.optimize_exo(cur_endo_mask, timestamp_dims, mark, u, n_dim, edge,
                                                   mat_excition_mark, penalty_mark, sentiments)
        fn[u] = fn_time_u
    exo_idxs = []
    for i in range(iters):
        if verbose:
            print("Iteration %s started.." % i)
            start = time.time()
        if exo_idx is not None:
            u = timestamp_dims[exo_idx]
            _, fn_time_u = mark_estimator.optimize_exo(cur_endo_mask, timestamp_dims, mark, u, n_dim, edge,
                                                       mat_excition_mark, penalty_mark, sentiments)
            fn[u] = fn_time_u
        stochastic_mask = np.full(timestamps.size, False)
        stochastic_mask[np.random.choice(timestamps.size, stochastic_size, replace=False)] = True
        incr_fn = np.full(size, -np.inf)
        for j, q in enumerate(timestamps):
            if not cur_endo_mask[j] or not stochastic_mask[j]:
                continue
            to_dim = timestamp_dims[j]
            _cur_endo_mask = copy.copy(cur_endo_mask)
            _cur_endo_mask[j] = False
            _, _fn = mark_estimator.optimize_exo(_cur_endo_mask, timestamp_dims, mark, u, n_dim, edge,
                                                 mat_excition_mark, penalty_mark, sentiments)
            incr_fn[j] = _fn - fn[to_dim]
        exo_idx = np.argmax(incr_fn)
        cur_endo_mask[exo_idx] = False
        exo_idxs += [exo_idx]
        if verbose:
            end = time.time()
            print("Iteration %s ended.. takes %.2f, selected exogenous message: %s" % (i, end - start, exo_idx))
    return np.array(exo_idxs)


# Algorithm 1 that considers both time and mark influence
def hybrid_greedy(timestamps, timestamp_dims, mark, iters, omega, v, n_dim, T, penalty_time, penalty_mark, edge,
                  sentiments, stochastic_size=300, verbose=False, return_all=False, skip_first=False):
    not_first_occurence_dim = np.array([True if d in timestamp_dims[:i] else False for i, d in enumerate(timestamp_dims)])
    if verbose:
        print("Starting greedy.. size=%s, iters=%s" % (len(timestamps), iters))
    if stochastic_size is None:
        stochastic_size = timestamps.size
    cur_endo_mask = np.full(len(timestamps), True)
    size = len(timestamps)
    exo_idx = None
    mat_excition_time = time_estimator.calc_excition(timestamps, timestamp_dims, None, omega, T)
    mat_excition_mark = mark_estimator.calc_excition(timestamps, timestamp_dims, None, mark, v)

    fn = np.empty(n_dim)
    # for every user in network, calculate endogenous likelihood
    for u in range(n_dim):
        _, fn_time_u = time_estimator.optimize_exo(cur_endo_mask, timestamp_dims, u, n_dim, omega, edge,
                                                   mat_excition_time, penalty_time)
        _, fn_mark_u = mark_estimator.optimize_exo(cur_endo_mask, timestamp_dims, mark, u, n_dim, edge,
                                                   mat_excition_mark, penalty_mark, sentiments)
        fn[u] = fn_time_u + fn_mark_u
    exo_idxs = []
    if return_all:
        incr_fns = []

    for i in range(iters):
        if verbose:
            print("Iteration %s started.." % i)
            start = time.time()
        if exo_idx is not None:
            u = timestamp_dims[exo_idx]
            _, fn_time_u = time_estimator.optimize_exo(cur_endo_mask, timestamp_dims, u, n_dim, omega, edge,
                                                       mat_excition_time, penalty_time)
            _, fn_mark_u = mark_estimator.optimize_exo(cur_endo_mask, timestamp_dims, mark, u, n_dim, edge,
                                                       mat_excition_mark, penalty_mark, sentiments)
            fn[u] = fn_time_u + fn_mark_u

        incr_fn = np.full(size, -np.inf)
        stochastic_mask = np.full(timestamps.size, False)
        remaining = np.setdiff1d(list(range(timestamps.size)), exo_idxs)
        stochastic_ind = np.random.choice(remaining, min(stochastic_size, remaining.size), replace=False)
        stochastic_mask[stochastic_ind] = True

        for j, q in enumerate(timestamps):
            if not stochastic_mask[j]:
                continue
            to_dim = timestamp_dims[j]
            _cur_endo_mask = copy.copy(cur_endo_mask)
            _cur_endo_mask[j] = False
            _, _fn_endo_time = time_estimator.optimize_exo(_cur_endo_mask, timestamp_dims, to_dim, n_dim, omega, edge,
                                                           mat_excition_time, penalty_time)
            _, _fn_endo_mark = mark_estimator.optimize_exo(_cur_endo_mask, timestamp_dims, mark, to_dim, n_dim, edge,
                                                           mat_excition_mark, penalty_mark, sentiments)
            _fn = _fn_endo_time + _fn_endo_mark
            incr_fn[j] = _fn - fn[to_dim]
        if return_all:
            incr_fns += [incr_fn]
        if skip_first:
            exo_idx = np.argmax(incr_fn[not_first_occurence_dim])
        else:
            exo_idx = np.argmax(incr_fn)
        cur_endo_mask[exo_idx] = False
        if exo_idx in exo_idxs:
            print(123)
        exo_idxs += [exo_idx]
        if verbose:
            end = time.time()
            print("Iteration %s ended.. takes %.2f, selected exogenous message: %s" % (i, end - start, exo_idx))
    if return_all:
        return np.array(exo_idxs), incr_fns
    else:
        return np.array(exo_idxs)
