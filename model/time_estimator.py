import autograd.numpy as np
from scipy.optimize import minimize
from autograd import grad

epsilon = 1e-50

# calculate event-wise excition matrix
def calc_excition(timestamps, timestamp_dims, to_dim, omega, T):
    if to_dim is not None:
        to_timestamps = np.append(timestamps[timestamp_dims == to_dim], T)
    else:
        to_timestamps = np.append(timestamps, T)
    diff = np.tile(to_timestamps, (timestamps.shape[0], 1)) - timestamps[:, np.newaxis]
    mat_excition = np.exp(-omega * diff) * (diff > 0)
    return mat_excition


def hawkes_likelihood_exo(alpha_u, endo_num, mat_excition, timestamp_dims, omega, sign, rho, verbose=False):
    mat_excition_alpha = alpha_u[timestamp_dims]
    term1 = np.sum(np.log(np.sum(mat_excition[:, :-1] * mat_excition_alpha[:, np.newaxis], axis=0) + epsilon))
    term2 = np.sum(mat_excition_alpha * (1 - mat_excition[:, -1])) / omega
    # regularizer
    regularizer = endo_num * rho * np.sum(alpha_u ** 2)
    if verbose:
        print('%5.3f, %5.3f' % (term1, term2))
    return sign * (term1 - term2 - regularizer)


# Optimize parameters of cascade with presence of exogenous events
def optimize_exo(endo_mask, timestamp_dims, to_dim, dim, omega, edge, mat_excition, rho):
    # global funs, mat_excitions
    endo_mask_u = np.logical_and(timestamp_dims == to_dim, endo_mask)
    endo_num = np.sum(endo_mask_u)
    try:
        neighbor_dims = np.arange(dim)[edge[:, to_dim] == 1]
    except Exception:
        print(1)
    neighbor_timestamp_idxs = np.isin(timestamp_dims, neighbor_dims)

    mat_excition = mat_excition[:, np.append(endo_mask_u, True)]
    mat_excition = mat_excition[neighbor_timestamp_idxs]

    timestamp_dims = timestamp_dims[neighbor_timestamp_idxs]

    result_alpha = np.zeros(dim)
    if mat_excition.shape[0] == 0:
        return result_alpha, -np.inf
    # bounds = [(1e-15,None) for _ in range(neighbor_dims.size)]
    bounds = [(0, None) for _ in range(neighbor_dims.size)]
    optimization_param_id = {}
    for neighbor_dim in neighbor_dims:
        optimization_param_id[neighbor_dim] = len(optimization_param_id)
    optimization_timestamp_dims = np.array([optimization_param_id[dim] for dim in timestamp_dims])

    alpha_u = np.random.uniform(0, 1, size=neighbor_dims.size)
    res = minimize(hawkes_likelihood_exo, alpha_u, args=(endo_num, mat_excition, optimization_timestamp_dims, omega, -1, rho, False),
            method="L-BFGS-B",
            jac=grad(hawkes_likelihood_exo),
            bounds=bounds,
            options={"ftol": 1e-10, "maxls": 50, "maxcor":50, "maxiter":100, "maxfun": 100})
    result_alpha[neighbor_dims] = res.x
    result_alpha[result_alpha < 0] = 0
    return result_alpha, -res.fun
