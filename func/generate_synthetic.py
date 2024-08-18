from model import mark_simulator as mark_simulator, time_simulator as time_simulator
import numpy as np
import networkx as nx


def get_edge(dim, edge_frac):
    edge = np.zeros((dim, dim))
    while True:
        g = nx.generators.random_graphs.binomial_graph(dim, edge_frac, directed=True)
        if len(g.edges) == int(dim * dim * edge_frac):
            break
    for e in g.edges:
        edge[e[0]][e[1]] = 1
    return edge


def exo_by_num(start_t, end_t, dim, num_exo):
    mu = np.full(dim, num_exo / ((end_t - start_t) * dim))
    while True:
        exo = time_simulator.simu_poiss(mu, end_t - start_t)
        if len(exo) == num_exo:
            break
    exo = [(t[0], t[1]+start_t, t[2]) for t in exo]
    return exo


def gen_synthetic(dim, num_message, frac_exo, omega, T, edge_frac=0.1, start_from_zero=True):
    num_exo = int(frac_exo * num_message)
    num_endo = num_message - num_exo
    edge = get_edge(dim, edge_frac)
    # alpha = np.full((dim, dim), 20 / dim) * edge
    alpha = np.full((dim, dim), 100 / dim) * edge
    while True:
        history = time_simulator.multivariate_exo(np.zeros(dim), alpha, omega, T, numEvents=num_endo)
        if len(history) == num_endo:
            break
    if start_from_zero:
        start_t = 0
    else:
        start_t = history[0][1]
    end_t = history[-1][1]
    exo = exo_by_num(start_t, end_t, dim, num_exo)
    history = history + exo
    history.sort(key=lambda x: x[1])
    print('exo len: %s, total len %s' % (len(exo), len(history)))
    return edge, history


def gen_synthetic_with_mark(dim, num_message, frac_exo, omega, v, T, num_sentiments, start_from_zero=True):
    edge, history = gen_synthetic(dim, num_message, frac_exo, omega, T, edge_frac=0.8, start_from_zero=start_from_zero)
    endo_mask = np.array([False if h[2] == 1 else True for h in history])
    beta = np.full((dim, dim), 0.1 / dim) * edge
    mark = mark_simulator.simulate_exo_history(history, endo_mask, beta, dim, v, num_sentiments)
    history = [(t[0], t[1]-history[0][1], t[2]) for t in history]
    return edge, history, mark, beta


def process_history(history):
    timestamp_dims = np.array([h[0] for h in history])
    timestamps = np.array([h[1] for h in history])
    endo_mask = np.array([False if h[2] == 1 else True for h in history])
    return timestamps, timestamp_dims, endo_mask

