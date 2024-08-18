from itertools import product

from baseline.pca import pca_pred
from baseline.em import em_pred
from baseline.facloc import facloc_pred
from baseline.kmeans import kmeans_pred
from func.generate_synthetic import gen_synthetic_with_mark, process_history
import numpy as np
# from autodump import Autodumper
import time

from greedy.greedy import hybrid_greedy

if __name__ == '__main__':
    dim, num_message = 5, 310
    num_sentiments = 2
    frac_exo = [0.1, 0.3, 0.5, 0.7]
    T = 100
    omega, v, penalty_time, penalty_mark = 0.1, 0.1, 0.1, 0.1

    accus_ours, accus_facloc, accus_kmeans, accus_pca, accus_em = {}, {}, {}, {}, {}
    for _frac_exo in frac_exo:
        for accus in [accus_ours, accus_facloc, accus_kmeans, accus_pca, accus_em]:
            accus[_frac_exo] = []

    n_trials = 5
    times = {}
    for _frac_exo in frac_exo:
        times[_frac_exo] = []

    for _frac_exo in frac_exo:
        for i in range(n_trials):
            exo_size = int(num_message * _frac_exo)
            sentiments = np.linspace(-1, 1, num_sentiments)

            edge, history, mark, beta_real = gen_synthetic_with_mark(dim, num_message, _frac_exo, omega, v, T,
                                                                     num_sentiments, start_from_zero=False)
            timestamps, timestamp_dims, true_endo_mask = process_history(history)
            real_exo_idxs = np.where(true_endo_mask == False)[0]

            real_endo = np.full(len(timestamps), True)
            real_endo[real_exo_idxs] = False

            # # ours
            start = time.time()
            pred_exo_idxs_ours = hybrid_greedy(timestamps, timestamp_dims, mark, exo_size, omega, v, dim, T,
                                               penalty_time, penalty_mark, edge, sentiments,
                                               stochastic_size=None, verbose=True, skip_first=False)
            # pred_endo_ours = np.full(len(timestamps), True)
            # pred_endo_ours[pred_exo_idxs_ours] = False
            # accus_ours[_frac_exo] += [np.mean(real_endo == pred_endo_ours)]
            accus_ours[_frac_exo] += [np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idxs_ours])]
            print("ours:", accus_ours[_frac_exo][-1])
            times[_frac_exo] += [time.time() - start]

            # facloc
            pred_exo_idxs_facloc = facloc_pred(timestamps, timestamp_dims, mark, exo_size)
            # pred_endo_facloc = np.full(len(timestamps), True)
            # pred_endo_facloc[pred_exo_idxs_facloc] = False
            # accus_facloc[_frac_exo] += [np.mean(real_endo == pred_endo_facloc)]
            accus_facloc[_frac_exo] += [np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idxs_facloc])]
            print("facloc:", accus_facloc[_frac_exo][-1])

            # kmeans
            pred_exo_idx_kmeans = kmeans_pred(timestamps, timestamp_dims, mark, exo_size)
            # pred_endo_kmeans = np.full(len(timestamps), True)
            # pred_endo_kmeans[pred_exo_idx_kmeans] = False
            # accus_kmeans[_frac_exo] += [np.mean(real_endo == pred_endo_kmeans)]
            accus_kmeans[_frac_exo] += [np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idx_kmeans])]
            print("kmeans:", accus_kmeans[_frac_exo][-1])

            # PCA
            pred_exo_idx_pca = pca_pred(timestamps, timestamp_dims, mark, exo_size)
            # pred_endo_pca = np.full(len(timestamps), True)
            # pred_endo_pca[pred_exo_idx_pca] = False
            # accus_pca[_frac_exo] += [np.mean(real_endo == pred_endo_pca)]
            accus_pca[_frac_exo] += [np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idx_pca])]
            print("pca:", accus_pca[_frac_exo][-1])

            # EM
            pred_exo_idx_em = em_pred(timestamps, timestamp_dims, mark, exo_size, edge, omega, v, dim, sentiments, T)
            # pred_endo_em = np.full(len(timestamps), True)
            # pred_endo_em[pred_exo_idx_em] = False
            # accus_em[_frac_exo] += [np.mean(real_endo == pred_endo_em)]
            accus_em[_frac_exo] += [np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idx_em])]
            print("em:", accus_em[_frac_exo][-1])

    print("ours: %s" % accus_ours)
    print("pca: %s" % accus_pca)
    print("em: %s" % accus_em)
    print("kmeans: %s"% accus_kmeans)
    print("facloc: %s"% accus_facloc)
    print("time: %s"% time)
