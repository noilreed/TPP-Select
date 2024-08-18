from baseline.em import em_pred
from baseline.facloc import facloc_pred
from baseline.kmeans import kmeans_pred
from baseline.pca import pca_pred
from func.generate_synthetic import exo_by_num
import pickle
import numpy as np
from greedy.greedy import hybrid_greedy
from params import params_classification
from scipy.stats import sem
from statistics import mean

if __name__ == '__main__':
    filenames = ['Club', 'Election', 'Series', 'Verdict', 'BookOrder']
    n_iter = 50

    accus_ours = {}
    accus_facloc = {}
    accus_kmeans = {}
    accus_pca = {}
    accus_em = {}

    for accu in [accus_ours, accus_facloc, accus_kmeans, accus_pca, accus_em]:
        for filename in filenames:
            accu[filename] = []

    n_trial = 5

    for i in range(n_trial):
        for filename in filenames:
            param = params_classification[filename]
            penalty_time, penalty_mark, omega, v = param['penalty_time'], param['penalty_mark'], param['omega'], param[
                'v']
            skip_first = param['skip_first']
            stochastic_size = param['stochastic_size']

            print("-----------------------------------------------------------------")
            print(filename)

            with open('./data/%s.pickle' % filename, 'rb') as handle:
                timestamps, timestamp_dims, edge, mark = pickle.load(handle)
            fake_num = int(0.1 * len(timestamps))
            T = max(timestamps)
            n_dim = len(np.unique(timestamp_dims))
            sentiments = np.sort(np.unique(mark))
            exo_T = T
            fake = exo_by_num(0, exo_T, n_dim, fake_num)
            n_mark = len(np.unique(mark))
            fake_dim = [f[0] for f in fake]
            fake_time = [f[1] for f in fake]
            fake_mark = np.random.choice(sentiments, len(fake_time))
            timestamp_dims = np.concatenate((timestamp_dims, fake_dim))
            timestamps = np.concatenate((timestamps, fake_time))
            mark = np.concatenate((mark, fake_mark))
            sort_idx = np.argsort(timestamps)
            timestamp_dims, timestamps, mark = timestamp_dims[sort_idx], timestamps[sort_idx], mark[sort_idx]

            print("filename: %s" % filename)
            print("fake_num: %s" % fake_num)
            real_exo_idxs = np.where(np.isin(timestamps, fake_time) == True)[0]
            exo_mask = np.full_like(timestamps, False)
            exo_mask[real_exo_idxs] = True
            real_exo_idxs = sort_idx[len(timestamps) - fake_num:]

            # Our method
            pred_exo_idxs_ours, incr_fns = hybrid_greedy(timestamps, timestamp_dims, mark, n_iter, omega, v, n_dim, T,
                                                         penalty_time, penalty_mark, edge, sentiments,
                                                         stochastic_size=stochastic_size, verbose=True,
                                                         return_all=True, skip_first=skip_first)

            ours_exo_mask = np.full_like(timestamps, False)
            ours_exo_mask[pred_exo_idxs_ours] = True
            _accu_ours = np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idxs_ours])
            accus_ours[filename] += [_accu_ours]
            print("ours: %s" % accus_ours[filename])

            # facloc
            pred_exo_idxs_facloc = facloc_pred(timestamps, timestamp_dims, mark, n_iter)
            _accu_facloc = np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idxs_facloc])
            accus_facloc[filename] += [_accu_facloc]
            print("facloc: %s" % accus_facloc[filename])

            # kmeans
            pred_exo_idx_kmeans = kmeans_pred(timestamps, timestamp_dims, mark, n_iter)
            _accu_kmeans = np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idx_kmeans])
            accus_kmeans[filename] += [_accu_kmeans]
            print("kmeans: %s" % accus_kmeans[filename])

            # PCA
            pred_exo_idx_pca = pca_pred(timestamps, timestamp_dims, mark, n_iter)
            _accu_pca = np.mean([(1 if _ in real_exo_idxs else 0) for _ in pred_exo_idx_pca])
            accus_pca[filename] += [_accu_pca]
            print("pca: %s" % accus_pca[filename])

            # EM
            prex_exo_idx_em = em_pred(timestamps, timestamp_dims, mark, n_iter, edge, omega, v, n_dim, sentiments, T)
            _accu_em = np.mean([(1 if _ in real_exo_idxs else 0) for _ in prex_exo_idx_em])
            accus_em[filename] += [_accu_em]
            print("em: %s" % accus_em[filename])

            print("-----------------------------------------------------------------")

    print(accus_ours)
    print(accus_facloc)
    print(accus_kmeans)
    print(accus_pca)
    print(accus_em)
    print("--------------------------------------------")

    for filename in filenames:
        for method, accus in zip(["ours", "facloc", "kmeans", "pca", "em"],
                                 [accus_ours, accus_facloc, accus_kmeans, accus_pca, accus_em]):
            print("accus_%s: %s,  mean: %s, std: %s" % (method, mean(accus[filename]), sem(accus[filename])))
            print(accus)
            print("--------------------------------------------")
