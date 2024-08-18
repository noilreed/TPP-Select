import sys
sys.path.append(".")

import numpy as np
from greedy.greedy import hybrid_greedy
from func.generate_synthetic import gen_synthetic_with_mark, process_history
import matplotlib
import matplotlib.pyplot as plt


def latexify():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath,amsfonts}"]
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{bm}"]
    matplotlib.rcParams['lines.markersize'] = 10
    plt.rc('axes', linewidth=1)


def plot_predict(name, selected_exo_idx, timestamps, endo_mask, mark):
    latexify()
    for exo_num in np.linspace(0, selected_exo_idx.size, 4):
        percentage = int((exo_num * 100) / selected_exo_idx.size)
        exo_num = int(exo_num)
        selected_exo = selected_exo_idx[:exo_num]
        selected_endo_mask = np.full_like(endo_mask, True)
        selected_endo_mask[selected_exo] = False
        fig, ax = plt.subplots(figsize=(6, 3.3), constrained_layout=True)
        mark_levels = [0.5 if m == 1 else 1 for m in mark]
        plot_exo_mask, selected_exo_mask = np.invert(endo_mask), np.invert(selected_endo_mask)
        green_idx = np.logical_and(plot_exo_mask, selected_exo_mask)
        red_idx = np.logical_and(endo_mask, selected_exo_mask)
        blue_idx = np.invert(np.logical_or(green_idx, red_idx))
        plot_levels = np.array(
            [mark_levels[i] if endo_mask[i] == False else -mark_levels[i] for i, timestamps in
             enumerate(timestamps)])
        orange_idx = np.logical_and(blue_idx, np.abs(plot_levels) == 0.5)
        blue_idx = np.logical_and(blue_idx, np.invert(orange_idx))
        blue_idx, orange_idx = orange_idx, blue_idx
        ax.vlines(timestamps[orange_idx], 0, plot_levels[orange_idx], color="tab:blue", zorder=0)
        ax.scatter(timestamps[orange_idx], plot_levels[orange_idx], color='tab:blue', zorder=1)
        ax.vlines(timestamps[blue_idx], 0, plot_levels[blue_idx], color="tab:blue", zorder=2)
        ax.scatter(timestamps[blue_idx], plot_levels[blue_idx], color='tab:blue', zorder=3)
        ax.vlines(timestamps[red_idx], 0, plot_levels[red_idx], color="tab:red", zorder=4)
        ax.scatter(timestamps[red_idx], plot_levels[red_idx], color='tab:red', zorder=5)
        ax.vlines(timestamps[green_idx], 0, plot_levels[green_idx], color="tab:green", zorder=6)
        ax.scatter(timestamps[green_idx], plot_levels[green_idx], color='tab:green', zorder=7)
        ax2 = ax.twinx()
        ax.get_yaxis().set_ticks([])
        ax2.get_xaxis().set_ticks([0, 25, 50, 75, 100])
        ax2.get_yaxis().set_ticks([])
        ax.axhline(linewidth=2, color='black')
        ax.set_xlabel(r'$t\rightarrow$', fontsize=35)
        ax.set_ylabel(r'$\mathcal{S}\hspace{0.5cm}$', fontsize=35, loc='top')
        ax2.set_ylabel(r'$\mathcal{H}_T\backslash \mathcal{S}$', fontsize=35, loc='bottom')
        plt.box(on=True)
        plt.grid(axis='y', linestyle='-', linewidth=1)
        plt.grid(axis='x', linestyle='-', linewidth=1)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        plt.savefig("./assets/synthetic_visulization/%s_%s.pdf" % (name, percentage), bbox_inches='tight', format='pdf')


if __name__=='__main__':
    dim, num_message = 30, 300
    omega, v = 1, 1
    num_sentiments = 2
    frac_exo = 0.3
    exo_size = int(num_message * frac_exo)
    T = 100
    sentiments = np.linspace(-1, 1, num_sentiments)
    penalty_time, penalty_mark = 0.1, 0.1

    edge, history, mark, beta_real = gen_synthetic_with_mark(dim, num_message, frac_exo, omega, v, T, num_sentiments, start_from_zero=False)
    timestamps, timestamp_dims, true_endo_mask = process_history(history)

    pred_exo_idxs = hybrid_greedy(timestamps, timestamp_dims, mark, exo_size, omega, v, dim, T, penalty_time, \
                                  penalty_mark, edge, sentiments, stochastic_size=None, verbose=True)

    plot_predict("hybrid", pred_exo_idxs, timestamps, true_endo_mask, mark)