import numpy as np

from .io import load_dat


def plot_all_dat_results():
    import matplotlib.pyplot as plt

    x_eproj, y_eproj = load_dat('eproj_abs_hist.dat')
    x_emax, y_emax = load_dat('emax_hist.dat')

    if x_eproj is not None and x_emax is not None:
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        width_eproj = (x_eproj[1] - x_eproj[0]) * 0.9 if len(x_eproj) > 1 else 0.1
        plt.bar(x_eproj, y_eproj, width=width_eproj, color='#3498DB', edgecolor='black', alpha=0.7)
        plt.title("Distribution of Raw Noise $|e_{proj}|$", fontsize=12)
        plt.xlabel("Absolute Projected Noise", fontsize=11)
        plt.ylabel("Probability Density", fontsize=11)

        plt.subplot(1, 2, 2)
        width_emax = (x_emax[1] - x_emax[0]) * 0.9 if len(x_emax) > 1 else 0.1
        plt.bar(x_emax, y_emax, width=width_emax, color='#8E44AD', edgecolor='black', alpha=0.7)
        plt.title("Distribution of Worst-case $E_{max}$", fontsize=12)
        plt.xlabel("Absolute Max Projected Noise", fontsize=11)
        plt.ylabel("Probability Density", fontsize=11)

        plt.tight_layout()
        plt.show()

    L = 3
    fig, axes = plt.subplots(1, L, figsize=(15, 4))

    files_found = False
    for i in range(1, L + 1):
        x_sig, y_sig = load_dat(f'sigma_max_W{i}.dat')
        if x_sig is not None:
            files_found = True
            width = (x_sig[1] - x_sig[0]) * 0.9 if len(x_sig) > 1 else 0.1
            axes[i - 1].bar(x_sig, y_sig, width=width, color='#2ECC71', edgecolor='black', alpha=0.7)
            axes[i - 1].set_title(fr"Spectral Norm $\sigma_{{max}}(W_{i})$", fontsize=12)
            axes[i - 1].set_xlabel(r"$\sigma_{max}$", fontsize=11)
            mean_sig = np.average(x_sig, weights=y_sig)
            axes[i - 1].axvline(mean_sig, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sig:.2f}')
            axes[i - 1].legend()

    if files_found:
        axes[0].set_ylabel("Probability Density", fontsize=11)
        plt.tight_layout()
        plt.show()
