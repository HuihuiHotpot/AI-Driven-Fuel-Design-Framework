import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def LDS(labels, kernel_std, n_bins):
    min_label, max_label = labels.min(), labels.max()
    bins = np.linspace(min_label, max_label, n_bins + 1)
    hist, bin_edges = np.histogram(labels, bins=bins, density=True)

    smoothed_hist = gaussian_filter1d(hist, sigma=kernel_std)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]) * 0.9, alpha=0.6, color="skyblue",label="Original Histogram")
    axes[0].set_title("Original Histogram")
    axes[0].set_xlabel("Bins (Ranges)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    ax2 = axes[1]
    ax2.bar(bin_centers, smoothed_hist, width=(bin_edges[1] - bin_edges[0]) * 0.9, alpha=0.6, color="lightcoral", label='Smoothed Histogram')
    ax2.set_title("Smoothed Histogram with Weights")
    ax2.set_xlabel("Bins (Ranges)")
    ax2.set_ylabel("Smoothed Frequency", color='orange')
    ax2.legend(loc='upper left')

    ax3 = ax2.twinx()
    ax3.plot(bin_centers, 1 / (smoothed_hist + 1e-6), color='green', marker='o', label='Weights')
    ax3.set_ylabel("Weights", color='green')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return hist, bin_centers, bin_edges, smoothed_hist