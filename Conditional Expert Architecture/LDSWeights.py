import numpy as np
import torch
import torch.nn as nn

class LDSWeights(nn.Module):
    def __init__(self):
        super(LDSWeights, self).__init__()

    def forward(self, targets, bin_edges, smoothed_hist):
        targets_numpy = targets.cpu().numpy()
        targets_numpy = np.clip(targets_numpy, bin_edges[0] + 1e-6, bin_edges[-1] - 1e-6)
        bin_indices = np.digitize(targets_numpy, bins=bin_edges, right=True) - 1
        epsilon = 1e-6
        batch_weights = 1 / (smoothed_hist[bin_indices] + epsilon)
        batch_weights = batch_weights + 1
        batch_weights_tensor = torch.tensor(batch_weights, dtype=torch.float32).to("cuda")
        return batch_weights_tensor