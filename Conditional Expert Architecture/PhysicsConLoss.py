import torch
import torch.nn as nn

class PhysicsConLoss(nn.Module):
    def __init__(self):
        super(PhysicsConLoss, self).__init__()

    def forward(self, physics_pre, physics_true):

        diff = physics_pre - physics_true
        physics_loss = torch.abs(diff) ** 2

        return physics_loss.mean()