import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
        cat_dim (int): Dimension along which to concatenate input tensors.
    """
    def __init__(self, num_channels, cat_dim=1):
        super(InvConv, self).__init__()
        if cat_dim == 1:
            num_channels *= 2
        self.num_channels = num_channels
        self.cat_dim = cat_dim

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
        x = torch.cat(x, dim=self.cat_dim)

        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        x = F.conv2d(x, weight)
        x = x.chunk(2, dim=self.cat_dim)

        return x, sldj
