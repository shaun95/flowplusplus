import torch
import torch.nn as nn

from util import mean_dim


class ActNorm(nn.Module):
    """Activation Normalization as described in Flow++

    When height = 1 and width = 1: Glow-style activation normalization.

    When height > 1 or width > 1:
    Normalizes every activation independently (note this differs from the variant
    used in in Glow, where they normalize each channel). The mean and stddev get
    initialized using the mean and stddev of the first mini-batch. After the
    initialization, `mean` and `inv_std` become trainable parameters.

    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, norm_shape, return_ldj=True, cat_dim=1):
        super(ActNorm, self).__init__()
        num_channels, height, width = norm_shape
        if cat_dim == 1:
            num_channels *= 2
        self.register_buffer('is_initialized', torch.zeros(1))
        self.mean = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.inv_std = nn.Parameter(torch.zeros(1, num_channels, height, width))

        self.eps = 1e-6
        self.return_ldj = return_ldj
        self.cat_dim = cat_dim
        self.per_channel = (height == 1 and width == 1)

    def init_params(self, x):
        if not self.training:
            return

        with torch.no_grad():
            if self.per_channel:
                mean = mean_dim(x.clone(), [0, 2, 3], keepdims=True)
                var = mean_dim((x.clone() - mean) ** 2, [0, 2, 3], keepdims=True)
            else:
                mean = torch.mean(x.clone(), dim=0, keepdim=True)
                var = torch.mean((x.clone() - mean) ** 2, dim=0, keepdim=True)
            inv_std = 1. / (var.sqrt() + self.eps)
            self.mean.data.copy_(mean.data)
            self.inv_std.data.copy_(inv_std.data)
            self.is_initialized += 1.

    def forward(self, x, sldj=None, reverse=False):
        x = torch.cat(x, dim=self.cat_dim)
        if not self.is_initialized:
            self.init_params(x)

        if reverse:
            x = x / self.inv_std + self.mean
            sldj = sldj - self.inv_std.log().sum()
        else:
            x = (x - self.mean) * self.inv_std
            sldj = sldj + self.inv_std.log().sum()

        x = x.chunk(2, dim=self.cat_dim)

        if self.return_ldj:
            return x, sldj

        return x
