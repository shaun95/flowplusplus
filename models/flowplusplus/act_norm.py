import torch
import torch.nn as nn

from util import mean_dim


class _BaseNorm(nn.Module):
    """Base class for ActNorm (Glow) and PixNorm (Flow++).

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.

    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_channels, height, width, return_ldj=True, cat_dim=1):
        super(_BaseNorm, self).__init__()
        if cat_dim == 1:
            num_channels *= 2

        self.register_buffer('is_initialized', torch.zeros(1))
        self.mean = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.inv_std = nn.Parameter(torch.zeros(1, num_channels, height, width))

        self.eps = 1e-6
        self.return_ldj = return_ldj
        self.cat_dim = cat_dim
        self.is_channelwise = (height == width == 1)

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            if self.is_channelwise:
                mean = mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
                var = mean_dim((x.clone() - mean) ** 2, dim=[0, 2, 3], keepdims=True)
                inv_std = -torch.log(var.sqrt() + self.eps)
            else:
                mean = torch.mean(x.clone(), dim=0, keepdim=True)
                var = torch.mean((x.clone() - mean) ** 2, dim=0, keepdim=True)
                inv_std = 1. / (var.sqrt() + self.eps)

            self.mean.data.copy_(mean.data)
            self.inv_std.data.copy_(inv_std.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x + self.mean
        else:
            return x - self.mean

    def _scale(self, x, sldj, reverse=False):
        raise NotImplementedError('Subclass of _BaseNorm must implement _scale')

    def forward(self, x, ldj=None, reverse=False):
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)

        if self.return_ldj:
            return x, ldj

        return x


class ActNorm(_BaseNorm):
    """Activation Normalization used in Glow

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    """
    def __init__(self, num_features, return_ldj=False, cat_dim=1):
        super(ActNorm, self).__init__(num_features, 1, 1, return_ldj, cat_dim)

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x * self.inv_std.mul(-1).exp()
        else:
            x = x * self.inv_std.exp()

        if sldj is not None:
            if reverse:
                sldj = sldj - self.inv_std.sum() * x.size(2) * x.size(3)
            else:
                sldj = sldj + self.inv_std.sum() * x.size(2) * x.size(3)

        return x, sldj


class PixNorm(_BaseNorm):
    """Pixel-wise Activation Normalization used in Flow++

    Normalizes every activation independently (note this differs from the variant
    used in in Glow, where they normalize each channel). The mean and stddev get
    initialized using the mean and stddev of the first mini-batch. After the
    initialization, `mean` and `inv_std` become trainable parameters.
    """
    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
        else:
            x = x * self.inv_std

        if sldj is not None:
            if reverse:
                sldj = sldj - self.inv_std.log().sum()
            else:
                sldj = sldj + self.inv_std.log().sum()

        return x, sldj
