import torch


class ElementwiseMask(torch.nn.Module):
    """
    Elementwise mask. If a weight_getter is provided, we ignore the incoming W
    """

    def __init__(self, mask_getter, weight_getter=None):
        super().__init__()
        self._mask_getter = mask_getter
        self._weight_getter = weight_getter
        self.eps = 1e-9

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        K = self._mask_getter().to(device=W.device, dtype=W.dtype)
        if self.eps > 0:
            K = K.clamp(self.eps, 1.0 - self.eps)
        if self._weight_getter:
            W = self._weight_getter().to(device=W.device, dtype=W.dtype)
        return W * K


class MaskedWeight(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, X):
        return X * self.mask
