"""Pruning weights in a transformer model based on magnitudes."""

from typing import Literal

import torch
from torch import linalg as LA

from ..models import ModelName
from ..utils.utils import EDITED_DIR_PATH
from .pruner import FillMode, Pruner

PRUNED_WEIGHTS_PATH = "output/pruned_weights/{}/structured_magnitude/{}/{}/"


class MagnitudePruner(Pruner):
    """
    Structured pruning based on magnitude.
    """

    def __init__(
        self,
        sample_size: Literal[100, 1000] = 100,
        top_k: float = 0.05,
        model_name: ModelName = ModelName.GPT2XL,
        fill_mode: FillMode = FillMode.ZERO,
    ):
        w_path = EDITED_DIR_PATH.format(model_name, str(sample_size))
        pruned_w_path = PRUNED_WEIGHTS_PATH.format(
            str(model_name), str(fill_mode), str(int(top_k * 100))
        )
        super().__init__(
            weight_path=w_path,
            pruned_weight_path=pruned_w_path,
            num_examples=sample_size,
            top_k=top_k,
            fill_mode=fill_mode,
            model_name=model_name,
        )
        self.criterion_func = self._prune_on_magnitude

    def _prune_on_magnitude(self, case_id: str):
        """
        Prune weights based on their magnitude.
        :param  case_id: The identifier for the weights file.
        :return: A mask indicating which weights are pruned based on top_k.
        """
        norms = LA.norm(self._weight, dim=0)
        threshold = torch.quantile(norms, 1 - self.top_k)
        self._mask = (norms < threshold).float()
        return self._weight * self._mask

    def _prune_on_magnitude_per_dim(self, case_id: str):
        """
        Prune weights based on their magnitude per dimension.
        :param  case_id: The identifier for the weights file.
        :return: A mask indicating which weights are pruned based on top_k.
        """
        top_idx = self._weight.sort(descending=True).indices[
            :, : int(self.top_k * self._weight.shape[-1])
        ]
        self._mask = torch.zeros_like(self._weight)
        self._mask[:, :] = 1
        self._mask.scatter_(-1, top_idx, 0)
        return self._weight * self._mask
