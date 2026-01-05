"""Pruning weights in a transformer model based on magnitudes."""

from typing import Literal

import torch

from ..models import ModelName
from ..utils.utils import EDITED_DIR_PATH
from .pruner import FillMode, Pruner

PRUNED_WEIGHTS_PATH = "output/pruned_weights/{}/unstructured_magnitude/{}/{}/"


class UnstructuredMagnitudePruner(Pruner):
    """
    Unstructured pruning based on magnitude.
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
        Prune weights by zeroing out the top-k% largest-magnitude entries.
        :param case_id: The identifier for the weights file.
        :return: The pruned weight tensor.
        """
        abs_w = self._weight.abs()
        flat_abs_w = abs_w.view(-1)
        k = int(self.top_k * flat_abs_w.numel())

        topk_vals, _ = torch.topk(flat_abs_w, k, largest=True)
        threshold = topk_vals[-1]

        self._mask = (abs_w < threshold).float()
        return self._weight * self._mask
