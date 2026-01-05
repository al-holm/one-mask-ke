"""Pruning weights in a transformer model based on different criteria."""

import logging
import os
from enum import Enum

import numpy as np
import torch

from ..models import GPT2XLModel, LLamaModel, ModelName

ORIGINAL_WEIGHT_PATH = "output/{}.npz"


class FillMode(Enum):
    """How to fill the positions that were zeroed‑out by the mask."""

    ORIGINAL = "original"
    AVG = "average"
    ZERO = "zero"
    SOFT = "soft"

    def __str__(self):
        return self.value


class Pruner:
    """
    Base class for pruning weights in a transformer model based on different criteria.
    """

    def __init__(
        self,
        weight_path: str,
        pruned_weight_path: str,
        num_examples: int = 1,
        top_k: float = 0.05,
        fill_mode: FillMode = FillMode.ZERO,
        model_name=ModelName.GPT2XL,
    ):
        self.w_path = weight_path
        self.pruned_w_path = pruned_weight_path
        os.makedirs(self.pruned_w_path, exist_ok=True)
        self.num_examples = num_examples
        self.top_k = top_k
        self.fill_mode = fill_mode
        self._weight = None
        self._mask = None
        self.logger = logging.getLogger(__name__)
        original_w_path = ORIGINAL_WEIGHT_PATH.format(model_name)
        self.original_weight = torch.tensor(np.load(original_w_path)["arr"]).float()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_weight = self.original_weight.to(self.device)

    def set_model(self, model_name: ModelName):
        if model_name == ModelName.GPT2XL:
            self.model = GPT2XLModel()
        elif model_name == ModelName.LLAMA3_3B:
            self.model = LLamaModel()

    def prune(self):
        """
        Prune weights based on a given criterion function and a top_k value.
        """
        w_file_list = os.listdir(self.w_path)
        self.logger.info(f"Found {len(w_file_list)} weight files in {self.w_path}.")
        for f in w_file_list[: self.num_examples]:
            if f.endswith(".csv"):
                continue
            else:
                case_id = f.split(".")[0]
                self.load_weights(case_id=case_id, path=self.w_path)
                pruned_weights = self._fill_pruned(self.criterion_func(case_id))
                self.logger.info(
                    f"Kept {self._mask.sum()} weights out of {self._mask.numel()}.\n\n"
                )
                self.save_weights(
                    case_id=case_id,
                    pruned_weights=pruned_weights,
                    path=self.pruned_w_path,
                )

    def _fill_pruned(self, pruned_weights: torch.Tensor) -> torch.Tensor:
        """Apply ``mask`` and replace pruned positions according to ``fill_mode``."""
        if self.fill_mode is FillMode.ZERO:
            pruned = pruned_weights
        elif self.fill_mode is FillMode.ORIGINAL:
            pruned = torch.where(
                self._mask.bool(), pruned_weights, self.original_weight
            )
        else:
            raise ValueError(f"Unsupported fill_mode: {self.fill_mode}")
        return pruned

    def save_weights(self, case_id: str, pruned_weights: torch.Tensor, path: str):
        """
        Save the pruned weights to a file.
        :param case_id: The identifier for the weights file.
        :param pruned_weights: The pruned weights tensor.
        """
        save_path = os.path.join(path, f"{case_id}_pruned.npz")
        np.savez(save_path, arr=pruned_weights.cpu().numpy())

    def load_weights(self, case_id: str, path: str):
        """
        Load weights from the specified path.
        """
        w = np.load(path + case_id + ".npz")["arr"]
        self._weight = torch.tensor(w).float()
        self._weight = self._weight.to(self.device)
