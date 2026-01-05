"""Pruning weights in a transformer model based on activation of the targeted layer."""

import logging

import torch

from ..models import ModelName
from ..utils.utils import EDITED_DIR_PATH, load_data
from .pruner import FillMode, Pruner

PRUNED_WEIGHTS_PATH = "output/pruned_weights/{}/avg_activation/{}/{}/"


class ActivationPruner(Pruner):
    """
    Pruning weights in a transformer model based on average GELU activation.
    """
    def __init__(
        self,
        num_examples: int = 1,
        top_k: float = 0.05,
        model_name: ModelName = ModelName.GPT2XL,
        fill_mode: FillMode = FillMode.ZERO,
    ):
        w_path = EDITED_DIR_PATH.format(model_name, str(num_examples))
        pruned_w_path = PRUNED_WEIGHTS_PATH.format(
            str(model_name), str(fill_mode), str(int(top_k * 100))
        )
        super().__init__(
            weight_path=w_path,
            pruned_weight_path=pruned_w_path,
            num_examples=num_examples,
            top_k=top_k,
            fill_mode=fill_mode,
            model_name=model_name,
        )
        self.logger = logging.getLogger(__name__)
        self.set_model(model_name)
        self.data = load_data(num_examples)
        self.criterion_func = self._prune_on_activation

    def _prune_on_activation(self, case_id: str):
        """
        Prune weights based on their activation.
        :param case_id: The identifier for the weights file.
        :return: A mask indicating which weights are pruned based on top_k.
        """
        self.model.set_weight(self._weight)
        prompt = self._build_prompt(case_id)
        activation = self.model.get_activation(prompt)
        avg_activation = activation.mean(dim=1).squeeze()
        activation_last_token = torch.Tensor(avg_activation)
        threshold = torch.quantile(avg_activation, 1 - self.top_k)
        col_mask = (activation_last_token < threshold).float()
        if col_mask.shape[0] == self._weight.shape[1]:
            # LLaMA: [out, in]
            self._mask = col_mask.unsqueeze(0).expand(self._weight.shape[0], -1)
        elif col_mask.shape[0] == self._weight.shape[0]:
            # GPT: [in, out]
            self._mask = col_mask.unsqueeze(1).expand(-1, self._weight.shape[1])
        else:
            raise ValueError(
                f"Incompatible shapes: activation {col_mask.shape}, weight {self._weight.shape}"
            )

        return self._weight * self._mask

    def _build_prompt(self, case_id: str):
        """
        Build the prompt for the given case_id.
        :param case_id: The identifier for the weights file.
        :return: The prompt string.
        """
        for sample in self.data:
            if str(sample["case_id"]) == str(case_id):
                subject = sample["requested_rewrite"]["subject"]
                return sample["requested_rewrite"]["prompt"].format(subject)
        self.logger.error(f"Case ID {case_id} not found in data.")
