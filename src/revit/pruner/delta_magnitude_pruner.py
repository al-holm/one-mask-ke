from ..models import ModelName
from ..utils.utils import EDITED_DIR_PATH
from .pruner import FillMode, Pruner

PRUNED_WEIGHTS_PATH = "output/pruned_weights/{}/delta_magnitude/{}/{}/"


class DeltaMagnitudePruner(Pruner):
    """
    Pruning weights in a transformer model based on update magnitude.
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
        self.criterion_func = self._prune_on_magnitude

    def _prune_on_magnitude(self, case_id: str):
        """
        Prune weights based on the magnitude of the update from the original weights.
        :param  case_id: The identifier for the weights file.
        :return: A mask indicating which weights are pruned based on top_k.
        """
        delta_w = self._weight - self.original_weight
        flat_delta = delta_w.view(-1).abs()
        k = int(self.top_k * flat_delta.numel())
        topk_indices = flat_delta.topk(k, largest=True).indices
        pruned_weight = self._weight.clone().view(-1)
        pruned_weight[topk_indices] = 0.0
        self._mask = (pruned_weight != 0).float().view_as(self._weight)
        return pruned_weight.view_as(self._weight)
