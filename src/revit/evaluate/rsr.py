import logging
from typing import Literal

import numpy as np
import torch

from ..models import GPT2XLModel, LLamaModel, ModelName
from ..utils.utils import load_data


class ReverseEfficiacyEvaluator:
    """
    A class to evaluate the reverse efficacy of knowledge editing in a transformer model."""

    def __init__(
        self,
        pruned_weights_path: str,
        top_k: int,
        sample_size: int = 100,
        model_name=ModelName.GPT2XL,
    ):
        self.pruned_w_path = pruned_weights_path
        self.edited_dir_path = f"output/edited_weights/{str(model_name)}_100/"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_model(model_name)
        self.top_k = top_k
        self.sample_size = sample_size
        self.logger = logging.getLogger(__name__)

    def set_model(self, model_name: ModelName):
        if model_name == ModelName.GPT2XL:
            self.model = GPT2XLModel()
        elif model_name == ModelName.LLAMA3_3B:
            self.model = LLamaModel()

    def evaluate(
        self,
        mode: Literal["edited", "pruned", "original"] = "pruned",
    ):
        """
        Evaluate the model on a set of examples.
        It saves the results with probability change for the new target and the true target.
        :param mode: The mode of evaluation, either "edited" or "pruned".
        """
        self.logger.info(f"\n=== {self.model.model_name} Evaluation ===\n")
        results = {
            "case_id": [],
            "pred": [],
            "obj_new": [],
            "p_new": [],
            "obj_true": [],
            "p_true": [],
            "rsr": [],
            "rsm": [],
            "top_k": [],
        }
        records = load_data(self.sample_size)
        for record in records:
            req = record["requested_rewrite"]
            prompt = req["prompt"].format(req["subject"])

            new_obj = req["target_new"]["str"]
            true_obj = req["target_true"]["str"]
            if not self._load_weights(
                record["case_id"], mode
            ):  # for workin onb subset of test set
                continue

            eval_result = self.model.evaluate_rome_rsr(
                prompt, true_object=true_obj, new_object=new_obj
            )
            results["case_id"].append(record["case_id"])
            results["pred"].append(eval_result["prediction"])
            results["obj_new"].append(new_obj)
            results["p_new"].append(eval_result["new_object_prob"])
            results["obj_true"].append(true_obj)
            results["p_true"].append(eval_result["true_object_prob"])
            results["rsr"].append(eval_result["rsr"])
            results["rsm"].append(eval_result["rsm"])

            results["top_k"].append(self.top_k)
            self.logger.info(
                f"Evaluated case {record['case_id']}: top_k={self.top_k}, rsr={eval_result['rsr']}, rsm={eval_result['rsm']}"
            )
        return results

    def _load_weights(
        self, case_id: str, mode: Literal["edited", "pruned", "non_target"]
    ):
        """
        Load the weights from a file and apply them to the model."""
        try:
            if mode == "edited":
                self.load_edited_weights(case_id)
            elif mode == "pruned":
                self.load_pruned_weights(case_id)
            elif mode == "original":
                pass
            else:
                raise NotImplementedError("Evaluation mode is not implemented yet.")
            return True
        except Exception as e:
            print(f"Warning: Failed to load or apply weights: {e}")
            return False

    def load_edited_weights(self, case_id, target_layer_num: int = 17):
        file_path = self.edited_dir_path + f"{case_id}.npz"
        npz_data = np.load(file_path)
        edited_weight = npz_data["arr"]
        edited_weight_tensor = torch.tensor(
            edited_weight, dtype=torch.float32, device=self.device
        )
        self.model.set_weight(edited_weight_tensor, target_layer_num=target_layer_num)

    def load_pruned_weights(self, case_id):
        file_path = self.pruned_w_path + f"{case_id}_pruned.npz"
        self.logger.info(f"Loading pruned weights from {file_path}")
        npz_data = np.load(file_path)
        pruned_weight = npz_data["arr"]
        pruned_weight_tensor = torch.tensor(
            pruned_weight, dtype=torch.float32, device=self.device
        )
        self.model.set_weight(pruned_weight_tensor)
