"""implements the WeightEditor class, which is used to edit model weights using the ROME method."""

import logging
import os
from typing import Dict, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from easyeditor import BaseEditor, ROMEHyperParams

from ..models import ModelName
from .utils import load_data

HPARAMS_PATH = "res/hparams/ROME/{}.yaml"
EDITED_DIR_PATH = "output/quant/{}_{}/"


class WeightEditor:
    """Class to edit model weights using the ROME method."""

    def __init__(
        self,
        model_name: ModelName,
        dataset: Dict,
        out_prefix: str,
        sample_size: Literal[100, 1000, 1500, None] = 100,
    ):
        self.model_name = str(model_name)
        self.sample_size = sample_size
        self.target_layer = None
        self.model_attr_template = None
        self.hparams = None
        self.original_weight = torch.from_numpy(
            np.load(str(f"output/{self.model_name}.npz"))["arr"]
        )
        if sample_size:
            self.out_dir = EDITED_DIR_PATH.format(self.model_name, str(sample_size))
            self.data = load_data(sample_size)
        else:
            self.out_dir = EDITED_DIR_PATH.format(self.model_name, out_prefix)
            self.data = dataset
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.logger = logging.getLogger(__name__)
        self._load_hparams()

    def edit_model(self):
        """Edit the model parameters for the target relation using the specified # of examples."""
        metrics = {
            "case_id": [],
            "pre_acc": [],
            "post_acc": [],
        }
        self.logger.info(f"Start editing {self.model_name}")
        for sample in self.data:
            case_id = sample["case_id"]
            subjects = [sample["requested_rewrite"]["subject"]]
            editor = BaseEditor.from_hparams(self.hparams)
            tokenizer = editor.tok
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"

            path = f"{self.out_dir}{case_id}.pt"
            if os.path.exists(path):
                self.logger.info(
                    f"Edited weights for case_id {case_id} already exist at {path}. Skipping..."
                )
                continue

            prompts = [
                sample["requested_rewrite"]["prompt"].format(
                    sample["requested_rewrite"]["subject"]
                )
            ]

            ground_truth = [sample["requested_rewrite"]["target_true"]["str"]]
            targent_new = [sample["requested_rewrite"]["target_new"]["str"]]

            assert ground_truth[0] != targent_new[0]

            metrics_sample, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=targent_new,
                subject=subjects,
                sequential_edit=True,
            )
            metrics["case_id"].append(case_id)
            metrics["pre_acc"].append(metrics_sample[0]["pre"]["rewrite_acc"][0])
            metrics["post_acc"].append(metrics_sample[0]["post"]["rewrite_acc"][0])

            self._save_target_weights(case_id, edited_model)
        self._save_metrics_csv(metrics)

    def _save_metrics_csv(self, metrics, path=""):
        df = pd.DataFrame(metrics)
        df.to_csv(
            os.path.join(self.out_dir, str(path + "metrics.csv")),
            index=False,
        )

    def _load_hparams(self):
        """Load hyperparameters and model specific configuration from a YAML file."""
        hparams_path = HPARAMS_PATH.format(self.model_name)
        hparams = ROMEHyperParams.from_hparams(hparams_path)
        with open(hparams_path, "r") as f:
            config = yaml.safe_load(f)
            self.target_layer = config["layers"][0]
            self.model_attr_template = config["rewrite_module_tmp"]
        self.hparams = hparams

    def _save_target_weights(self, case_id, edited_model):
        """Save the edited model's weights for the target layer to a file."""
        current_attr = edited_model
        for part in self.model_attr_template.split("."):
            if "{" in part:
                current_attr = current_attr[self.target_layer]
            else:
                current_attr = getattr(current_attr, part)
        W_hat = current_attr.weight.detach().cpu().to(torch.float32)
        W0 = self.original_weight.detach().to(torch.float32)
        delta_fp16 = (
            (W_hat - W0).to(torch.float16).cpu().contiguous()
        )  # save only quintified deltas
        # params = current_attr.weight.detach().cpu().numpy() # or save matrices
        path = f"{self.out_dir}{case_id}.pt"
        torch.save(
            {
                "delta_fp16": delta_fp16,
                "shape": torch.tensor(self.original_weight.shape, dtype=torch.int32),
            },
            path,
        )
        print(f"Saving edited weights to {path}")

    def check_editability(self, mask, rel):
        self.out_dir = "./"
        metrics = {
            "case_id": [],
            "pre_acc": [],
            "post_acc": [],
        }
        self.logger.info(f"Start editing {self.model_name}")
        editor = BaseEditor.from_hparams(self.hparams)
        tokenizer = editor.tok
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        layer = editor.model.transformer.h[self.target_layer].mlp.c_proj
        mask = mask.to(layer.weight.device)

        def masked_forward(self, input, *args, **kwargs):
            W = self.weight * mask
            return F.linear(input, W.T, self.bias)

        # bind method to this instance
        layer.forward = masked_forward.__get__(layer, layer.__class__)

        for sample in self.data:
            case_id = sample["case_id"]
            subjects = [sample["requested_rewrite"]["subject"]]
            prompts = [
                sample["requested_rewrite"]["prompt"].format(
                    sample["requested_rewrite"]["subject"]
                )
            ]

            ground_truth = [sample["requested_rewrite"]["target_true"]["str"]]
            targent_new = [sample["requested_rewrite"]["target_new"]["str"]]

            assert ground_truth[0] != targent_new[0]

            metrics_sample, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=targent_new,
                subject=subjects,
                sequential_edit=True,
            )
            metrics["case_id"].append(case_id)
            metrics["pre_acc"].append(metrics_sample[0]["pre"]["rewrite_acc"][0])
            metrics["post_acc"].append(metrics_sample[0]["post"]["rewrite_acc"][0])
            print(f"Done {len(metrics['case_id'])} / {len(self.data)}%")
        self._save_metrics_csv(metrics, f"{rel}_test_")
