"""implements the MEMITBatchEditor class, which is used to edit model weights using the MEMIT method in batch."""

import logging
import os
from typing import Dict, Literal

import pandas as pd
import torch

from easyeditor import BaseEditor
from easyeditor.editors.utils import _prepare_requests
from easyeditor.evaluate import compute_edit_quality
from easyeditor.models.memit import MEMITHyperParams

from ..models import ModelName
from .utils import load_data

HPARAMS_PATH = "res/hparams/MEMIT/{}.yaml"
EDITED_DIR_PATH = "output/memit_w/{}_{}_MEMIT/"


class MEMITBatchEditor:
    """Class to edit model weights using the MEMIT method in batch."""

    def __init__(
        self,
        model_name: ModelName,
        dataset: Dict,
        out_prefix: str,
        sample_size: Literal[100, 1000, 1500, None] = 100,
    ):
        self.model_name = str(model_name)
        self.sample_size = sample_size
        self.hparams = None

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

    def _load_hparams(self):
        """Load hyperparameters and model specific configuration from a YAML file."""
        hparams_path = HPARAMS_PATH.format(self.model_name)
        self.hparams = MEMITHyperParams.from_hparams(hparams_path)

    def edit_model(self):
        """Edit the model parameters for the target relation using the specified # of examples in batch."""
        self.logger.info(f"Start batch editing {self.model_name} with MEMIT")

        prompts = []
        ground_truth = []
        target_new = []
        subjects = []

        for sample in self.data:
            prompts.append(
                sample["requested_rewrite"]["prompt"].format(
                    sample["requested_rewrite"]["subject"]
                )
            )
            ground_truth.append(sample["requested_rewrite"]["target_true"]["str"])
            target_new.append(sample["requested_rewrite"]["target_new"]["str"])
            subjects.append(sample["requested_rewrite"]["subject"])

        editor = BaseEditor.from_hparams(self.hparams)

        # Prepare requests manually
        requests = _prepare_requests(
            prompts=prompts,
            target_new=target_new,
            ground_truth=ground_truth,
            subject=subjects,
        )

        # Execute MEMIT batch edit
        # apply_algo returns (edited_model, weights_copy)
        edited_model, _ = editor.apply_algo(
            editor.model,
            editor.tok,
            requests,
            editor.hparams,
            copy=False,
            return_orig_weights=True,
        )

        self._save_batch_weights(edited_model)

        # Manually compute metrics for the edited model
        flat_metrics = {
            "case_id": [s["case_id"] for s in self.data],
            "pre_acc": [],
            "post_acc": [],
        }

        self.logger.info("Computing metrics for verified edited model...")
        for i, request in enumerate(requests):
            post_metric = compute_edit_quality(
                edited_model,
                self.model_name,
                self.hparams,
                editor.tok,
                request,
                self.hparams.device,
            )

            flat_metrics["pre_acc"].append(0.0)
            flat_metrics["post_acc"].append(post_metric["rewrite_acc"][0])

        self._save_metrics_csv(flat_metrics)

    def _save_batch_weights(self, edited_model):
        """Save the edited model's weight deltas for the target layers to a file."""
        weights = {}

        for layer in self.hparams.layers:
            module_name = self.hparams.rewrite_module_tmp.format(layer)
            current_attr = edited_model
            for part in module_name.split("."):
                current_attr = getattr(current_attr, part)
            weights[layer] = current_attr.weight.detach().cpu().to(torch.float32)

        path = os.path.join(self.out_dir, "memit_batch.pt")
        torch.save(weights, path)
        self.logger.info(f"Saved batch weights to {path}")

    def _save_metrics_csv(self, metrics, path=""):
        df = pd.DataFrame(metrics)
        df.to_csv(
            os.path.join(self.out_dir, str(path + "metrics.csv")),
            index=False,
        )
