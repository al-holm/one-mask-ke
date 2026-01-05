import math
from abc import ABC, abstractmethod

import torch

from ..models import MaskedModel
from .loss import Loss, RestorationLoss, SparsityLoss


class MaskTrainer(ABC):
    def __init__(
        self,
        model: MaskedModel,
        n_epochs: int,
        data: dict,
        auxiliary_loss: Loss = None,
        restoration_loss: RestorationLoss = None,
        sparsity_loss: SparsityLoss = None,
        use_sigmoid: bool = True,
    ):
        """
        Args
        model (MaskedModel): The model to be pruned.
        n_epochs (int) : number of iterations to train
        dataset (list): Dataset containing paraphrases and target prompt.
        target_prompt (dict): The target prompt for the model.
        """
        self.max_iterations = n_epochs
        self.model = model
        self.device = model.device
        self.auxiliary_loss = auxiliary_loss
        self.restoration_loss = restoration_loss or RestorationLoss()
        self.sparsity_loss = sparsity_loss or SparsityLoss()
        self.data = self.build_dataset(data)
        self.init_mean = 0.85
        self.init_std = 0.1
        self.theta = torch.nn.Parameter(
            torch.zeros_like(self.model.original_weight, device=self.device)
        )
        torch.nn.init.normal_(self.theta, mean=self.init_mean, std=self.init_std)
        self.mask = torch.sigmoid(self.theta)

        self.optimizer = torch.optim.AdamW(
            [self.theta], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0
        )
        self.current_iteration = 0
        self.sigmoid_tau_initial = 6.0
        self.tau_decay_rate = 3.0
        self.lambda_s0 = 1e-4
        self.lambda_s_max = 100
        self.from_checkpoint = False
        self.use_sigmoid = use_sigmoid

    @abstractmethod
    def compute_loss(self, batch):
        pass

    @abstractmethod
    def train(self):
        pass

    def build_dataset(self, record: list) -> list:
        data = []
        for i in record:
            rec = self.build_record(i)
            rec["case_id"] = i["case_id"]
            rec["rel_id"] = i["requested_rewrite"]["relation_id"]
            true_str = rec["true_object"]
            new_str = rec["new_object"]
            tok = self.model.tokenizer(
                rec["data"],
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            attn = tok["attention_mask"]
            last_pos = attn.sum(dim=1) - 1
            first_true = self._get_target_idx(rec["data"], attn, true_str)
            first_new = self._get_target_idx(rec["data"], attn, new_str)

            with torch.no_grad():
                rec["_batched_inputs"] = {
                    "input_ids": tok["input_ids"].to(self.device, non_blocking=True),
                    "attention_mask": attn.to(self.device, non_blocking=True),
                    "last_pos": last_pos.to(self.device, non_blocking=True),
                    "token_id_true": first_true.to(
                        self.device, non_blocking=True, dtype=torch.long
                    ),
                    "token_id_new": first_new.to(
                        self.device, non_blocking=True, dtype=torch.long
                    ),
                }
                if (
                    self.auxiliary_loss
                    and self.auxiliary_loss.__class__.__name__ == "KLDivergenceLoss"
                ):
                    rec["ref_logits"] = self._compute_ref_scores(rec["data"]).detach()
            data.append(rec)
        return data

    def build_record(self, record: dict) -> dict:
        req = record["requested_rewrite"]
        prompt = req["prompt"].format(req["subject"])

        new_obj = req["target_new"]["str"]
        true_obj = req["target_true"]["str"]
        return {
            "target_prompt": prompt,
            "new_object": new_obj,
            "true_object": true_obj,
            "data": [prompt] + record["paraphrase_prompts"],
        }

    def anneal_tau(self, from_checkpoint=False):
        return self.sigmoid_tau_initial * math.exp(
            -self.tau_decay_rate * self.current_iteration / self.max_iterations
        )

    def update_lambda(self):
        t = self.current_iteration / max(1, self.max_iterations - 1)
        return self.lambda_s0 + t * (self.lambda_s_max - self.lambda_s0)

    def compute_mask(self, tau):  # sigmoid with temperature
        if self.use_sigmoid:
            self.mask = torch.sigmoid(self.theta / tau)
        else:
            self.mask = self.theta

    def _get_target_idx(
        self, prompts: list[str], attn_prompt: torch.Tensor, target: str
    ) -> torch.Tensor:
        texts = [(p if p.endswith(" ") else p + " ") + target for p in prompts]
        tok_pt = self.model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        Tp = attn_prompt.sum(dim=1)  # [B, last_token_id]
        row = torch.arange(Tp.size(0))
        first_ids = tok_pt["input_ids"][row, Tp]  # [B, last_token_id]
        return first_ids

    def _compute_ref_scores(self, prompts: list[str]) -> torch.Tensor:
        tok = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.llm(**tok, use_cache=False)
            B = tok["input_ids"].size(0)
            row = torch.arange(B, device=self.device)
            ref_logits = out.logits[
                row, tok["attention_mask"].sum(dim=1) - 1, :
            ]  # [B, V]
        return ref_logits
