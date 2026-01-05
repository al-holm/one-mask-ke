from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch
import torch.nn.functional as F


class Loss(Protocol):
    def __call__(
        self, student_logits: torch.Tensor, record: Dict[str, Any], epoch: int
    ) -> torch.Tensor: ...


@dataclass
class RestorationLoss:
    delta_margin: float = 3.0

    def restoration_loss_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=-1)
        lt = logp[:, 0]
        ln = logp[:, 1]
        margin = -(lt - ln)
        return torch.nn.functional.relu(margin + self.delta_margin)

    def __call__(
        self, model: Any, record: Dict[str, Any], epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = record["_batched_inputs"]
        ids = inp["input_ids"]
        attn = inp["attention_mask"]
        last = inp["last_pos"]  # shape [B]

        out = model.llm(
            input_ids=ids, attention_mask=attn, use_cache=False
        )  # [B, T, V]
        B = ids.size(0)
        row = torch.arange(B, device=ids.device)
        student_logits = out.logits[row, last, :]  # [B, V]

        logits_last_idx = torch.arange(student_logits.size(0), device=ids.device)
        z_true = student_logits[logits_last_idx, inp["token_id_true"]]
        z_new = student_logits[logits_last_idx, inp["token_id_new"]]
        z2 = torch.stack((z_true, z_new), dim=1)  # [B,2]

        return self.restoration_loss_from_logits(z2).mean(), student_logits


@dataclass
class SparsityLoss:
    sparsity_max: float = 0.1

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(((1.0 - mask).mean()) - self.sparsity_max, 0.0)


@dataclass
class NLLLoss:
    """
    Cross-entropy on the ground-truth 'original' token.
    """

    true_token_key: str = "token_id_true"
    reduction: str = "mean"

    def __call__(
        self, student_logits: torch.Tensor, record: Dict[str, Any], epoch: int
    ) -> torch.Tensor:
        inp = record["_batched_inputs"]
        target = inp[self.true_token_key]  # shape [B]
        return F.cross_entropy(student_logits.float(), target, reduction=self.reduction)


@dataclass
class KLDivergenceLoss:
    """
    KL(teacher || student) with temperature
    """

    temperature: float = 1.0
    max_temperature: float = 3.0
    schedule: str = "linear"
    reduction: str = "batchmean"
    detach_teacher: bool = True
    beta: float = 2.0
    total_epochs: int = 300

    def get_temperature(self, epoch: int) -> float:
        """Return scheduled temperature for this epoch."""
        if self.schedule == "linear":
            frac = min(epoch / self.total_epochs, 1.0)
            return self.temperature + frac * (self.max_temperature - self.temperature)
        elif self.schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * min(epoch / self.total_epochs, 1.0)))
            return self.temperature + frac * (self.max_temperature - self.temperature)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule}")

    def __call__(
        self, student_logits: torch.Tensor, record: Dict[str, Any], epoch: int
    ) -> torch.Tensor:
        T = self.get_temperature(epoch)

        log_p = F.log_softmax(student_logits / T, dim=-1)

        if "ref_logits" in record and record["ref_logits"] is not None:
            ref_logits = record["ref_logits"]
        else:
            raise NotImplementedError("Teacher logits not found.")

        q = F.softmax(ref_logits / T, dim=-1)
        if self.detach_teacher:
            q = q.detach()
        return F.kl_div(log_p, q, reduction=self.reduction) * (T**2)


@dataclass
class DummySparsityLoss:
    def __call__(self, mask: torch.Tensor):
        return torch.tensor(0.0, device=mask.device, requires_grad=True)



@dataclass
class DummyRestorationLoss:
    def __call__(self, model: Any, record: Dict[str, Any], epoch: int):
        """
        Return 0 restoration loss but preserve student_logits so KL works.
        """
        inp = record["_batched_inputs"]
        ids = inp["input_ids"]
        attn = inp["attention_mask"]
        last = inp["last_pos"]  # shape [B]

        out = model.llm(
            input_ids=ids, attention_mask=attn, use_cache=False
        )  # [B, T, V]
        B = ids.size(0)
        row = torch.arange(B, device=ids.device)
        student_logits = out.logits[row, last,:]
        return torch.tensor(0.0, device=model.device, requires_grad=True), student_logits