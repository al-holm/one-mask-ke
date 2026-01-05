import random
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from ..models import MaskedModel
from .delta_loader import FP16DeltaLoader
from .loss import Loss, RestorationLoss, SparsityLoss
from .mask_trainer_abc import MaskTrainer


class SharedMaskTrainer(MaskTrainer):
    def __init__(
        self,
        model: MaskedModel,
        n_epochs: int,
        target_record: dict,
        batch_size: int,
        weight_mode: Literal["full", "delta_fp16"] = "delta_fp16",
        prefix="1500",
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
        self.batch_size = batch_size
        self.weights_dir = Path(
            f"../output/edited_weights/{str(model.model_name)}_{prefix}/"
        )
        self._cpu_weight_cache = {}
        self.weight_mode = weight_mode
        self.weight_mode = weight_mode

        # Split data
        random.shuffle(target_record)
        split_idx = int(len(target_record) * 0.9)
        train_data = target_record[:split_idx]
        self.val_raw_data = target_record[split_idx:]

        super().__init__(
            model=model,
            n_epochs=n_epochs,
            data=train_data,
            auxiliary_loss=auxiliary_loss,
            restoration_loss=restoration_loss,
            sparsity_loss=sparsity_loss,
            use_sigmoid=use_sigmoid,
        )
        self.val_data = self.build_dataset(self.val_raw_data)
        self.model.attach_mask_parametrization(
            lambda: self.mask,
            use_weight_getter=True,  # weight getter for conditional forward pass
        )
        self.ckpt_dir = Path("./checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.loss_tracker = []
        self.delta_loader = None
        self.delta_margin = 2
        self.sigmoid_tau_initial = 6.0
        self.tau_decay_rate = 3.0
        self.lambda_s0 = 1e-6
        self.lambda_s_max = 60
        if self.weight_mode == "delta_fp16":
            self.delta_loader = FP16DeltaLoader(
                device=self.device,
                model_name=self.model.model_name,
            )
        self.prefix = prefix

    def compute_loss(self, batch):
        losses_r = []
        losses_aux = []
        for rec in batch:
            with torch.cuda.amp.autocast(enabled=False):
                edited_weight = self.load_weight_for(rec["case_id"], rec["rel_id"])
                self.model.set_current_weight(edited_weight)

            term_r, student_logits = self.restoration_loss(
                self.model, rec, self.current_iteration
            )
            losses_r.append(term_r)

            if self.auxiliary_loss:
                aux_loss = self.auxiliary_loss(
                    student_logits, rec, self.current_iteration
                )
                losses_aux.append(aux_loss)

        restoration_loss = torch.stack(losses_r).mean()
        sparsity_loss = self.sparsity_loss(self.mask)

        if not self.auxiliary_loss:
            return restoration_loss, sparsity_loss, None
        losses_aux = torch.stack(losses_aux).mean()
        return restoration_loss, sparsity_loss, losses_aux

    def train(self):
        N = len(self.data)
        idxs = list(range(N))
        progress_bar = tqdm(
            range(self.current_iteration, self.max_iterations), desc="Training Mask"
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        for self.current_iteration in progress_bar:
            random.shuffle(idxs)
            lambda_scalar = self.update_lambda()
            tau = self.anneal_tau(self.from_checkpoint)
            rl_sum = sl_sum = p_sum = 0.0
            n_batches = 0

            for s in range(0, N, self.batch_size):
                self.compute_mask(tau)
                self._current_tau = tau
                batch = [self.data[i] for i in idxs[s : s + self.batch_size]]

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    restoration_loss, sparsity_loss, aux_loss = self.compute_loss(batch)
                    reg = restoration_loss + lambda_scalar * sparsity_loss
                    if self.auxiliary_loss:
                        reg += aux_loss * self.auxiliary_loss.beta

                self.optimizer.zero_grad(set_to_none=True)
                reg.backward()
                torch.nn.utils.clip_grad_norm_([self.theta], 1.0)
                self.optimizer.step()

                rl_sum += restoration_loss.item()
                sl_sum += sparsity_loss.item()
                p_sum += aux_loss.item() if self.auxiliary_loss else 0.0
                n_batches += 1

            rl_epoch = rl_sum / max(1, n_batches)
            sl_epoch = sl_sum / max(1, n_batches)
            p_epoch = p_sum / max(1, n_batches)

            # Validation
            if self.val_data:
                val_rsr = self.compute_valuation_rsr() or 0.0
            progress_bar.set_postfix(
                {
                    "Restoration": f"{rl_epoch:.4f}",
                    "L_aux": f"{p_epoch:.4f}",
                    "Sparsity": f"{sl_epoch:.5f}",
                    "λ": f"{lambda_scalar:.2e}",
                    "Val RSR": f"{val_rsr:.4f}",
                }
            )
            self.loss_tracker.append(
                {
                    "L_rest": rl_epoch,
                    "L_spar": sl_epoch,
                    "L_aux": p_epoch,
                    "lambda": lambda_scalar,
                    "epoch": self.current_iteration + 1,
                    "val_rsr": val_rsr or 0.0,
                }
            )
            if (self.current_iteration) % 50 == 0:
                self.save_ckpt(self.current_iteration)
        return self.mask

    def train_step(self, idxs, N):
        random.shuffle(idxs)
        lambda_scalar = self.update_lambda()
        tau = self.anneal_tau(self.from_checkpoint)
        rl_sum = sl_sum = p_sum = 0.0
        n_batches = 0

        for s in range(0, N, self.batch_size):
            self.compute_mask(tau)
            self._current_tau = tau
            batch = [self.data[i] for i in idxs[s : s + self.batch_size]]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                restoration_loss, sparsity_loss, aux_loss = self.compute_loss(batch)
                reg = restoration_loss + lambda_scalar * sparsity_loss
                if self.auxiliary_loss:
                    reg += aux_loss * self.auxiliary_loss.beta

            self.optimizer.zero_grad(set_to_none=True)
            reg.backward()
            torch.nn.utils.clip_grad_norm_([self.theta], 1.0)
            self.optimizer.step()

            rl_sum += restoration_loss.item()
            sl_sum += sparsity_loss.item()
            p_sum += aux_loss.item() if self.auxiliary_loss else 0.0
            n_batches += 1

        rl_epoch = rl_sum / max(1, n_batches)
        sl_epoch = sl_sum / max(1, n_batches)
        p_epoch = p_sum / max(1, n_batches)

        self.loss_tracker.append(
            {
                "L_rest": rl_epoch,
                "L_spar": sl_epoch,
                "L_aux": p_epoch,
                "epoch": self.current_iteration + 1,
            }
        )

    def compute_valuation_rsr(self):
        rsr_scores = []
        for rec in self.val_data:
            edited_weight = self.load_weight_for(rec["case_id"], rec["rel_id"])
            self.model.set_current_weight(edited_weight)
            res = self.model.evaluate_rome_rsr(
                rec["target_prompt"], rec["true_object"], rec["new_object"]
            )
            rsr_scores.append(res["rsr"])

        val_rsr = sum(rsr_scores) / len(rsr_scores) if rsr_scores else 0.0
        return val_rsr

    def load_weight_for(
        self, case_id: str, rel_id: str, device: torch.device = None
    ) -> torch.Tensor:
        target_device = device if device is not None else self.device
        if self.weight_mode == "delta_fp16":
            # reconstruct W = W0 + ΔW_fp16 on device
            return self.delta_loader.reconstruct(case_id, rel_id, device=target_device)
        if self._cpu_weight_cache is not None and case_id in self._cpu_weight_cache:
            return self._cpu_weight_cache[case_id].to(target_device, non_blocking=True)
        wpath = self.weights_dir / f"{case_id}.pt"
        tens = torch.load(str(wpath), map_location="cpu").to(torch.float32).pin_memory()
        if self._cpu_weight_cache is not None:
            self._cpu_weight_cache[case_id] = tens
        return tens.to(target_device, non_blocking=True)

    def save_ckpt(self, epoch: int):
        if getattr(self, "rank", 0) != 0:
            return
        ckpt = {
            "theta": self.theta.detach().cpu(),
            "opt": self.optimizer.state_dict(),
            "rng_cpu": torch.random.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "loss_tracker": self.loss_tracker,
        }
        postfix = ""
        if not self.auxiliary_loss:
            postfix = "no_aux"
        path = str(
            self.ckpt_dir
            / f"{self.prefix}{postfix}_{self.model.model_name}_mask_ep{epoch:04d}.pt"
        )
        torch.save(
            ckpt,
            path,
        )

    def load_ckpt(self, path: str) -> int:
        ckpt = torch.load(path, map_location="cpu")
        self.theta.data.copy_(ckpt["theta"].to(self.device))
        self.optimizer.load_state_dict(ckpt["opt"])
        torch.random.set_rng_state(ckpt["rng_cpu"])
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        self.current_iteration = int(ckpt.get("current_iteration", 0))
        self.loss_tracker = ckpt.get("loss_tracker", [])
        self.from_checkpoint = True
