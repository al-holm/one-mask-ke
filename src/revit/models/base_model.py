from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch.nn.utils.parametrize import (
    is_parametrized,
    register_parametrization,
    remove_parametrizations,
)

from .weight_mask import ElementwiseMask


class BaseLLMModel(ABC):
    """Abstract base class for causal LLMs."""

    def __init__(self, model_name: str, target_layer: int, is_rome=True):
        self.model_name = model_name
        self.target_layer = target_layer
        llm, tokenizer, activation_attr, mlp_out_attr, memit_layers = self._init_model(
            target_layer
        )
        self.memit_layers = memit_layers
        self.llm = llm
        self.tokenizer = tokenizer
        self.activation_attr = activation_attr
        self.mlp_out_attr = mlp_out_attr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.to(self.device).eval()
        self.original_weight = self._get_target_weight().clone()
        if not is_rome:
            self.original_memit = {}
            self.save_original_memit()

    @staticmethod
    @abstractmethod
    def _init_model(target_layer: int):
        """Return (llm, tokenizer, activation_attr, mlp_out_attr)."""
        pass

    @abstractmethod
    def _get_target_weight(self) -> torch.nn.Parameter:
        """Return the target MLP weight parameter."""
        pass

    def set_weight(self, weight: torch.Tensor, layer: int = None):
        """Replace the MLP weight at the given layer."""
        mod = self.mlp_out_attr
        with torch.no_grad():
            if is_parametrized(mod, "weight"):
                mod.parametrizations.weight.original.copy_(weight)
            else:
                mod.weight.copy_(weight)

    def set_weights_memit(self, weight_dict: Dict[int, torch.Tensor]):
        """Replace the MLP weight at the given layer with MEMIT weights."""
        for layer in self.memit_layers:
            weight = weight_dict[layer]
            self._set_weight_to_layer(weight, layer)

    def save_original_memit(self):
        for layer in self.memit_layers:
            weight = self._get_target_weight(layer)
            self.original_memit[layer] = weight.clone()

    def reset_weights_memit(self):
        if len(self.original_memit.keys()) == 0:
            return
        for layer in self.memit_layers:
            weight = self.original_memit[layer]
            self._set_weight_to_layer(weight, layer)

    def get_activation(self, prompt: str):
        """
        Get the activation of the model for the given input.
        :param prompt: The input prompt.
        :return: The activation of the model.
        """
        activation = None

        def hook_fn(module, input_tensor, output_tensor):
            nonlocal activation
            gelu_output = torch.nn.functional.gelu(output_tensor)
            activation = gelu_output.detach().cpu()

        hook = self.activation_attr.register_forward_hook(hook_fn)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            self.device
        )
        with torch.no_grad():
            self.llm(input_ids)
        hook.remove()
        return activation

    def evaluate_rome_rsr(self, prompt: str, true_object: str, new_object: str) -> Dict:
        """
        Evaluate ROME edit reversal by comparing probabilities of true vs new object.

        Args:
            prompt: The input prompt (e.g., "The capital of France is")
            true_object: The original/correct answer (e.g., "Paris")
            new_object: The edited answer (e.g., "London")
        """
        lp_true = self._score_candidate_sequence(prompt, true_object)
        lp_new = self._score_candidate_sequence(prompt, new_object)

        probs = self.get_next_token_probabilities(prompt)
        top_predictions = self.generate_topk_predictions(probs=probs)

        return {
            "true_object_prob": lp_true,
            "new_object_prob": lp_new,
            "prediction": top_predictions,
            "rsr": lp_true > lp_new,
            "rsm": lp_true - lp_new,
        }

    def get_next_token_probabilities(self, prompt: str) -> torch.Tensor:
        """
        Get probabilities for tokens at the next position.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.llm(**inputs)
            logits = outputs.logits[
                0, -1, :
            ]  # Last token logits (because of autoregressive model, we'd like to predict the next token after the prompt)
            probs = torch.log_softmax(logits, dim=-1)
        return probs

    def _score_candidate_sequence(self, prompt: str, object_str: str) -> float:
        """
        Returns the summed log-probability over all candidate tokens.
        eg helpful for multitoken obj like new york
        """
        # Ensure there is a space between prompt and object if missing
        full_text = prompt + object_str
        if not prompt.endswith(" ") and not object_str.startswith(" "):
            full_text = prompt + " " + object_str

        enc_prompt = self.tokenizer(prompt, return_tensors="pt").to(
            self.device
        )  # [1, T]
        enc_full = self.tokenizer(full_text, return_tensors="pt").to(  # [1, T_full]
            self.device
        )

        with torch.no_grad():
            out = self.llm(**enc_full)
            logprobs = torch.log_softmax(out.logits, dim=-1)  # [1, T_full, V]

        T_prompt = enc_prompt["input_ids"].size(1)  # T
        cand_ids = enc_full["input_ids"][0, T_prompt:]  # # [T_full - T]

        # each token at position i is predicted from logits at i-1
        positions = torch.arange(
            T_prompt - 1, T_prompt + cand_ids.size(0) - 1, device=self.device
        )
        lp = logprobs[
            0, positions, cand_ids.to(self.device)
        ]  # pick entries for last position/s for token_ids from enc_full
        return float(lp.sum().item())

    def generate_topk_predictions(self, probs: torch.Tensor, top_k=10) -> str:
        """
        Generate prediction for the next tokens.
        """
        prob, idx = torch.topk(probs, top_k)
        decoded = [self.tokenizer.decode([i]) for i in idx.tolist()]
        pairs = [(d.strip(), float(p)) for d, p in zip(decoded, prob.tolist())]
        return pairs

    def attach_mask_parametrization(self, mask_getter):
        target = self.mlp_out_attr
        if is_parametrized(target, "weight"):
            remove_parametrizations(target, "weight", leave_parametrized=False)
        register_parametrization(target, "weight", ElementwiseMask(mask_getter))

        target.parametrizations.weight.original.requires_grad_(False)
