import torch
from torch.nn.utils.parametrize import (
    is_parametrized,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .base_model import BaseLLMModel
from .model_enum import ModelName

MODEL_NAME = ModelName.GPT2XL


class GPT2XLModel(BaseLLMModel):
    def __init__(self, target_layer: int = 17, is_rome: bool = False):
        super().__init__(MODEL_NAME, target_layer, is_rome)

    @staticmethod
    def _init_model(target_layer: int = 17):
        """Instantiate self.model and self.tokenizer and target activation attribute."""
        assert target_layer == 17
        llm = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-xl")
        tokenizer = GPT2Tokenizer.from_pretrained(
            "openai-community/gpt2-xl", add_prefix_space=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        llm.config.pad_token_id = tokenizer.pad_token_id
        activation_attr = llm.transformer.h[target_layer].mlp.c_fc
        mlp_out_attr = llm.transformer.h[target_layer].mlp.c_proj
        memit_layers = [13, 14, 15, 16, 17]
        return llm, tokenizer, activation_attr, mlp_out_attr, memit_layers

    def _get_target_weight(self, layer: int = None) -> torch.nn.Parameter:
        """Return the target MLP weight parameter."""
        if layer is not None:
            return getattr(self.llm.transformer.h[layer].mlp.c_proj, "weight")
        return getattr(self.llm.transformer.h[self.target_layer].mlp.c_proj, "weight")

    def _set_weight_to_layer(self, weight: torch.Tensor, layer: int):
        """Replace the MLP weight at the given layer."""
        mod = self.llm.transformer.h[layer].mlp.c_proj
        with torch.no_grad():
            if is_parametrized(mod, "weight"):
                mod.parametrizations.weight.original.copy_(weight)
            else:
                mod.weight.copy_(weight)
