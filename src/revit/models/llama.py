import torch
from torch.nn.utils.parametrize import (
    is_parametrized,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_model import BaseLLMModel
from .model_enum import ModelName

MODEL_NAME = ModelName.LLAMA3_3B


class LLamaModel(BaseLLMModel):
    def __init__(self, target_layer=5, is_rome: bool = False):
        super().__init__(MODEL_NAME, target_layer, is_rome)

    @staticmethod
    def _init_model(target_layer: int = 5):
        """Instantiate self.model and self.tokenizer and target activation attribute."""
        print(f"Loading LLaMA-3 3B model with target layer {target_layer}...")
        llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        if tokenizer.eos_token is None:
            tokenizer.eos_token_id = 2
            tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        activation_attr = llm.model.layers[target_layer].mlp.gate_proj
        mlp_out_attr = llm.model.layers[target_layer].mlp.down_proj
        memit_layers = [4, 5, 6, 7, 8]
        return llm, tokenizer, activation_attr, mlp_out_attr, memit_layers

    def _get_target_weight(self, layer: int = None):
        if layer is not None:
            return getattr(self.llm.model.layers[layer].mlp.down_proj, "weight")
        return getattr(self.llm.model.layers[self.target_layer].mlp.down_proj, "weight")

    def _set_weight_to_layer(self, weight: torch.Tensor, layer: int):
        """Replace the MLP weight at the given layer."""
        mod = self.llm.model.layers[layer].mlp.down_proj
        with torch.no_grad():
            if is_parametrized(mod, "weight"):
                mod.parametrizations.weight.original.copy_(weight)
            else:
                mod.weight.copy_(weight)
